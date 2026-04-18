if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
from workspace.base_workspace import BaseWorkspace
from dataset.base_dataset import BaseDataset
from env_runner.base_runner import BaseRunner
from policy.diffusion_pointcloud_policy import DiffusionPointcloudPolicy
from common.checkpoint_util import TopKCheckpointManager
from common.json_logger import JsonLogger
from common.pytorch_util import dict_apply, optimizer_to
from model.diffusion.ema_model import EMAModel
from model.common.lr_scheduler import get_scheduler

try:
    import wandb
except ImportError:
    wandb = None

OmegaConf.register_new_resolver("eval", eval, replace=True)


class _NoOpWandbRun:
    def log(self, *args, **kwargs):
        return None

    def finish(self):
        return None

class iDP3Workspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionPointcloudPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionPointcloudPolicy = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)


        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def _init_wandb_run(self, cfg):
        if wandb is None:
            cprint("[WandB] wandb is not installed, falling back to no-op logging.", "yellow")
            return _NoOpWandbRun()

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )
        return wandb_run

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        if cfg.training.debug:
            cfg.training.num_epochs = 40
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseDataset = hydra.utils.instantiate(cfg.task.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        env_runner: BaseRunner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        if env_runner is not None:
            assert isinstance(env_runner, BaseRunner)

        

        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
        wandb_run = self._init_wandb_run(cfg)

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None


        # training loop
        start_epoch = min(self.epoch, cfg.training.num_epochs)
        if start_epoch > 0:
            cprint(
                f"Resuming training from epoch {start_epoch} to {cfg.training.num_epochs}.",
                "yellow")
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            epoch_iter = tqdm.tqdm(
                range(start_epoch, cfg.training.num_epochs),
                desc=f"Training",
                initial=start_epoch,
                total=cfg.training.num_epochs)
            for local_epoch_idx in epoch_iter:
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                for batch_idx, batch in enumerate(train_dataloader):
                    t1 = time.time()
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x)
                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                
                    # compute loss
                    t1_1 = time.time()
                    raw_loss, loss_dict = self.model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()
                    
                    t1_2 = time.time()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    t1_3 = time.time()
                    # update ema
                    if cfg.training.use_ema:
                        ema.step(self.model)
                    t1_4 = time.time()
                    # logging
                    raw_loss_cpu = raw_loss.item()
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }
                    t1_5 = time.time()
                    step_log.update(loss_dict)
                    t2 = time.time()
                    
                    if verbose:
                        print(f"total one step time: {t2-t1:.3f}")
                        print(f" compute loss time: {t1_2-t1_1:.3f}")
                        print(f" step optimizer time: {t1_3-t1_2:.3f}")
                        print(f" update ema time: {t1_4-t1_3:.3f}")
                        print(f" logging time: {t1_5-t1_4:.3f}")

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        wandb_run.log(step_log, step=self.global_step)
                        json_logger.log(step_log)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()
                
                    
                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():                        
                        train_losses = list()
                        
                        for batch_idx, batch in enumerate(train_dataloader):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x)
                            obs_dict = batch['obs']
                            gt_action = batch['action']

                            result = policy.predict_action(obs_dict)
                            pred_action = result['action_pred']
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                            train_losses.append(mse.item())
                            
                            if (cfg.training.max_train_steps is not None) \
                                and batch_idx >= (cfg.training.max_train_steps-1):
                                break
                        train_loss = np.sum(train_losses)
                        # log epoch average validation loss
                        step_log['train_action_mse_error'] = train_loss
                        step_log['test_mean_score'] = - step_log['train_action_mse_error']
                        cprint(f"val loss: {train_loss:.7f}", "cyan")

                if (self.epoch % cfg.training.rollout_every) == 0 and env_runner is not None:
                    runner_log = env_runner.run(policy)
                    step_log.update(runner_log)


                # checkpoint
                completed_epoch = self.epoch + 1
                is_checkpoint_epoch = (
                    completed_epoch % cfg.training.checkpoint_every == 0
                    or completed_epoch == cfg.training.num_epochs
                )
                if is_checkpoint_epoch and cfg.checkpoint.save_ckpt:
                    # checkpointing
                    saved_epoch = self.epoch
                    self.epoch = completed_epoch
                    try:
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()
                            if completed_epoch % cfg.training.checkpoint_every == 0:
                                periodic_ckpt_path = pathlib.Path(self.output_dir).joinpath(
                                    'checkpoints', f'{completed_epoch:04d}.ckpt')
                                self.save_checkpoint(path=periodic_ckpt_path)
                        if cfg.checkpoint.save_last_snapshot:
                            self.save_snapshot()

                        # sanitize metric names
                        metric_dict = dict()
                        for key, value in step_log.items():
                            new_key = key.replace('/', '_')
                            metric_dict[new_key] = value
                        metric_dict['epoch'] = completed_epoch
                        
                        # We can't copy the last checkpoint here
                        # since save_checkpoint uses threads.
                        # therefore at this point the file might have been empty!
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)
                    finally:
                        self.epoch = saved_epoch
                    cprint("checkpoint saved.", "green")
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                
                self.global_step += 1
                self.epoch += 1
                del step_log

    def eval(self):
        cfg = copy.deepcopy(self.cfg)

        latest_ckpt_path = self.get_checkpoint_path(tag="latest")
        if latest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {latest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=latest_ckpt_path)

        env_runner: BaseRunner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseRunner)
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()

        runner_log = env_runner.run(policy)

        cprint(f"---------------- Eval Results --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')

    def run_robot(self):
        cfg = copy.deepcopy(self.cfg)

        latest_ckpt_path = self.get_checkpoint_path(tag="latest")
        if latest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {latest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=latest_ckpt_path)

        env_runner: BaseRunner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseRunner)
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()

        if hasattr(env_runner, 'run_robot'):
            env_runner.run_robot(policy)
        else:
            raise NotImplementedError('Configured env_runner does not implement run_robot().')
    
    def get_model(self):
        cfg = copy.deepcopy(self.cfg)
        
        tag = "latest"
        # tag = "best"
        lastest_ckpt_path = self.get_checkpoint_path(tag=tag)
        
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path)
        lastest_ckpt_path = str(lastest_ckpt_path)

        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model    
        policy.eval()

        return policy

        

@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)

def main(cfg):

    workspace = iDP3Workspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
