if __name__ == "__main__":
    import os
    import pathlib
    import sys

    root_dir = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(root_dir)
    os.chdir(root_dir)

import copy
import os
import pathlib
import random
import time

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from termcolor import cprint

from common.checkpoint_util import TopKCheckpointManager
from common.json_logger import JsonLogger
from common.pytorch_util import dict_apply, optimizer_to
from dataset.base_dataset import BaseImageDataset
from env_runner.base_runner import BaseRunner
from model.common.lr_scheduler import get_scheduler
from model.diffusion.ema_model import EMAModel
from policy.diffusion_image_policy import DiffusionImagePolicy
from workspace.base_workspace import BaseWorkspace

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


class DPWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: DiffusionImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: DiffusionImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
        self.global_step = 0
        self.epoch = 0

    def _init_wandb_run(self, cfg):
        if wandb is None:
            cprint("[WandB] wandb is not installed, falling back to no-op logging.", "yellow")
            return _NoOpWandbRun()
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})
        return wandb_run

    def _load_latest_checkpoint_if_available(self, cfg):
        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        self._load_latest_checkpoint_if_available(cfg)

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        ema = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        env_runner: BaseRunner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=self.output_dir)

        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        wandb_run = self._init_wandb_run(cfg)

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        train_sampling_batch = None
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        run_validation = False
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for _ in tqdm.tqdm(range(self.epoch, cfg.training.num_epochs), desc="Training"):
                step_log = {}
                train_losses = []
                for batch_idx, batch in enumerate(train_dataloader):
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                    raw_loss = self.model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()

                    if cfg.training.use_ema:
                        ema.step(self.model)

                    raw_loss_cpu = raw_loss.item()
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        "train_loss": raw_loss_cpu,
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    is_last_batch = batch_idx == (len(train_dataloader) - 1)
                    if not is_last_batch:
                        wandb_run.log(step_log, step=self.global_step)
                        json_logger.log(step_log)
                        self.global_step += 1
                    if (
                        cfg.training.max_train_steps is not None
                        and batch_idx >= (cfg.training.max_train_steps - 1)
                    ):
                        break

                train_loss = float(np.mean(train_losses))
                step_log["train_loss"] = train_loss

                policy = self.ema_model if cfg.training.use_ema else self.model
                policy.eval()

                if (self.epoch % cfg.training.val_every) == 0 and run_validation:
                    with torch.no_grad():
                        val_losses = []
                        for batch_idx, batch in enumerate(val_dataloader):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            val_losses.append(self.model.compute_loss(batch))
                            if (
                                cfg.training.max_val_steps is not None
                                and batch_idx >= (cfg.training.max_val_steps - 1)
                            ):
                                break
                        if len(val_losses) > 0:
                            step_log["val_loss"] = torch.mean(torch.tensor(val_losses)).item()

                if (self.epoch % cfg.training.sample_every) == 0 and train_sampling_batch is not None:
                    with torch.no_grad():
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        result = policy.predict_action(batch["obs"])
                        mse = torch.nn.functional.mse_loss(result["action_pred"], batch["action"])
                        step_log["train_action_mse_error"] = mse.item()

                if (
                    env_runner is not None
                    and cfg.training.rollout_every > 0
                    and (self.epoch % cfg.training.rollout_every) == 0
                ):
                    runner_log = env_runner.run(policy)
                    step_log.update(runner_log)

                step_log.setdefault("test_mean_score", -train_loss)
                completed_epoch = self.epoch + 1
                is_checkpoint_epoch = (
                    completed_epoch % cfg.training.checkpoint_every == 0
                    or completed_epoch == cfg.training.num_epochs
                )
                if cfg.checkpoint.save_last_ckpt and is_checkpoint_epoch:
                    saved_epoch = self.epoch
                    self.epoch = completed_epoch
                    try:
                        self.save_checkpoint()
                        if completed_epoch % cfg.training.checkpoint_every == 0:
                            periodic_ckpt_path = pathlib.Path(self.output_dir).joinpath(
                                "checkpoints", f"{completed_epoch:04d}.ckpt"
                            )
                            self.save_checkpoint(path=periodic_ckpt_path)
                        metric_dict = {}
                        for key, value in step_log.items():
                            metric_dict[key.replace("/", "_")] = value
                        metric_dict["epoch"] = completed_epoch
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)
                    finally:
                        self.epoch = saved_epoch

                policy.train()
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        wandb_run.finish()

    def run_robot(self):
        cfg = copy.deepcopy(self.cfg)
        latest_ckpt_path = self.get_checkpoint_path(tag="latest")
        if latest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {latest_ckpt_path}", "magenta")
            self.load_checkpoint(path=latest_ckpt_path)

        env_runner: BaseRunner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=self.output_dir)
        policy = self.ema_model if cfg.training.use_ema else self.model
        policy.eval()
        device = torch.device(cfg.training.device)
        policy.to(device)
        if hasattr(env_runner, "run_robot"):
            env_runner.run_robot(policy)
        else:
            raise NotImplementedError("Configured env_runner does not implement run_robot().")
