from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from common.model_util import print_params
from common.pytorch_util import dict_apply
from model.diffusion.conditional_unet1d import ConditionalUnet1D
from model.diffusion.mask_generator import LowdimMaskGenerator
from model.common.normalizer import LinearNormalizer
from model.vision.timm_obs_encoder import TimmObsEncoder
from policy.base_policy import BasePolicy


class DiffusionImagePolicy(BasePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        crop_shape=(76, 76),
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        condition_type="film",
        use_depth=False,
        use_depth_only=False,
        obs_encoder: TimmObsEncoder = None,
        **kwargs,
    ):
        super().__init__()
        del crop_shape

        self.use_depth = use_depth
        self.use_depth_only = use_depth_only
        self.rgb_keys = [
            key for key, attr in shape_meta["obs"].items() if attr.get("type", "low_dim") == "rgb"
        ]
        self.depth_keys = [
            key for key, attr in shape_meta["obs"].items() if attr.get("type", "low_dim") == "depth"
        ]

        action_shape = shape_meta["action"]["shape"]
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        if use_depth and not self.depth_keys:
            raise ValueError("use_depth=True but no depth keys are present in shape_meta.")
        if use_depth_only and not self.depth_keys:
            raise ValueError("use_depth_only=True but no depth keys are present in shape_meta.")

        if use_depth and not use_depth_only:
            for key in self.rgb_keys:
                shape_meta["obs"][key]["shape"][0] += 1
        if use_depth and use_depth_only:
            for key in self.rgb_keys:
                shape_meta["obs"][key]["shape"][0] = 1

        obs_feature_dim = np.prod(obs_encoder.output_shape())
        model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=obs_feature_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print_params(self)

    def _prepare_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        nobs = nobs.copy()
        for key in self.rgb_keys:
            nobs[key] = nobs[key] / 255.0
            if nobs[key].shape[-1] == 3:
                if len(nobs[key].shape) == 5:
                    nobs[key] = nobs[key].permute(0, 1, 4, 2, 3)
                elif len(nobs[key].shape) == 4:
                    nobs[key] = nobs[key].permute(0, 3, 1, 2)
            if self.use_depth:
                depth_key = f"{key}_depth"
                if depth_key not in nobs:
                    raise KeyError(f"Depth key {depth_key} is required when use_depth=True.")
                if self.use_depth_only:
                    nobs[key] = nobs[depth_key].unsqueeze(-3)
                else:
                    nobs[key] = torch.cat([nobs[key], nobs[depth_key].unsqueeze(-3)], dim=-3)
        return nobs

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        **kwargs,
    ):
        del kwargs
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self.model(
                trajectory,
                t,
                local_cond=local_cond,
                global_cond=global_cond,
            )
            trajectory = scheduler.step(
                model_output,
                t,
                trajectory,
                generator=generator,
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self._prepare_obs(obs_dict)
        value = next(iter(nobs.values()))
        batch_size, _ = value.shape[:2]
        horizon = self.horizon
        action_dim = self.action_dim
        obs_dim = self.obs_feature_dim
        obs_steps = self.n_obs_steps
        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :obs_steps, ...])
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
            cond_data = torch.zeros((batch_size, horizon, action_dim), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:, :obs_steps, ...])
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, obs_steps, -1)
            cond_data = torch.zeros((batch_size, horizon, action_dim + obs_dim), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :obs_steps, action_dim:] = nobs_features
            cond_mask[:, :obs_steps, action_dim:] = True

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        naction_pred = nsample[..., :action_dim]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        start = obs_steps - 1
        end = start + self.n_action_steps
        if end > action_pred.shape[1]:
            end = action_pred.shape[1]
            start = max(0, end - self.n_action_steps)
        action = action_pred[:, start:end]
        return {
            "action": action,
            "action_pred": action_pred,
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nobs = self._prepare_obs(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, : self.n_obs_steps, ...])
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            nobs_features = self.obs_encoder(nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        pred = self.model(
            noisy_trajectory,
            timesteps,
            local_cond=local_cond,
            global_cond=global_cond,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        return loss.mean()
