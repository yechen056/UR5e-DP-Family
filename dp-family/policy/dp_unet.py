from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from common.pytorch_util import dict_apply
from model.common.normalizer import LinearNormalizer
from model.diffusion.conditional_unet1d import ConditionalUnet1D
from model.vision.sensory_encoder import BaseSensoryEncoder
from policy.base_policy import BasePolicy


class DiffusionUnetPolicy(BasePolicy):
    def __init__(
        self,
        shape_meta: Dict,
        noise_scheduler: DDPMScheduler,
        obs_encoders: List[BaseSensoryEncoder],
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: Optional[int] = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: Tuple[int] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        **kwargs,
    ):
        super().__init__()

        modalities = ["robot_state"] + list(
            set(
                modality
                for obs_encoder in obs_encoders
                for modality in obs_encoder.modalities()
            )
        )

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta["obs"]
        obs_config = {key: [] for key in modalities}
        obs_key_shapes = {}
        obs_ports = []
        for key, attr in obs_shape_meta.items():
            shape = attr["shape"]
            obs_key_shapes[key] = list(shape)
            obs_type = attr["type"]
            if obs_type in modalities:
                obs_config[obs_type].append(key)
                obs_ports.append(key)

        obs_encoder = self.create_observation_encoder(
            obs_encoders=obs_encoders,
            modalities=modalities,
            obs_config=obs_config,
            obs_key_shapes=obs_key_shapes,
        )

        obs_feature_dim = sum(obs_encoder.output_feature_dim().values())
        global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.modalities = modalities
        self.obs_key_shapes = obs_key_shapes
        self.obs_ports = obs_ports
        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def _prepare_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        nobs = nobs.copy()
        for key, attr in self.obs_key_shapes.items():
            if len(attr) == 3:
                value = nobs[key]
                # Convert image observations from HWC / THWC to CHW / TCHW.
                if value.shape[-1] in (1, 3):
                    if len(value.shape) == 5:
                        value = value.permute(0, 1, 4, 2, 3)
                    elif len(value.shape) == 4:
                        value = value.permute(0, 3, 1, 2)
                nobs[key] = value / 255.0
        return nobs

    def create_observation_encoder(
        self,
        obs_encoders: List[BaseSensoryEncoder],
        modalities: List[str],
        obs_config,
        obs_key_shapes,
    ):
        class ObservationEncoder(BaseSensoryEncoder):
            def __init__(self):
                super().__init__()
                self.obs_encoders = nn.ModuleList(obs_encoders)
                self.modalities_ = modalities

            def forward(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
                features = {}
                for obs_encoder in self.obs_encoders:
                    features.update(obs_encoder(obs_dict))
                for key in obs_config["robot_state"]:
                    features[key] = obs_dict[key]
                return features

            def output_feature_dim(self) -> Dict[str, int]:
                output_feature_dim = {}
                for obs_encoder in obs_encoders:
                    output_feature_dim.update(obs_encoder.output_feature_dim())
                for key in obs_config["robot_state"]:
                    output_feature_dim[key] = obs_key_shapes[key][-1]
                return output_feature_dim

            def modalities(self) -> List[str]:
                return self.modalities_

        return ObservationEncoder()

    def conditional_sample(self, global_cond, generator=None, **kwargs):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=(len(global_cond), self.horizon, self.action_dim),
            dtype=global_cond.dtype,
            device=global_cond.device,
            generator=generator,
        )

        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            trajectory = scheduler.step(
                model(trajectory, t, global_cond=global_cond),
                t,
                trajectory,
                generator=generator,
                **kwargs,
            ).prev_sample
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self._prepare_obs(obs_dict)
        value = next(iter(nobs.values()))
        batch_size, obs_steps = value.shape[:2]
        del obs_steps

        features = self.obs_encoder(
            dict_apply(
                nobs,
                lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
            )
        )
        features = dict_apply(features, lambda x: x.reshape(batch_size, -1))
        global_cond = torch.cat(list(features.values()), dim=-1)
        nsample = self.conditional_sample(global_cond=global_cond, **self.kwargs)

        action_pred = self.normalizer["action"].unnormalize(nsample[..., : self.action_dim])
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {
            "action": action,
            "action_pred": action_pred,
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nobs = self._prepare_obs(batch["obs"])
        trajectory = self.normalizer["action"].normalize(batch["action"])
        batch_size = trajectory.shape[0]

        features = self.obs_encoder(
            dict_apply(
                nobs,
                lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
            )
        )
        features = dict_apply(features, lambda x: x.reshape(batch_size, -1))
        global_cond = torch.cat(list(features.values()), dim=-1)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        pred = self.model(noisy_trajectory, timesteps, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        assert pred_type == "epsilon", "Only epsilon prediction is supported"
        return F.mse_loss(pred, noise)
