import copy

import cv2
import numpy as np
import torch

from common.pytorch_util import dict_apply
from common.replay_buffer import ReplayBuffer
from common.sampler import SequenceSampler, downsample_mask, get_val_mask
from dataset.base_dataset import BaseImageDataset
from model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


class UR5EMultiImageDataset(BaseImageDataset):
    def __init__(
        self,
        zarr_path,
        camera_keys,
        image_size=(224, 224),
        agent_pos_key="robot_eef_pose",
        action_key="cartesian_action",
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        task_name=None,
    ):
        super().__init__()
        self.task_name = task_name
        self.camera_keys = list(camera_keys)
        self.image_size = tuple(image_size)
        self.agent_pos_key = agent_pos_key
        self.action_key = action_key

        buffer_keys = [self.agent_pos_key, self.action_key] + self.camera_keys
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=buffer_keys)
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = downsample_mask(mask=~val_mask, max_n=max_train_episodes, seed=seed)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer[self.action_key],
            "agent_pos": self.replay_buffer[self.agent_pos_key],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        for camera_key in self.camera_keys:
            obs_key = self._camera_key_to_obs_key(camera_key)
            normalizer[obs_key] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key].astype(np.float32))

    def __len__(self):
        return len(self.sampler)

    def _camera_key_to_obs_key(self, camera_key):
        return camera_key[:-6] if camera_key.endswith("_color") else camera_key

    def _resize_image_sequence(self, images):
        target_h, target_w = self.image_size
        resized = [
            cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            for frame in images
        ]
        return np.stack(resized, axis=0).astype(np.float32)

    def _sample_to_data(self, sample):
        obs = {
            "agent_pos": sample[self.agent_pos_key].astype(np.float32),
        }
        for camera_key in self.camera_keys:
            obs_key = self._camera_key_to_obs_key(camera_key)
            obs[obs_key] = self._resize_image_sequence(sample[camera_key])
        return {
            "obs": obs,
            "action": sample[self.action_key].astype(np.float32),
        }

    def __getitem__(self, idx):
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)
