from typing import Dict
import torch
import numpy as np
import copy
from common.pytorch_util import dict_apply
from common.replay_buffer import ReplayBuffer
from common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from model.common.normalizer import LinearNormalizer
from dataset.base_dataset import BaseDataset


class RobotDataset(BaseDataset):
    def __init__(self,
            zarr_path,
            point_cloud_key='fused_pointcloud',
            agent_pos_key='robot_eef_pose',
            action_key='cartesian_action',
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
        self.point_cloud_key = point_cloud_key
        self.agent_pos_key = agent_pos_key
        self.action_key = action_key
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=[self.agent_pos_key, self.action_key, self.point_cloud_key],
        )
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
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
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer[self.action_key],
            'agent_pos': self.replay_buffer[self.agent_pos_key][...,:],
            'point_cloud': self.replay_buffer[self.point_cloud_key],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample[self.agent_pos_key].astype(np.float32)
        point_cloud = sample[self.point_cloud_key].astype(np.float32)

        data = {
            'obs': {
                'point_cloud': point_cloud,
                'agent_pos': agent_pos,
            },
            'action': sample[self.action_key].astype(np.float32)
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
