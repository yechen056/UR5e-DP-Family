import argparse
import shutil
from pathlib import Path
import sys

import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
DP3_ROOT = REPO_ROOT / 'dp-family'
if str(DP3_ROOT) not in sys.path:
    sys.path.insert(0, str(DP3_ROOT))

from common.replay_buffer import ReplayBuffer


KEEP_KEYS = {
    'timestamp',
    'robot_joint',
    'robot_joint_vel',
    'robot_eef_pose',
    'robot_eef_pose_vel',
    'cartesian_action',
    'joint_action',
    'stage',
    'teleop_fallback_used',
}


def flatten_episode_tree(tree):
    flat = {}
    stack = [tree]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                if isinstance(value, dict):
                    stack.append(value)
                else:
                    flat[key] = value
    return flat


def flatten_zarr_structure(buffer_path):
    obs_path = buffer_path / 'data' / 'observations'
    images_path = buffer_path / 'data' / 'images'
    data_path = buffer_path / 'data'

    if not obs_path.exists():
        return

    for item in obs_path.iterdir():
        dest_path = data_path / item.name
        if dest_path.exists():
            if dest_path.is_dir():
                shutil.rmtree(dest_path)
            else:
                dest_path.unlink()
        shutil.move(str(item), str(dest_path))

    if images_path.exists():
        for item in images_path.iterdir():
            dest_path = data_path / item.name
            if dest_path.exists():
                if dest_path.is_dir():
                    shutil.rmtree(dest_path)
                else:
                    dest_path.unlink()
            shutil.move(str(item), str(dest_path))
        try:
            images_path.rmdir()
        except OSError:
            pass

    try:
        obs_path.rmdir()
    except OSError:
        pass


def normalize_episode(episode):
    normalized = {}
    for key, value in episode.items():
        if key.startswith('camera_'):
            normalized[key] = value
            continue
        if key in KEEP_KEYS:
            normalized[key] = value

    if 'joint_action' not in normalized and 'robot_joint' in normalized:
        robot_joint = np.asarray(normalized['robot_joint'])
        if len(robot_joint) > 1:
            normalized['joint_action'] = robot_joint[1:]

    truncation_length = None
    if 'cartesian_action' in normalized:
        truncation_length = len(normalized['cartesian_action'])
    elif 'joint_action' in normalized:
        truncation_length = len(normalized['joint_action'])

    if truncation_length is not None:
        for key, value in list(normalized.items()):
            if hasattr(value, '__len__') and len(value) > truncation_length:
                normalized[key] = value[:truncation_length]

    for key, value in list(normalized.items()):
        if not hasattr(value, 'dtype'):
            continue
        if 'color' in key and value.dtype != np.uint8:
            if value.max() <= 1.0:
                normalized[key] = (value * 255).clip(0, 255).astype(np.uint8)
            else:
                normalized[key] = value.clip(0, 255).astype(np.uint8)
        elif value.dtype == np.float64:
            normalized[key] = value.astype(np.float32, copy=False)

    return normalized


def convert_dataset(raw_dir, output_zarr):
    raw_dir = Path(raw_dir)
    output_zarr = Path(output_zarr)
    old_buffer_path = raw_dir / 'replay_buffer.zarr'
    if not old_buffer_path.exists():
        raise FileNotFoundError(f'Raw replay buffer not found: {old_buffer_path}')

    if output_zarr.exists():
        shutil.rmtree(output_zarr)

    flatten_zarr_structure(old_buffer_path)
    old_buffer = ReplayBuffer.create_from_path(str(old_buffer_path))
    new_buffer = ReplayBuffer.create_empty_zarr()

    for episode_idx in range(old_buffer.n_episodes):
        episode = old_buffer.get_episode(episode_idx, copy=False)
        flat_episode = flatten_episode_tree(episode)
        normalized = normalize_episode(flat_episode)
        new_buffer.add_episode(normalized)

    compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=1)
    new_buffer.save_to_path(str(output_zarr), compressors=compressor)


def main():
    parser = argparse.ArgumentParser(
        description='Convert raw UR5e demos into a flat zarr replay buffer.',
    )
    parser.add_argument('raw_dir', help='Directory containing replay_buffer.zarr.')
    parser.add_argument('output_zarr', help='Path to the converted zarr dataset.')
    args = parser.parse_args()
    convert_dataset(args.raw_dir, args.output_zarr)


if __name__ == '__main__':
    main()
