import argparse
import shutil
from pathlib import Path

import numpy as np
import zarr


ALIASES = {
    'point_cloud': ['point_cloud', 'fused_pointcloud'],
    'robot_eef_pose': ['robot_eef_pose', 'ActualTCPPose', 'TargetTCPPose'],
    'cartesian_action': ['cartesian_action', 'TargetTCPPose', 'ActualTCPPose'],
    'robot_joint': ['robot_joint', 'ActualQ', 'TargetQ'],
    'joint_action': ['joint_action', 'TargetQ', 'ActualQ'],
}
MANDATORY_KEYS = ['point_cloud', 'robot_eef_pose', 'cartesian_action']
OPTIONAL_KEYS = ['robot_joint', 'joint_action']


def resolve_source(data_group, target_key):
    for candidate in ALIASES[target_key]:
        if candidate in data_group:
            return candidate
    return None


def copy_dataset(src_group, dst_group, src_key, dst_key):
    array = src_group[src_key]
    dst_group.create_dataset(
        dst_key,
        data=array[:],
        shape=array.shape,
        chunks=array.chunks,
        dtype=array.dtype,
        overwrite=True,
    )


def prepare_dataset(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.exists():
        shutil.rmtree(output_path)

    src_root = zarr.open(str(input_path), mode='r')
    dst_root = zarr.open(str(output_path), mode='w')
    src_data = src_root['data']
    src_meta = src_root['meta']
    dst_data = dst_root.create_group('data')
    dst_meta = dst_root.create_group('meta')

    for target_key in MANDATORY_KEYS:
        source_key = resolve_source(src_data, target_key)
        if source_key is None:
            raise KeyError(
                f'Cannot find a source array for "{target_key}". '
                f'Checked aliases: {ALIASES[target_key]}'
            )
        copy_dataset(src_data, dst_data, source_key, target_key)

    for target_key in OPTIONAL_KEYS:
        source_key = resolve_source(src_data, target_key)
        if source_key is None:
            continue
        copy_dataset(src_data, dst_data, source_key, target_key)

    episode_ends = np.asarray(src_meta['episode_ends'][:])
    dst_meta.create_dataset(
        'episode_ends',
        data=episode_ends,
        shape=episode_ends.shape,
        chunks=src_meta['episode_ends'].chunks,
        dtype=src_meta['episode_ends'].dtype,
        overwrite=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Prepare a canonical UR5e zarr dataset for DP3 training.',
    )
    parser.add_argument('input_zarr', help='Path to the source processed zarr dataset.')
    parser.add_argument('output_zarr', help='Path to the output canonical zarr dataset.')
    args = parser.parse_args()
    prepare_dataset(args.input_zarr, args.output_zarr)


if __name__ == '__main__':
    main()
