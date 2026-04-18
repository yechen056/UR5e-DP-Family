import argparse
import shutil
from pathlib import Path

import numpy as np
import tqdm
import zarr
from numba import jit


DEFAULT_POINT_COUNT = 1024


@jit(nopython=True)
def depth_to_pcd_grid(depth_flat, grid_u, grid_v, fx, fy, cx, cy):
    z = depth_flat
    x = (grid_u - cx) * z / fx
    y = (grid_v - cy) * z / fy
    n = depth_flat.size
    pcd = np.empty((n, 3), dtype=np.float32)
    pcd[:, 0] = x
    pcd[:, 1] = y
    pcd[:, 2] = z
    return pcd


@jit(nopython=True)
def farthest_point_sampling(points, n_samples):
    total = points.shape[0]
    if total <= n_samples:
        return np.arange(total)
    centroids = np.zeros(n_samples, dtype=np.int64)
    distances = np.full(total, 1e10, dtype=np.float32)
    farthest = np.random.randint(0, total)
    for i in range(n_samples):
        centroids[i] = farthest
        centroid = points[farthest, :]
        dist_sq = ((points - centroid) ** 2).sum(axis=1)
        mask = dist_sq < distances
        distances[mask] = dist_sq[mask]
        farthest = np.argmax(distances)
    return centroids


def voxel_grid_sample(points, voxel_size):
    if voxel_size <= 0 or points.shape[0] == 0:
        return points
    grid_coords = np.floor(points[:, :3] / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(grid_coords, axis=0, return_index=True)
    return points[np.sort(unique_idx)]


def sample_points(points, point_count, sampling_method, voxel_size):
    if sampling_method == 'voxel_uniform':
        points = voxel_grid_sample(points, voxel_size)
        sampling_method = 'uniform'

    if points.shape[0] == 0:
        return np.zeros((point_count, 3), dtype=np.float32)
    if points.shape[0] > point_count:
        if sampling_method == 'fps':
            if points.shape[0] > 4096:
                sampled = np.random.choice(points.shape[0], 4096, replace=False)
                points = points[sampled]
            fps_idx = farthest_point_sampling(points, point_count)
            return points[fps_idx].astype(np.float32)
        if sampling_method == 'uniform':
            idx = np.random.choice(points.shape[0], point_count, replace=False)
            return points[idx].astype(np.float32)
        raise ValueError(f'Unsupported sampling method: {sampling_method}')

    if points.shape[0] == point_count:
        return points.astype(np.float32)

    pad_idx = np.random.choice(points.shape[0], point_count - points.shape[0], replace=True)
    return np.vstack([points, points[pad_idx]]).astype(np.float32)


def process_dataset(
    zarr_path,
    point_count,
    depth_min,
    depth_max,
    camera_idx,
    depth_scale,
    sampling_method,
    voxel_size,
    output_zarr=None,
    viz=False,
):
    src_root = zarr.open(zarr_path, mode='r')
    src_data = src_root['data']
    src_meta = src_root['meta']
    episode_ends = src_meta['episode_ends'][:]
    total_steps = int(episode_ends[-1])

    if output_zarr is None:
        root = zarr.open(zarr_path, mode='r+')
        data = root['data']
        if 'point_cloud' in data:
            del data['point_cloud']
        pointcloud_ds = data.create_dataset(
            'point_cloud',
            shape=(total_steps, point_count, 3),
            chunks=(1, point_count, 3),
            dtype='float32',
            overwrite=True,
        )
    else:
        output_path = Path(output_zarr)
        if output_path.exists():
            shutil.rmtree(output_path)
        dst_root = zarr.open(str(output_path), mode='w')
        data = dst_root.create_group('data')
        meta = dst_root.create_group('meta')
        for src_key, dst_key in [
            ('robot_eef_pose', 'robot_eef_pose'),
            ('cartesian_action', 'cartesian_action'),
        ]:
            src = src_data[src_key]
            data.create_dataset(
                dst_key,
                data=src[:],
                shape=src.shape,
                chunks=src.chunks,
                dtype=src.dtype,
                overwrite=True,
            )
        meta.create_dataset(
            'episode_ends',
            data=episode_ends,
            shape=episode_ends.shape,
            chunks=src_meta['episode_ends'].chunks,
            dtype=src_meta['episode_ends'].dtype,
            overwrite=True,
        )
        pointcloud_ds = data.create_dataset(
            'point_cloud',
            shape=(total_steps, point_count, 3),
            chunks=(1, point_count, 3),
            dtype='float32',
            overwrite=True,
        )

    depth_key = f'camera_{camera_idx}_depth'
    intr_key = f'camera_{camera_idx}_intrinsics'
    if depth_key not in src_data or intr_key not in src_data:
        raise RuntimeError(
            f'Single-camera dataset is missing {depth_key} or {intr_key}. '
            f'Available keys: {list(src_data.keys())}'
        )

    height, width = src_data[depth_key].shape[1:]
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    grid_u = u.flatten().astype(np.float32)
    grid_v = v.flatten().astype(np.float32)

    preview_geometries = None
    progress = tqdm.tqdm(total=total_steps)
    for step_idx in range(total_steps):
        depth = src_data[depth_key][step_idx].reshape(-1).astype(np.float32) * depth_scale
        intr = src_data[intr_key][step_idx].astype(np.float32)
        valid = (depth > depth_min) & (depth < depth_max)
        if not np.any(valid):
            pointcloud_ds[step_idx] = np.zeros((point_count, 3), dtype=np.float32)
            progress.update(1)
            continue

        camera_points = depth_to_pcd_grid(
            depth,
            grid_u,
            grid_v,
            intr[0, 0],
            intr[1, 1],
            intr[0, 2],
            intr[1, 2],
        )
        cropped_pcd = camera_points[valid]
        final_pcd = sample_points(cropped_pcd, point_count, sampling_method, voxel_size)

        pointcloud_ds[step_idx] = final_pcd.astype(np.float32)

        if viz and step_idx == 0:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(final_pcd)
            preview_geometries = [
                pcd,
                o3d.geometry.TriangleMesh.create_coordinate_frame(0.2),
            ]

        progress.update(1)

    progress.close()

    if preview_geometries is not None:
        import open3d as o3d

        o3d.visualization.draw_geometries(preview_geometries)


def main():
    parser = argparse.ArgumentParser(
        description='Create single-camera egocentric point clouds for UR5e DP3 data.',
    )
    parser.add_argument('dataset', help='Processed zarr dataset path.')
    parser.add_argument('--point-count', type=int, default=DEFAULT_POINT_COUNT)
    parser.add_argument('--depth-min', type=float, default=0.05)
    parser.add_argument('--depth-max', type=float, default=2.0)
    parser.add_argument('--camera-idx', type=int, default=0, help='Single camera index to convert.')
    parser.add_argument(
        '--depth-scale',
        type=float,
        default=0.00025,
        help='Raw RealSense depth unit to meters. L515 is commonly 0.00025; D400 is commonly 0.001.',
    )
    parser.add_argument(
        '--sampling-method',
        choices=['fps', 'uniform', 'voxel_uniform'],
        default='fps',
        help='Point sampling method. DP3 uses fps; iDP3 uses voxel_uniform.',
    )
    parser.add_argument('--voxel-size', type=float, default=0.005)
    parser.add_argument(
        '--output-zarr',
        default=None,
        help='If set, write a canonical training zarr instead of overwriting data/point_cloud in-place.',
    )
    parser.add_argument('--viz', action='store_true', help='Visualize the first fused frame.')
    args = parser.parse_args()

    process_dataset(
        zarr_path=args.dataset,
        point_count=args.point_count,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        camera_idx=args.camera_idx,
        depth_scale=args.depth_scale,
        sampling_method=args.sampling_method,
        voxel_size=args.voxel_size,
        output_zarr=args.output_zarr,
        viz=args.viz,
    )


if __name__ == '__main__':
    main()
