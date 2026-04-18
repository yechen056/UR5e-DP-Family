import argparse
import sys
from pathlib import Path

import numpy as np
import zarr


def add_visualizer_to_path():
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "dp-family" / "visualizer",
        repo_root / "visualizer",
    ]
    for visualizer_root in candidates:
        if visualizer_root.exists():
            if str(visualizer_root) not in sys.path:
                sys.path.insert(0, str(visualizer_root))
            return
    raise FileNotFoundError(
        "visualizer package not found. Expected at "
        "'dp-family/visualizer' or 'visualizer'."
    )


def find_first_nonzero_frame(point_clouds, start=0):
    for frame_idx in range(start, point_clouds.shape[0]):
        point_cloud = point_clouds[frame_idx]
        if np.any(np.isfinite(point_cloud)) and np.any(np.abs(point_cloud) > 1e-8):
            return frame_idx
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a point_cloud frame from a UR5e DP3 zarr dataset.",
    )
    parser.add_argument(
        "dataset",
        help="Path to zarr dataset, e.g. data/state.zarr",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Frame index to visualize. Defaults to the first non-zero point cloud.",
    )
    parser.add_argument(
        "--point-cloud-key",
        default="point_cloud",
        help="Point cloud key under data/. Default: point_cloud",
    )
    parser.add_argument(
        "--save-html",
        default=None,
        help="Save an interactive HTML file instead of starting the Flask viewer.",
    )
    args = parser.parse_args()

    root = zarr.open(args.dataset, mode="r")
    data = root["data"]
    if args.point_cloud_key not in data:
        raise KeyError(
            f"data/{args.point_cloud_key} not found. Available keys: {list(data.keys())}"
        )

    point_clouds = data[args.point_cloud_key]
    frame_idx = args.frame
    if frame_idx is None:
        frame_idx = find_first_nonzero_frame(point_clouds)
        if frame_idx is None:
            raise RuntimeError(f"No non-zero point cloud frame found in {args.dataset}")

    point_cloud = np.asarray(point_clouds[frame_idx], dtype=np.float32)
    finite = np.isfinite(point_cloud).all(axis=1)
    nonzero = np.any(np.abs(point_cloud) > 1e-8, axis=1)
    point_cloud = point_cloud[finite & nonzero]
    if point_cloud.shape[0] == 0:
        raise RuntimeError(f"Frame {frame_idx} has no finite non-zero points.")

    print(f"dataset: {args.dataset}")
    print(f"frame: {frame_idx}")
    print(f"points: {point_cloud.shape}")
    print(f"xyz min: {point_cloud[:, :3].min(axis=0)}")
    print(f"xyz max: {point_cloud[:, :3].max(axis=0)}")

    add_visualizer_to_path()
    from visualizer import Visualizer, visualize_pointcloud

    if args.save_html:
        Visualizer().save_visualization_to_file(point_cloud, args.save_html)
    else:
        visualize_pointcloud(point_cloud)


if __name__ == "__main__":
    main()
