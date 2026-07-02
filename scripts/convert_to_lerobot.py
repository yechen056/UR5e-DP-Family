#!/usr/bin/env python3
"""Convert UR5e DP-Family zarr replay buffers to LeRobot v3 datasets."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np


JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow.pos",
    "wrist_1.pos",
    "wrist_2.pos",
    "wrist_3.pos",
    "gripper.pos",
)

EEF_NAMES = (
    "tcp.x",
    "tcp.y",
    "tcp.z",
    "tcp.rx",
    "tcp.ry",
    "tcp.rz",
    "gripper.pos",
)

DEFAULT_TASK = "Replay a bimanual UR5e demonstration collected with Quest teleoperation."


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def prefixed_names(names: tuple[str, ...]) -> list[str]:
    return [f"left_{name}" for name in names] + [f"right_{name}" for name in names]


def make_features(action_mode: str, image_shape: tuple[int, int, int], use_videos: bool) -> dict:
    if action_mode == "joint":
        names = prefixed_names(JOINT_NAMES)
    elif action_mode == "eef":
        names = prefixed_names(EEF_NAMES)
    else:
        raise ValueError(f"Unsupported action mode: {action_mode}")

    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(names),),
            "names": names,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(names),),
            "names": names,
        },
        "observation.images.global": {
            "dtype": "video" if use_videos else "image",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
    }


def import_dependencies():
    if sys.version_info < (3, 12):
        raise SystemExit(
            "LeRobot v3 dataset conversion requires Python >= 3.12. "
            f"Current Python is {sys.version.split()[0]}. "
            "Create/use a Python 3.12+ environment and rerun with "
            "PYTHON=/path/to/python bash scripts/convert_to_lerobot.sh ..."
        )

    try:
        import zarr
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency `zarr`. Run with a Python environment that has DP-Family/LeRobot "
            "dataset dependencies installed, or pass PYTHON=/path/to/python to convert_to_lerobot.sh."
        ) from exc

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise SystemExit(
            "Failed to import `LeRobotDataset`. Ensure /home/yechen/UR5e-LeRobot/src is on "
            "PYTHONPATH and that the current Python environment has LeRobot dataset dependencies "
            f"installed. Original error: {exc}"
        ) from exc

    return zarr, LeRobotDataset


def resolve_replay_buffer(input_path: Path) -> Path:
    if input_path.name == "replay_buffer.zarr":
        return input_path
    candidate = input_path / "replay_buffer.zarr"
    if candidate.exists():
        return candidate
    return input_path


def require_array(root, key: str):
    if key not in root["data"]:
        raise KeyError(f"Input dataset is missing data/{key}.")
    return root["data"][key]


def validate_episode_lengths(
    episode_idx: int,
    episode_end: int,
    arrays: dict[str, object],
) -> None:
    for key, array in arrays.items():
        if array.shape[0] < episode_end:
            raise ValueError(
                f"Episode {episode_idx} requires frame index {episode_end - 1}, "
                f"but data/{key} only has {array.shape[0]} frames."
            )


def convert_dataset(args: argparse.Namespace) -> None:
    zarr, LeRobotDataset = import_dependencies()

    input_path = resolve_replay_buffer(Path(args.input).expanduser())
    output_path = Path(args.output).expanduser()

    if not input_path.exists():
        raise FileNotFoundError(f"Input replay buffer not found: {input_path}")
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")
        shutil.rmtree(output_path)

    root = zarr.open(str(input_path), mode="r")
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    if episode_ends.size == 0:
        raise ValueError(f"Input replay buffer has no episodes: {input_path}")
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ValueError("--max-episodes must be positive.")
        episode_ends = episode_ends[: args.max_episodes]

    image_arr = require_array(root, args.image_key)
    if len(image_arr.shape) != 4 or image_arr.shape[-1] != 3:
        raise ValueError(f"Expected data/{args.image_key} shape [T,H,W,3], got {image_arr.shape}.")
    image_shape = tuple(int(x) for x in image_arr.shape[1:])

    if args.action_mode == "joint":
        state_key = "robot_joint"
        action_key = "joint_action"
    else:
        state_key = "robot_eef_pose"
        action_key = "cartesian_action"

    state_arr = require_array(root, state_key)
    action_arr = require_array(root, action_key)
    if state_arr.shape[1:] != (14,):
        raise ValueError(f"Expected data/{state_key} shape [T,14], got {state_arr.shape}.")
    if action_arr.shape[1:] != (14,):
        raise ValueError(f"Expected data/{action_key} shape [T,14], got {action_arr.shape}.")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=make_features(args.action_mode, image_shape, args.use_videos),
        root=output_path,
        robot_type="bi_ur5e_pgi",
        use_videos=args.use_videos,
        image_writer_threads=args.image_writer_threads,
        streaming_encoding=args.streaming_encoding,
        vcodec=args.vcodec,
    )

    start = 0
    try:
        for episode_idx, end in enumerate(episode_ends):
            end = int(end)
            validate_episode_lengths(
                episode_idx,
                end,
                {
                    args.image_key: image_arr,
                    state_key: state_arr,
                    action_key: action_arr,
                },
            )
            for frame_idx in range(start, end):
                frame = {
                    "observation.state": np.asarray(state_arr[frame_idx], dtype=np.float32),
                    "action": np.asarray(action_arr[frame_idx], dtype=np.float32),
                    "observation.images.global": np.asarray(image_arr[frame_idx]),
                    "task": args.task,
                }
                dataset.add_frame(frame)
            dataset.save_episode()
            print(f"saved episode {episode_idx}: frames {start}..{end - 1} ({end - start})")
            start = end
    finally:
        dataset.finalize()

    print(f"LeRobot dataset written to: {output_path}")
    print(f"repo_id: {args.repo_id}")
    print(f"action_mode: {args.action_mode}")
    print(f"episodes: {len(episode_ends)}")
    print(f"frames: {int(episode_ends[-1])}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a UR5e DP-Family replay_buffer.zarr into a LeRobot v3 dataset.",
    )
    parser.add_argument("--input", default="data/ur5e_bimanual_quest_raw/replay_buffer.zarr")
    parser.add_argument("--output", default=None)
    parser.add_argument("--action-mode", choices=("joint", "eef"), default="joint")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--use-videos", type=parse_bool, default=True)
    parser.add_argument("--image-key", default="camera_0_color")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--streaming-encoding", type=parse_bool, default=False)
    parser.add_argument("--vcodec", default="h264")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.output is None:
        args.output = f"data/lerobot_ur5e_bimanual_quest_{args.action_mode}"
    if args.repo_id is None:
        args.repo_id = f"FANYECHEN/ur5e-bimanual-quest-{args.action_mode}"
    convert_dataset(args)


if __name__ == "__main__":
    main()
