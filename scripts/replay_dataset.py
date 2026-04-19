import sys
import time
from pathlib import Path

import click
import numpy as np
from termcolor import cprint

REPO_ROOT = Path(__file__).resolve().parents[1]
DP3_ROOT = REPO_ROOT / "dp-family"
if str(DP3_ROOT) not in sys.path:
    sys.path.insert(0, str(DP3_ROOT))

from common.precise_sleep import precise_wait
from common.replay_buffer import ReplayBuffer
from common.trans_utils import are_joints_close, interpolate_joints, interpolate_poses


def resolve_dataset_path(dataset_path: str) -> Path:
    path = Path(dataset_path).expanduser()
    if path.is_dir():
        path = path / "replay_buffer.zarr"
    if not path.exists():
        raise click.ClickException(f"Dataset not found: {path}")
    if path.suffix != ".zarr":
        raise click.ClickException(
            f"Unsupported dataset path: {path}. Expected a directory or a .zarr path."
        )
    return path


def infer_control_mode(data_keys, requested_mode):
    if requested_mode is not None:
        return requested_mode
    if "cartesian_action" in data_keys:
        return "eef"
    if "joint_action" in data_keys:
        return "joint"
    raise click.ClickException(
        "Unable to infer control mode. Dataset must contain `cartesian_action` or `joint_action`, "
        "or pass `--control_mode` explicitly."
    )


def get_episode_bounds(episode_ends, episode_idx: int):
    if episode_idx < 0 or episode_idx >= len(episode_ends):
        raise click.ClickException(
            f"Episode index {episode_idx} is out of range. "
            f"Available episodes: 0 to {len(episode_ends) - 1}."
        )
    start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    end = int(episode_ends[episode_idx])
    return start, end


def select_action_key(control_mode: str, data_keys):
    if control_mode == "eef":
        action_key = "cartesian_action"
    else:
        action_key = "joint_action"
    if action_key not in data_keys:
        raise click.ClickException(
            f"Dataset is missing required key `{action_key}` for control_mode={control_mode}. "
            f"Available keys: {sorted(data_keys)}"
        )
    return action_key


def get_joint_compare_dim(is_bimanual: bool):
    return 12 if is_bimanual else 6


def filter_eef_actions(actions: np.ndarray, threshold: float = 0.001) -> np.ndarray:
    if len(actions) <= 1:
        return actions

    action_dim = actions.shape[1]
    if action_dim in (12, 14):
        translation_dims = [0, 1, 2, action_dim // 2, action_dim // 2 + 1, action_dim // 2 + 2]
    else:
        translation_dims = [0, 1, 2]

    pose_diff = np.diff(actions[:, translation_dims], axis=0)
    pose_movement = np.linalg.norm(pose_diff, axis=1)
    moving_indices = np.where(pose_movement > threshold)[0] + 1
    moving_mask = np.zeros(len(actions), dtype=bool)
    moving_mask[0] = True
    moving_mask[moving_indices] = True
    return actions[moving_mask]


def filter_joint_actions(obs_joints: np.ndarray, joint_actions: np.ndarray, is_bimanual: bool) -> np.ndarray:
    compare_dim = get_joint_compare_dim(is_bimanual)
    obs_joints_cmp = obs_joints[:, :compare_dim]
    joint_actions_cmp = joint_actions[:, :compare_dim]
    not_moving_mask = are_joints_close(obs_joints_cmp, joint_actions_cmp)
    moving_mask = ~not_moving_mask
    if len(moving_mask) > 0:
        moving_mask[0] = True
    return joint_actions[moving_mask]


def build_placeholder_joint_actions(length: int, is_bimanual: bool, use_gripper: bool) -> np.ndarray:
    joint_dim = (2 if is_bimanual else 1) * (6 + int(use_gripper))
    return np.zeros((length, joint_dim), dtype=np.float32)


def build_placeholder_eef_actions(length: int, is_bimanual: bool, use_gripper: bool) -> np.ndarray:
    eef_dim = (2 if is_bimanual else 1) * (6 + int(use_gripper))
    return np.zeros((length, eef_dim), dtype=np.float32)


def is_eef_close(current_pose: np.ndarray, target_pose: np.ndarray) -> bool:
    current_pose = np.asarray(current_pose, dtype=np.float32)
    target_pose = np.asarray(target_pose, dtype=np.float32)
    if current_pose.shape != target_pose.shape:
        return False

    if current_pose.shape[-1] in (12, 14):
        per_arm_dim = current_pose.shape[-1] // 2
        left_current, right_current = current_pose[:per_arm_dim], current_pose[per_arm_dim:]
        left_target, right_target = target_pose[:per_arm_dim], target_pose[per_arm_dim:]
        return is_eef_close(left_current, left_target) and is_eef_close(right_current, right_target)

    translation_close = np.linalg.norm(current_pose[:3] - target_pose[:3]) <= 0.005
    rotation_close = np.linalg.norm(current_pose[3:6] - target_pose[3:6]) <= 0.10
    gripper_close = True
    if current_pose.shape[-1] >= 7 and target_pose.shape[-1] >= 7:
        gripper_close = abs(float(current_pose[6]) - float(target_pose[6])) <= 0.25
    return translation_close and rotation_close and gripper_close


def wait_until_target_reached(env, control_mode: str, target_action: np.ndarray, is_bimanual: bool,
                              frequency: float, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    target_action = np.asarray(target_action, dtype=np.float32)
    while time.monotonic() < deadline:
        robot_state = env.get_robot_state()
        if control_mode == "joint":
            current_joint = np.asarray(robot_state["ActualQ"], dtype=np.float32)
            compare_dim = min(
                get_joint_compare_dim(is_bimanual),
                current_joint.shape[-1],
                target_action.shape[-1],
            )
            if are_joints_close(
                current_joint[None, :compare_dim],
                target_action[None, :compare_dim],
            )[0]:
                return True
        else:
            current_pose = np.asarray(robot_state["ActualTCPPose"], dtype=np.float32)
            if is_eef_close(current_pose, target_action):
                return True
        time.sleep(max(0.05, 0.5 / frequency))
    return False


def append_eef_trajectory(container, start_pose: np.ndarray, target_pose: np.ndarray, speed: float = 0.03):
    start_pose = np.asarray(start_pose, dtype=np.float32)
    target_pose = np.asarray(target_pose, dtype=np.float32)

    if start_pose.shape != target_pose.shape:
        raise click.ClickException(
            f"Cannot move to first replay pose: current pose shape {start_pose.shape} "
            f"does not match target pose shape {target_pose.shape}."
        )

    if start_pose.shape[-1] in (12, 14):
        per_arm_dim = start_pose.shape[-1] // 2
        left_start, right_start = start_pose[:per_arm_dim], start_pose[per_arm_dim:]
        left_target, right_target = target_pose[:per_arm_dim], target_pose[per_arm_dim:]

        left_waypoints, left_steps, left_close = interpolate_poses(left_start, left_target, speed)
        right_waypoints, right_steps, right_close = interpolate_poses(right_start, right_target, speed)

        if left_close and right_close:
            container.append(target_pose.copy())
            return

        max_steps = max(left_steps, right_steps)
        left_waypoints += [left_target.copy()] * (max_steps - len(left_waypoints))
        right_waypoints += [right_target.copy()] * (max_steps - len(right_waypoints))

        for left_pose, right_pose in zip(left_waypoints, right_waypoints):
            container.append(
                np.concatenate(
                    [np.asarray(left_pose, dtype=np.float32), np.asarray(right_pose, dtype=np.float32)],
                    axis=0,
                )
            )
    else:
        waypoints, _, is_close = interpolate_poses(start_pose, target_pose, speed)
        if is_close:
            container.append(target_pose.copy())
            return
        container.extend([np.asarray(waypoint, dtype=np.float32) for waypoint in waypoints])

    container.append(target_pose.copy())


def build_move_to_first_actions(control_mode: str, current_state: dict, first_action: np.ndarray, is_bimanual: bool):
    first_action = np.asarray(first_action, dtype=np.float32)
    if control_mode == "joint":
        if "ActualQ" not in current_state:
            raise click.ClickException("Robot state is missing `ActualQ`, cannot move to first joint target.")
        current_joint = np.asarray(current_state["ActualQ"], dtype=np.float32)
        if current_joint.shape != first_action.shape:
            raise click.ClickException(
                f"Current joint shape {current_joint.shape} does not match first replay joint shape {first_action.shape}."
            )
        joint_waypoints = interpolate_joints(current_joint, first_action, speed=0.05)
        if len(joint_waypoints) == 0 or not np.allclose(joint_waypoints[-1], first_action):
            joint_waypoints = np.concatenate([joint_waypoints, first_action[None, :]], axis=0)
        return joint_waypoints.astype(np.float32), None

    if "ActualTCPPose" not in current_state:
        raise click.ClickException("Robot state is missing `ActualTCPPose`, cannot move to first EEF target.")
    current_pose = np.asarray(current_state["ActualTCPPose"], dtype=np.float32)
    eef_waypoints = []
    append_eef_trajectory(eef_waypoints, current_pose, first_action, speed=0.03)
    return None, np.asarray(eef_waypoints, dtype=np.float32)


def print_replay_summary(dataset_path: Path, total_episodes: int, episode: int, episode_start: int, episode_end: int,
                         control_mode: str, is_bimanual: bool, action_key: str, actions_before: int, actions_after: int,
                         action_shape):
    cprint(f"Dataset: {dataset_path}", "green", attrs=["bold"])
    cprint(f"Episodes available: {total_episodes}", "green", attrs=["bold"])
    cprint(f"Replay episode: {episode}", "green", attrs=["bold"])
    cprint(f"Episode steps: {episode_start} -> {episode_end - 1}", "green", attrs=["bold"])
    cprint(f"Control mode: {control_mode}", "green", attrs=["bold"])
    cprint(f"Bimanual mode: {is_bimanual}", "green", attrs=["bold"])
    cprint(f"Action key: {action_key}", "green", attrs=["bold"])
    cprint(f"Action shape: {action_shape}", "green", attrs=["bold"])
    cprint(f"Actions before filtering: {actions_before}", "green", attrs=["bold"])
    cprint(f"Actions after filtering: {actions_after}", "green", attrs=["bold"])


@click.command()
@click.argument("dataset_path")
@click.option("--episode", type=int, required=True, help="Episode index to replay.")
@click.option(
    "--control_mode",
    type=click.Choice(["eef", "joint"]),
    default=None,
    help="Replay control mode. Defaults to auto-infer from dataset keys.",
)
@click.option("--bimanual/--single_arm", default=False, help="Enable dual-arm replay mode.")
@click.option("--robot_ip", default="192.168.1.5", help="Single-arm robot IP.")
@click.option("--robot_left_ip", default="192.168.1.3", help="Left robot IP for bimanual replay.")
@click.option("--robot_right_ip", default="192.168.1.5", help="Right robot IP for bimanual replay.")
@click.option("--frequency", default=10.0, type=float, show_default=True, help="Replay frequency in Hz.")
@click.option("--speed_slider_value", default=0.6, type=float, show_default=True, help="Robot speed slider value during replay.")
@click.option("--use_gripper/--no-use_gripper", default=True, help="Replay gripper dimensions.")
@click.option("--filter_static/--no-filter_static", default=False, help="Filter out obvious static segments.")
@click.option("--dry_run", is_flag=True, default=False, help="Print replay plan without connecting to robots.")
def main(
    dataset_path,
    episode,
    control_mode,
    bimanual,
    robot_ip,
    robot_left_ip,
    robot_right_ip,
    frequency,
    speed_slider_value,
    use_gripper,
    filter_static,
    dry_run,
):
    dataset_path = resolve_dataset_path(dataset_path)
    replay_buffer = ReplayBuffer.create_from_path(str(dataset_path), mode="r")
    total_episodes = replay_buffer.n_episodes
    if total_episodes == 0:
        raise click.ClickException(f"No episodes found in dataset: {dataset_path}")

    episode_start, episode_end = get_episode_bounds(replay_buffer.episode_ends[:], episode)
    episode_data = replay_buffer.get_episode(episode, copy=True)
    data_keys = set(episode_data.keys())

    control_mode = infer_control_mode(data_keys, control_mode)
    action_key = select_action_key(control_mode, data_keys)
    actions = np.asarray(episode_data[action_key], dtype=np.float32)
    actions_before = len(actions)

    if control_mode == "joint" and filter_static:
        if "joint_action" not in data_keys or "robot_joint" not in data_keys:
            raise click.ClickException(
                "Joint replay with filtering requires both `robot_joint` and `joint_action` in the dataset."
            )
        actions = filter_joint_actions(
            np.asarray(episode_data["robot_joint"], dtype=np.float32),
            np.asarray(episode_data["joint_action"], dtype=np.float32),
            is_bimanual=bimanual,
        )
    elif control_mode == "eef" and filter_static:
        actions = filter_eef_actions(actions)

    print_replay_summary(
        dataset_path=dataset_path,
        total_episodes=total_episodes,
        episode=episode,
        episode_start=episode_start,
        episode_end=episode_end,
        control_mode=control_mode,
        is_bimanual=bimanual,
        action_key=action_key,
        actions_before=actions_before,
        actions_after=len(actions),
        action_shape=tuple(actions.shape),
    )

    if len(actions) == 0:
        raise click.ClickException("No actions remain after filtering.")

    batch_size = max(1, int(round(frequency)))
    dt = 1.0 / frequency

    if dry_run:
        cprint("dry_run=True, skipping robot connection.", "green", attrs=["bold"])
        cprint("Will move robot to the first replay action before executing the episode.", "green", attrs=["bold"])
        cprint(
            f"Replay batches: {int(np.ceil(len(actions) / float(batch_size)))} at batch_size={batch_size}",
            "green",
            attrs=["bold"],
        )
        return

    from real_world.real_ur5e_env import RealUR5eEnv

    output_dir = "/tmp/ur5e_dataset_replay"
    with RealUR5eEnv(
        output_dir=output_dir,
        robot_left_ip=robot_left_ip if bimanual else robot_ip,
        robot_right_ip=robot_right_ip if bimanual else robot_ip,
        frequency=frequency,
        speed_slider_value=speed_slider_value,
        single_arm_type="right",
        ctrl_mode=control_mode,
        use_gripper=use_gripper,
        is_bimanual=bimanual,
    ) as env:
        cprint("Environment created, starting replay...", "green", attrs=["bold"])
        current_state = env.get_robot_state()
        move_joint_actions, move_eef_actions = build_move_to_first_actions(
            control_mode=control_mode,
            current_state=current_state,
            first_action=actions[0],
            is_bimanual=bimanual,
        )
        if control_mode == "joint":
            move_count = len(move_joint_actions)
            move_timestamps = time.time() + np.arange(move_count, dtype=np.float64) * dt + 0.2
            env.exec_actions(
                joint_actions=move_joint_actions,
                eef_actions=build_placeholder_eef_actions(move_count, bimanual, use_gripper),
                timestamps=move_timestamps,
                mode="joint",
            )
        else:
            move_count = len(move_eef_actions)
            move_timestamps = time.time() + np.arange(move_count, dtype=np.float64) * dt + 0.2
            env.exec_actions(
                joint_actions=build_placeholder_joint_actions(move_count, bimanual, use_gripper),
                eef_actions=move_eef_actions,
                timestamps=move_timestamps,
                mode="eef",
            )
        reached_target = wait_until_target_reached(
            env,
            control_mode=control_mode,
            target_action=actions[0],
            is_bimanual=bimanual,
            frequency=frequency,
            timeout=max(15.0, move_count / float(frequency) + 5.0),
        )
        if reached_target:
            cprint(
                f"Moved to first replay action using {move_count} preparation steps.",
                "green",
                attrs=["bold"],
            )
        else:
            cprint(
                "Timed out while waiting to reach the first replay action. Starting replay anyway.",
                "yellow",
                attrs=["bold"],
            )

        start_step = 0

        while start_step < len(actions):
            loop_end_time = time.monotonic() + 1.0
            end_step = min(start_step + batch_size, len(actions))
            action_batch = actions[start_step:end_step]
            timestamp_batch = time.time() + np.arange(len(action_batch), dtype=np.float64) * dt + 0.2

            if control_mode == "joint":
                joint_actions = action_batch
                eef_actions = build_placeholder_eef_actions(len(action_batch), bimanual, use_gripper)
            else:
                joint_actions = build_placeholder_joint_actions(len(action_batch), bimanual, use_gripper)
                eef_actions = action_batch

            env.exec_actions(
                joint_actions=joint_actions,
                eef_actions=eef_actions,
                timestamps=timestamp_batch,
                mode=control_mode,
            )
            cprint(
                f"Executed actions {start_step} to {end_step - 1} / {len(actions)}",
                "green",
                attrs=["bold"],
            )
            start_step = end_step
            precise_wait(loop_end_time)

    cprint("Replay completed successfully!", "green", attrs=["bold"])


if __name__ == "__main__":
    main()
