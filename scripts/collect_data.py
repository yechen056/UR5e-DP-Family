import time
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
import sys
import warnings

import click
import cv2
import numpy as np
import scipy.spatial.transform as st
from termcolor import cprint

warnings.filterwarnings("ignore", message="ignoring keyword argument 'compressors'")

REPO_ROOT = Path(__file__).resolve().parents[1]
DP3_ROOT = REPO_ROOT / 'dp-family'
if str(DP3_ROOT) not in sys.path:
    sys.path.insert(0, str(DP3_ROOT))

from common.precise_sleep import precise_wait
from common.trans_utils import interpolate_poses
from real_world.keystroke_counter import KeystrokeCounter, Key, KeyCode
from real_world.quest_shared_memory import QuestTeleop
from real_world.real_ur5e_env import RealUR5eEnv
from real_world.spacemouse_shared_memory import Spacemouse


HOME_POSE = np.array([-0.11130, -0.48927, 0.22326, 3.152, -0.007, -0.001, 1.0], dtype=np.float32)
TOP_POSE = np.array([0.16399, -0.46193, 0.40260, 0.001, 3.133, -0.129, 1.0], dtype=np.float32)
LEFT_HOME_POSE = np.array([0.141773, -0.196927, 0.421262, -1.225401, 1.290381, 1.263347], dtype=np.float32)
RIGHT_HOME_POSE = np.array([-0.15458, -0.43091, 0.26655, 0.022, 3.124, 0.044], dtype=np.float32)


def build_vis(obs, camera_idx):
    key = f'camera_{camera_idx}_color'
    if key not in obs:
        return None
    return obs[key][-1, :, :, ::-1].copy()


def safe_imshow(window_name, image):
    try:
        cv2.imshow(window_name, image)
        cv2.waitKey(1)
        return True
    except cv2.error:
        return False


def append_pose_trajectory(container, start_pose, target_pose, speed=0.01):
    start_pose = np.asarray(start_pose, dtype=np.float32)
    target_pose = np.asarray(target_pose, dtype=np.float32)

    if start_pose.shape != target_pose.shape:
        raise ValueError('Start and target poses must have the same shape.')

    if start_pose.shape[-1] in (12, 14):
        per_arm_dim = start_pose.shape[-1] // 2
        left_start, right_start = start_pose[:per_arm_dim], start_pose[per_arm_dim:]
        left_target, right_target = target_pose[:per_arm_dim], target_pose[per_arm_dim:]

        left_waypoints, left_steps, left_close = interpolate_poses(left_start, left_target, speed)
        right_waypoints, right_steps, right_close = interpolate_poses(right_start, right_target, speed)

        if left_close and right_close:
            container.append(np.array(target_pose, dtype=np.float32, copy=True))
            return

        max_steps = max(left_steps, right_steps)
        left_waypoints += [np.array(left_target, dtype=np.float32, copy=True)] * (max_steps - len(left_waypoints))
        right_waypoints += [np.array(right_target, dtype=np.float32, copy=True)] * (max_steps - len(right_waypoints))

        for left_pose, right_pose in zip(left_waypoints, right_waypoints):
            container.append(
                np.concatenate(
                    [
                        np.asarray(left_pose, dtype=np.float32),
                        np.asarray(right_pose, dtype=np.float32),
                    ],
                    axis=0,
                )
            )
    else:
        waypoints, _, _ = interpolate_poses(start_pose, target_pose, speed)
        container.extend(waypoints)

    container.append(np.array(target_pose, dtype=np.float32, copy=True))


def get_bimanual_home_pose(use_gripper: bool) -> np.ndarray:
    if use_gripper:
        return np.concatenate(
            [
                LEFT_HOME_POSE,
                np.array([1.0], dtype=np.float32),
                RIGHT_HOME_POSE,
                np.array([1.0], dtype=np.float32),
            ],
            axis=0,
        )
    return np.concatenate([LEFT_HOME_POSE, RIGHT_HOME_POSE], axis=0)


@click.command()
@click.option('--output', '-o', default='data/ur5e_raw', help='Directory to save the raw demo dataset.')
@click.option('--robot_ip', default='192.168.1.5', help='UR5e IP address.')
@click.option('--camera_idx', default=0, type=int, help='Single L515 camera index to display and record.')
@click.option('--init_joints/--no-init_joints', default=True, help='Move to init joints on startup.')
@click.option('--frequency', '-f', default=10, type=float, help='Control frequency in Hz.')
@click.option('--command_latency', '-cl', default=0.01, type=float, help='Command latency in seconds.')
@click.option('--debug', '-d', is_flag=True, default=False, help='Enable debug logging.')
@click.option('--dummy_robot', '-dr', is_flag=True, default=False, help='Use dummy robot mode.')
@click.option('--use_gripper/--no-use_gripper', default=True, help='Enable gripper commands.')
@click.option('--no_gui', is_flag=True, default=False, help='Disable OpenCV preview window.')
@click.option(
    '--input_device',
    type=click.Choice(['spacemouse', 'quest']),
    default='spacemouse',
    show_default=True,
    help='Teleop input device.',
)
@click.option('--bimanual/--single_arm', default=False, help='Enable dual-arm collection mode.')
@click.option('--robot_left_ip', default='192.168.1.3', help='Left UR5e IP address.')
@click.option('--robot_right_ip', default='192.168.1.5', help='Right UR5e IP address.')
@click.option(
    '--quest_single_hand',
    type=click.Choice(['l', 'r']),
    default='r',
    show_default=True,
    help='Quest single-arm mode hand selection.',
)
@click.option(
    '--quest_translation_scale',
    default=3.0,
    type=float,
    show_default=True,
    help='Position scaling factor for Quest pose mapping.',
)
@click.option(
    '--control_mode',
    type=click.Choice(['eef', 'joint']),
    default='eef',
    show_default=True,
    help='Teleop execution mode. SpaceMouse always outputs EEF deltas; joint mode converts targets via IK.',
)
def main(
    output,
    robot_ip,
    camera_idx,
    init_joints,
    frequency,
    command_latency,
    debug,
    dummy_robot,
    use_gripper,
    no_gui,
    input_device,
    bimanual,
    robot_left_ip,
    robot_right_ip,
    quest_single_hand,
    quest_translation_scale,
    control_mode,
):
    if bimanual and input_device != 'quest':
        raise click.ClickException('Dual-arm collection currently supports `--input_device quest` only.')

    dt = 1.0 / frequency
    max_pos_speed = 0.2
    max_rot_speed = 0.4
    bimanual_home_pose = get_bimanual_home_pose(use_gripper=use_gripper) if bimanual else None

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter:
            if input_device == 'quest':
                device = QuestTeleop(
                    bimanual=bimanual,
                    single_hand=quest_single_hand,
                    translation_scale=quest_translation_scale,
                    use_gripper=use_gripper,
                )
            else:
                device = Spacemouse(shm_manager=shm_manager, use_gripper=use_gripper)

            with device:
                with RealUR5eEnv(
                    output_dir=output,
                    robot_left_ip=robot_left_ip if bimanual else robot_ip,
                    robot_right_ip=robot_right_ip if bimanual else robot_ip,
                    obs_image_resolution=(640, 480),
                    frequency=frequency,
                    init_joints=init_joints,
                    max_pos_speed=max_pos_speed,
                    max_rot_speed=max_rot_speed,
                    speed_slider_value=1,
                    enable_multi_cam_vis=False,
                    record_raw_video=True,
                    thread_per_video=3,
                    video_crf=21,
                    shm_manager=shm_manager,
                    enable_depth=True,
                    debug=debug,
                    dummy_robot=dummy_robot,
                    single_arm_type='right',
                    ctrl_mode=control_mode,
                    use_gripper=use_gripper,
                    is_bimanual=bimanual,
                ) as env:
                    env.configure_teleop_metadata(
                        input_device=input_device,
                        mapping_version=(
                            'quest_v1_bimanual'
                            if input_device == 'quest' and bimanual
                            else 'quest_v1_single'
                            if input_device == 'quest'
                            else ''
                        ),
                        translation_scale=quest_translation_scale if input_device == 'quest' else np.nan,
                    )

                    product_lines = [
                        camera.worker_state.get('product_line', '')
                        for camera in env.realsense.cameras.values()
                    ]
                    if any(product_line == 'L500' for product_line in product_lines):
                        env.realsense.set_depth_preset('Default')
                        env.realsense.set_depth_exposure()
                        env.realsense.set_exposure()
                        env.realsense.set_contrast(contrast=30)
                        env.realsense.set_white_balance()
                    else:
                        env.realsense.set_depth_preset('Default')
                        env.realsense.set_depth_exposure(33000, 16)
                        env.realsense.set_exposure(exposure=115, gain=64)
                        env.realsense.set_contrast(contrast=60)
                        env.realsense.set_white_balance(white_balance=3100)

                    cv2.setNumThreads(1)
                    ready_deadline = time.monotonic() + 20.0
                    while not env.is_ready and time.monotonic() < ready_deadline:
                        time.sleep(0.1)
                    if not env.is_ready:
                        raise RuntimeError('RealUR5eEnv failed to become ready within 20 seconds.')

                    time.sleep(0.5)
                    state = env.get_robot_state()
                    target_pose = np.asarray(state['TargetTCPPose'], dtype=np.float32)
                    if target_pose.shape[-1] == 6 and use_gripper:
                        target_pose = np.concatenate([target_pose, np.array([1.0], dtype=np.float32)])

                    gui_enabled = not no_gui

                    if gui_enabled:
                        blank = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(blank, 'Ready', (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                        if not safe_imshow('ur5e_collect', blank):
                            gui_enabled = False
                            print('OpenCV GUI is unavailable, continuing in no_gui mode.')

                    cprint('Ready.', 'green', attrs=['bold'])
                    cprint(f'Control mode: {control_mode}', 'green', attrs=['bold'])
                    cprint(f'Input device: {input_device}', 'green', attrs=['bold'])
                    cprint(f'Bimanual mode: {bimanual}', 'green', attrs=['bold'])
                    if bimanual:
                        cprint(
                            'Keyboard: c=start, s=stop, h=home, backspace=drop, q=quit',
                            'green',
                            attrs=['bold'],
                        )
                    else:
                        cprint(
                            'Keyboard: c=start, s=stop, h=home, t=top, backspace=drop, q=quit',
                            'green',
                            attrs=['bold'],
                        )

                    t_start = time.monotonic()
                    iter_idx = 0
                    stop = False
                    is_recording = False
                    intermediate_pose = []
                    gripper_pose = 1.0

                    while not stop:
                        t_cycle_end = t_start + (iter_idx + 1) * dt
                        t_sample = t_cycle_end - command_latency
                        t_command_target = t_cycle_end + dt

                        if not env.is_ready:
                            time.sleep(0.05)
                            continue

                        obs = env.get_obs()
                        robot_state = env.get_robot_state()
                        current_pose = np.asarray(robot_state['ActualTCPPose'], dtype=np.float32)

                        if use_gripper and input_device == 'spacemouse':
                            if device.is_button_pressed(0) and not device.is_button_pressed(1):
                                gripper_pose = 0.0
                            elif not device.is_button_pressed(0) and device.is_button_pressed(1):
                                gripper_pose = 1.0

                        press_events = key_counter.get_press_events()
                        filtered_events = []
                        seen_control_keys = set()
                        for key_stroke in press_events:
                            if key_stroke in seen_control_keys:
                                continue
                            seen_control_keys.add(key_stroke)
                            filtered_events.append(key_stroke)

                        for key_stroke in filtered_events:
                            if key_stroke == KeyCode(char='c'):
                                env.start_episode(
                                    t_start + (iter_idx + 2) * dt - time.monotonic() + time.time()
                                )
                                key_counter.clear()
                                is_recording = True
                                print('Recording!')
                            elif key_stroke == KeyCode(char='s'):
                                env.end_episode()
                                key_counter.clear()
                                is_recording = False
                                if not bimanual:
                                    append_pose_trajectory(intermediate_pose, current_pose, HOME_POSE)
                                print('Stopped.')
                            elif key_stroke == Key.backspace:
                                env.drop_episode()
                                key_counter.clear()
                                is_recording = False
                            elif key_stroke == KeyCode(char='h'):
                                if bimanual:
                                    intermediate_pose.clear()
                                    append_pose_trajectory(
                                        intermediate_pose,
                                        current_pose,
                                        bimanual_home_pose,
                                    )
                                    if input_device == 'quest':
                                        device.set_hold_pose(bimanual_home_pose, clear_reference=True)
                                    key_counter.clear()
                                else:
                                    intermediate_pose.clear()
                                    append_pose_trajectory(intermediate_pose, current_pose, HOME_POSE)
                                    key_counter.clear()
                            elif key_stroke == KeyCode(char='t'):
                                if bimanual:
                                    print('Top shortcut is not configured for bimanual mode.')
                                else:
                                    intermediate_pose.clear()
                                    append_pose_trajectory(intermediate_pose, current_pose, TOP_POSE)
                                    print(
                                        f"[teleop] t pressed | current_pose={np.round(current_pose, 5).tolist()} "
                                        f"| top_pose={np.round(TOP_POSE, 5).tolist()} "
                                        f"| queued_waypoints={len(intermediate_pose)}"
                                    )
                                    key_counter.clear()

                        if any(key_stroke == KeyCode(char='q') for key_stroke in filtered_events):
                            if is_recording:
                                env.end_episode()
                                is_recording = False
                            stop = True

                        stage = key_counter[Key.space]

                        vis_img = build_vis(obs, camera_idx)
                        if vis_img is not None:
                            status = f'Episode: {env.replay_buffer.n_episodes}, Stage: {stage}'
                            if is_recording:
                                status += ', Recording'
                            cv2.putText(
                                vis_img,
                                status,
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                (255, 255, 255),
                                2,
                            )
                            if gui_enabled:
                                if not safe_imshow('ur5e_collect', vis_img):
                                    gui_enabled = False
                                    print('OpenCV GUI became unavailable, continuing in no_gui mode.')

                        precise_wait(t_sample)

                        if input_device == 'quest':
                            target_pose, quest_active = device.get_motion_state(current_pose)
                            if not quest_active.any() and intermediate_pose:
                                target_pose = np.asarray(intermediate_pose.pop(0), dtype=np.float32)
                        else:
                            device_state = device.get_motion_state_transformed()
                            dpos = device_state[:3] * (env.max_pos_speed / frequency)
                            drot_xyz = device_state[3:] * (env.max_rot_speed / frequency)
                            drot_xyz[0:2] = 0
                            drot = st.Rotation.from_euler('xyz', drot_xyz)

                            if intermediate_pose:
                                target_pose = np.asarray(intermediate_pose.pop(0), dtype=np.float32)
                            else:
                                target_pose[:3] += dpos
                                target_pose[3:6] = (
                                    drot * st.Rotation.from_rotvec(target_pose[3:6])
                                ).as_rotvec()

                            if use_gripper and target_pose.shape[-1] >= 7:
                                target_pose[6] = gripper_pose

                        if not dummy_robot:
                            current_joint = np.asarray(robot_state['ActualQ'], dtype=np.float32)
                            if use_gripper and current_joint.shape[-1] == 6:
                                current_joint = np.concatenate(
                                    [current_joint, np.array([gripper_pose], dtype=np.float32)]
                                )
                            elif not use_gripper and not bimanual and current_joint.shape[-1] > 6:
                                current_joint = current_joint[:6]

                            action_timestamp = t_command_target - time.monotonic() + time.time()
                            fallback_used = False

                            if control_mode == 'joint':
                                if bimanual:
                                    arm_dim = target_pose.shape[-1] // 2
                                    joint_dim = current_joint.shape[-1] // 2
                                    target_joint_parts = []
                                    fallback_flags = []
                                    arm_controllers = (env.robot_l, env.robot_r)
                                    for arm_idx, controller in enumerate(arm_controllers):
                                        pose_slice = slice(arm_idx * arm_dim, (arm_idx + 1) * arm_dim)
                                        joint_slice = slice(arm_idx * joint_dim, (arm_idx + 1) * joint_dim)
                                        arm_target_pose = np.asarray(target_pose[pose_slice], dtype=np.float32)
                                        arm_current_joint = np.asarray(current_joint[joint_slice], dtype=np.float32)
                                        try:
                                            arm_target_joint = np.asarray(
                                                controller.get_inverse_kinematics(arm_target_pose),
                                                dtype=np.float32,
                                            )
                                            if not np.isfinite(arm_target_joint).all():
                                                raise RuntimeError('IK returned non-finite joint values.')
                                            fallback_flags.append(False)
                                        except Exception as ik_error:
                                            print(
                                                f'[teleop] arm={arm_idx} IK failed, fallback to current joints: '
                                                f'{ik_error}'
                                            )
                                            arm_target_joint = arm_current_joint.copy()
                                            fallback_flags.append(True)
                                        target_joint_parts.append(arm_target_joint)
                                    target_joint = np.concatenate(target_joint_parts, axis=0)
                                    fallback_used = np.asarray(fallback_flags, dtype=bool)
                                else:
                                    try:
                                        target_joint = np.asarray(
                                            env.robot.get_inverse_kinematics(target_pose),
                                            dtype=np.float32,
                                        )
                                        if not np.isfinite(target_joint).all():
                                            raise RuntimeError('IK returned non-finite joint values.')
                                    except Exception as ik_error:
                                        print(f'[teleop] IK failed, fallback to current joints: {ik_error}')
                                        target_joint = current_joint.copy()
                                        fallback_used = True

                                env.exec_actions(
                                    joint_actions=[target_joint],
                                    eef_actions=[target_pose],
                                    mode='joint',
                                    timestamps=[action_timestamp],
                                    stages=[stage],
                                )
                            else:
                                env.exec_actions(
                                    joint_actions=[current_joint],
                                    eef_actions=[target_pose],
                                    mode='eef',
                                    timestamps=[action_timestamp],
                                    stages=[stage],
                                )
                            env.record_teleop_quality(
                                fallback_used=[fallback_used],
                                timestamps=[action_timestamp],
                            )

                        precise_wait(t_cycle_end)
                        iter_idx += 1

    if not no_gui:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == '__main__':
    main()
