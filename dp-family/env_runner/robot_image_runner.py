import time

import cv2
import numpy as np
import torch

from common.pytorch_util import dict_apply
from env_runner.base_runner import BaseRunner
from policy.base_policy import BasePolicy
from common.trans_utils import interpolate_poses
from real_world.keystroke_counter import KeystrokeCounter, KeyCode


HOME_POSE = np.array([-0.11130, -0.48927, 0.22326, 3.152, -0.007, -0.001, 1.0], dtype=np.float32)


class RobotImageRunner(BaseRunner):
    def __init__(
        self,
        output_dir,
        eval_episodes=1,
        max_steps=200,
        n_obs_steps=2,
        n_action_steps=16,
        image_size=(224, 224),
        task_name=None,
        frequency=10,
        action_exec_latency=0.05,
        robot_ip="192.168.1.5",
        robot_left_ip=None,
        robot_right_ip=None,
        camera_indices=(0,),
        single_arm_type="right",
        is_bimanual=False,
        use_gripper=True,
        dummy_robot=False,
        speed_slider_value=1.0,
        dry_run=False,
        interactive_control=False,
        action_mode="eef",
    ):
        super().__init__(output_dir)
        self.task_name = task_name
        self.eval_episodes = eval_episodes
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.image_size = tuple(image_size)
        self.frequency = frequency
        self.action_exec_latency = action_exec_latency
        self.robot_ip = robot_ip
        self.robot_left_ip = robot_ip if robot_left_ip is None else robot_left_ip
        self.robot_right_ip = robot_ip if robot_right_ip is None else robot_right_ip
        self.camera_indices = list(camera_indices)
        self.single_arm_type = single_arm_type
        self.is_bimanual = is_bimanual
        self.use_gripper = use_gripper
        self.dummy_robot = dummy_robot
        self.speed_slider_value = speed_slider_value
        self.dry_run = dry_run
        self.interactive_control = interactive_control
        if action_mode not in ("eef", "joint"):
            raise ValueError(f"Unsupported action_mode: {action_mode}")
        self.action_mode = action_mode
        self.agent_pos_key = "robot_joint" if action_mode == "joint" else "robot_eef_pose"
        self.gripper_close_threshold = 0.8

    def _ensure_real_robot_imports(self):
        from real_world.real_ur5e_env import RealUR5eEnv

        return RealUR5eEnv

    def _pad_obs_steps(self, array):
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        if array.shape[0] == self.n_obs_steps:
            return array
        if array.shape[0] > self.n_obs_steps:
            return array[-self.n_obs_steps :]
        pad = np.repeat(array[:1], self.n_obs_steps - array.shape[0], axis=0)
        return np.concatenate([pad, array], axis=0)

    def _resize_frames(self, frames):
        target_h, target_w = self.image_size
        resized = [
            cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            for frame in frames
        ]
        return np.stack(resized, axis=0).astype(np.float32)

    def build_policy_obs(self, real_obs):
        if self.agent_pos_key not in real_obs:
            raise KeyError(
                f'Missing key "{self.agent_pos_key}" in real observation. '
                f"Available keys: {list(real_obs.keys())}"
            )
        obs = {
            "agent_pos": self._pad_obs_steps(real_obs[self.agent_pos_key]).astype(np.float32),
        }
        for camera_idx in self.camera_indices:
            raw_key = f"camera_{camera_idx}_color"
            obs_key = f"camera_{camera_idx}"
            if raw_key not in real_obs:
                raise KeyError(
                    f"Configured camera {camera_idx} is missing from real observation. "
                    f"Expected key {raw_key}."
                )
            obs[obs_key] = self._resize_frames(real_obs[raw_key])
        return obs

    def get_action(self, policy: BasePolicy, obs=None):
        if obs is None:
            raise ValueError("Observation is required for policy inference.")

        device = policy.device
        obs_dict = dict_apply(obs, lambda x: torch.from_numpy(x).to(device=device))
        obs_dict_input = {key: value.unsqueeze(0) for key, value in obs_dict.items()}
        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict_input)
        np_action_dict = dict_apply(action_dict, lambda x: x.detach().cpu().numpy())
        return np_action_dict["action"].squeeze(0).astype(np.float32)

    def _sanitize_action_sequence(self, action_seq, current_pose):
        action_seq = np.asarray(action_seq, dtype=np.float32)
        if action_seq.ndim == 1:
            action_seq = action_seq[None, :]
        expected_dim = current_pose.shape[-1]
        if action_seq.shape[-1] != expected_dim or not np.isfinite(action_seq).all():
            print(
                f"[RobotImageRunner] Invalid action sequence; "
                f"shape={action_seq.shape}, expected_dim={expected_dim}, "
                f"finite={bool(np.isfinite(action_seq).all())}. Falling back to current pose."
            )
            return np.repeat(current_pose[None, :], self.n_action_steps, axis=0).astype(np.float32)
        if self.action_mode == "eef" and self.use_gripper and action_seq.shape[-1] >= 7:
            # Training labels for the gripper are binary (0/1), so force the
            # diffusion output back onto that manifold during robot execution.
            gripper_indices = [6]
            if action_seq.shape[-1] >= 14:
                gripper_indices.append(13)
            for idx in gripper_indices:
                action_seq[:, idx] = (
                    action_seq[:, idx] >= self.gripper_close_threshold
                ).astype(np.float32)
        return action_seq

    def _make_env(self):
        RealUR5eEnv = self._ensure_real_robot_imports()
        return RealUR5eEnv(
            output_dir=self.output_dir,
            robot_left_ip=self.robot_left_ip,
            robot_right_ip=self.robot_right_ip,
            frequency=self.frequency,
            n_obs_steps=self.n_obs_steps,
            debug=False,
            dummy_robot=self.dummy_robot,
            enable_depth=True,
            enable_pose=True,
            single_arm_type=self.single_arm_type,
            ctrl_mode=self.action_mode,
            use_gripper=self.use_gripper,
            speed_slider_value=self.speed_slider_value,
            is_bimanual=self.is_bimanual,
        )

    def run(self, policy: BasePolicy):
        return {"test_mean_score": 0.0}

    def run_robot(self, policy: BasePolicy):
        with self._make_env() as env:
            if self.interactive_control:
                with KeystrokeCounter() as key_counter:
                    self._run_robot_loop(env, policy, key_counter=key_counter)
            else:
                self._run_robot_loop(env, policy, key_counter=None)

    def _handle_key_events(self, env, policy, key_counter, running):
        for key_stroke in key_counter.get_press_events():
            if key_stroke == KeyCode(char="q"):
                print("[RobotImageRunner] q pressed, exiting inference.")
                return running, True
            if key_stroke == KeyCode(char="c"):
                print("[RobotImageRunner] c pressed, starting/resuming inference.")
                key_counter.clear()
                return True, False
            if key_stroke == KeyCode(char="s"):
                print("[RobotImageRunner] s pressed, pausing inference.")
                key_counter.clear()
                return False, False
            if key_stroke == KeyCode(char="h"):
                print("[RobotImageRunner] h pressed, moving to HOME_POSE and pausing inference.")
                key_counter.clear()
                policy.reset()
                self._move_home(env)
                return False, False
        return running, False

    def _move_home(self, env):
        robot_state = env.get_robot_state()
        current_pose = np.asarray(robot_state["ActualTCPPose"], dtype=np.float32)
        target_pose = HOME_POSE.copy()
        if current_pose.shape[-1] == 6:
            target_pose = target_pose[:6]
        elif current_pose.shape[-1] == 7 and target_pose.shape[-1] == 6:
            target_pose = np.concatenate([target_pose, np.array([1.0], dtype=np.float32)])
        waypoints, _, is_close = interpolate_poses(current_pose, target_pose, speed=0.03)
        if is_close:
            print("[RobotImageRunner] Already near HOME_POSE.")
            return
        waypoints.append(target_pose)
        eef_actions = np.asarray(waypoints, dtype=np.float32)
        timestamps = time.time() + self.action_exec_latency + np.arange(
            len(eef_actions), dtype=np.float64
        ) / float(self.frequency)
        joint_dim = 6 + int(self.use_gripper)
        joint_actions = np.zeros((len(eef_actions), joint_dim), dtype=np.float32)
        if self.dry_run:
            print(f"[RobotImageRunner] dry_run=True, skipping {len(eef_actions)} HOME actions.")
        else:
            env.exec_actions(
                joint_actions=joint_actions,
                eef_actions=eef_actions,
                timestamps=timestamps,
                mode="eef",
            )
            time.sleep(max(len(eef_actions) / float(self.frequency), 1.0 / float(self.frequency)))

    def _run_robot_loop(self, env, policy, key_counter=None):
        policy.reset()
        executed_steps = 0
        running = key_counter is None
        run_until_quit = key_counter is not None and (self.max_steps is None or self.max_steps <= 0)
        if key_counter is not None:
            print("Keyboard: c=start/resume, s=pause, h=home, q=quit")
            print("[RobotImageRunner] Waiting for c to start inference.")
            if run_until_quit:
                print("[RobotImageRunner] max_steps<=0, running until q is pressed.")

        while run_until_quit or executed_steps < self.max_steps:
            if key_counter is not None:
                running, should_quit = self._handle_key_events(env, policy, key_counter, running)
                if should_quit:
                    break
                if not running:
                    time.sleep(0.05)
                    continue

            real_obs = env.get_obs()
            policy_obs = self.build_policy_obs(real_obs)
            current_pose = policy_obs["agent_pos"][-1]
            action_seq = self.get_action(policy, policy_obs)
            action_seq = self._sanitize_action_sequence(action_seq, current_pose)
            if len(action_seq) > 0:
                pos_delta = np.linalg.norm(action_seq[:, :3] - current_pose[None, :3], axis=1)
                if action_seq.shape[-1] >= 7:
                    grip_min = float(action_seq[:, 6].min())
                    grip_max = float(action_seq[:, 6].max())
                else:
                    grip_min = float("nan")
                    grip_max = float("nan")
                print(
                    f"[RobotImageRunner] step={executed_steps} action_shape={action_seq.shape} "
                    f"pos_delta_mean={float(pos_delta.mean()):.4f} pos_delta_max={float(pos_delta.max()):.4f} "
                    f"gripper_min={grip_min:.4f} gripper_max={grip_max:.4f}"
                )

            timestamps = time.time() + self.action_exec_latency + np.arange(
                len(action_seq), dtype=np.float64
            ) / float(self.frequency)
            if self.action_mode == "joint":
                joint_actions = action_seq
                current_eef = np.asarray(real_obs["robot_eef_pose"][-1], dtype=np.float32)
                eef_actions = np.repeat(current_eef[None, :], len(action_seq), axis=0)
            else:
                eef_actions = action_seq
                joint_dim = 6 + int(self.use_gripper)
                joint_actions = np.zeros((len(action_seq), joint_dim), dtype=np.float32)
            if self.dry_run:
                print(f"[RobotImageRunner] dry_run=True, skipping {len(action_seq)} robot actions.")
            else:
                env.exec_actions(
                    joint_actions=joint_actions,
                    eef_actions=eef_actions,
                    timestamps=timestamps,
                    mode=self.action_mode,
                )
            executed_steps += len(action_seq)
            time.sleep(max(len(action_seq) / float(self.frequency), 1.0 / float(self.frequency)))

        if not run_until_quit and executed_steps >= self.max_steps:
            print(f"[RobotImageRunner] Reached max_steps={self.max_steps}, inference finished.")
