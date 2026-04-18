from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class _QuestArmState:
    hand: str
    prefix: str
    gripper_target: float
    control_active: bool = False
    reference_quest_pose: Optional[np.ndarray] = None
    reference_tcp_pos: Optional[np.ndarray] = None
    reference_tcp_rot: Optional[np.ndarray] = None
    last_target_tcp: Optional[np.ndarray] = None

    @property
    def trigger_key(self) -> str:
        return 'leftTrig' if self.hand == 'l' else 'rightTrig'

    @property
    def gripper_open_key(self) -> str:
        return 'Y' if self.hand == 'l' else 'B'

    @property
    def gripper_close_key(self) -> str:
        return 'X' if self.hand == 'l' else 'A'


class QuestTeleop:
    """Quest controller wrapper that outputs single- or dual-arm UR TCP targets."""

    _quest2ur = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    _ur2quest = np.linalg.inv(_quest2ur)

    def __init__(
        self,
        bimanual=False,
        single_hand='r',
        translation_scale=3.0,
        use_gripper=True,
        gripper_initial_pos=1.0,
        trigger_threshold=0.5,
    ):
        self.bimanual = bool(bimanual)
        self.single_hand = str(single_hand)
        self.translation_scale = float(translation_scale)
        self.use_gripper = bool(use_gripper)
        self.trigger_threshold = float(trigger_threshold)
        initial_gripper = float(np.clip(gripper_initial_pos, 0.0, 1.0))

        if self.single_hand not in ('l', 'r'):
            raise ValueError("`single_hand` must be either 'l' or 'r'.")
        if self.translation_scale <= 0:
            raise ValueError("`translation_scale` must be positive.")

        if self.bimanual:
            self._arms = [
                _QuestArmState(hand='l', prefix='left_', gripper_target=initial_gripper),
                _QuestArmState(hand='r', prefix='right_', gripper_target=initial_gripper),
            ]
        else:
            self._arms = [
                _QuestArmState(hand=self.single_hand, prefix='', gripper_target=initial_gripper)
            ]

        self._reader = None

    def start(self, wait=True):
        del wait
        try:
            from oculus_reader.reader import OculusReader
        except ImportError as exc:
            raise ImportError(
                "Quest teleoperation requires the `oculus_reader` package in the current environment."
            ) from exc

        self._reader = OculusReader()
        return self

    def stop(self, wait=True):
        del wait
        self._reader = None
        for arm in self._arms:
            arm.control_active = False
            arm.reference_quest_pose = None
            arm.reference_tcp_pos = None
            arm.reference_tcp_rot = None
            arm.last_target_tcp = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get_motion_state(self, current_tcp_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._reader is None:
            raise RuntimeError("Quest teleop is not started.")

        pose_data, button_data = self._reader.get_transformations_and_buttons()
        current_tcp_pose = np.asarray(current_tcp_pose, dtype=np.float64)
        arm_dim = current_tcp_pose.shape[-1] // len(self._arms)

        targets = []
        active = []
        for arm_idx, arm in enumerate(self._arms):
            start = arm_idx * arm_dim
            end = start + arm_dim
            arm_tcp = current_tcp_pose[start:end]
            target_tcp, arm_active = self._get_arm_target(arm, arm_tcp, pose_data, button_data)
            targets.append(target_tcp)
            active.append(arm_active)

        return np.concatenate(targets, axis=0).astype(np.float32), np.asarray(active, dtype=bool)

    def set_hold_pose(self, target_tcp_pose: np.ndarray, clear_reference: bool = True) -> None:
        target_tcp_pose = np.asarray(target_tcp_pose, dtype=np.float64)
        arm_dim = target_tcp_pose.shape[-1] // len(self._arms)
        for arm_idx, arm in enumerate(self._arms):
            start = arm_idx * arm_dim
            end = start + arm_dim
            arm.last_target_tcp = target_tcp_pose[start:end].copy()
            if self.use_gripper and arm.last_target_tcp.shape[-1] >= 7:
                arm.gripper_target = float(arm.last_target_tcp[6])
            if clear_reference:
                arm.control_active = False
                arm.reference_quest_pose = None
                arm.reference_tcp_pos = None
                arm.reference_tcp_rot = None

    def _get_arm_target(
        self,
        arm: _QuestArmState,
        current_tcp: np.ndarray,
        pose_data: Optional[Dict[str, Any]],
        button_data: Optional[Dict[str, Any]],
    ) -> Tuple[np.ndarray, bool]:
        tcp_dim = 7 if self.use_gripper and current_tcp.shape[-1] >= 7 else 6
        current_tcp = np.asarray(current_tcp[:tcp_dim], dtype=np.float64)

        if arm.last_target_tcp is None:
            arm.last_target_tcp = current_tcp.copy()

        self._update_gripper_target(arm, button_data)
        hold_tcp = arm.last_target_tcp.copy()
        if self.use_gripper and tcp_dim == 7:
            hold_tcp[6] = arm.gripper_target

        trigger_value = self._trigger_value(arm, button_data)
        if trigger_value <= self.trigger_threshold or not pose_data or arm.hand not in pose_data:
            arm.control_active = False
            arm.reference_quest_pose = None
            arm.reference_tcp_pos = None
            arm.reference_tcp_rot = None
            arm.last_target_tcp = hold_tcp
            return hold_tcp, False

        current_quest_pose = np.asarray(pose_data[arm.hand], dtype=np.float64)
        if not arm.control_active:
            arm.control_active = True
            arm.reference_quest_pose = current_quest_pose.copy()
            arm.reference_tcp_pos = current_tcp[:3].copy()
            arm.reference_tcp_rot = R.from_rotvec(current_tcp[3:6]).as_matrix()
            anchored_tcp = current_tcp.copy()
            if self.use_gripper and tcp_dim == 7:
                anchored_tcp[6] = arm.gripper_target
            arm.last_target_tcp = anchored_tcp
            return anchored_tcp, True

        target_tcp = self._map_quest_pose_to_tcp(arm, current_quest_pose, tcp_dim)
        arm.last_target_tcp = target_tcp
        return target_tcp, True

    def _map_quest_pose_to_tcp(
        self,
        arm: _QuestArmState,
        current_quest_pose: np.ndarray,
        tcp_dim: int,
    ) -> np.ndarray:
        assert arm.reference_quest_pose is not None
        assert arm.reference_tcp_pos is not None
        assert arm.reference_tcp_rot is not None

        delta_rot = current_quest_pose[:3, :3] @ np.linalg.inv(arm.reference_quest_pose[:3, :3])
        delta_pos = current_quest_pose[:3, 3] - arm.reference_quest_pose[:3, 3]

        delta_pos_ur = self._quest2ur @ delta_pos * self.translation_scale
        delta_pos_ur[0] *= -1.0
        delta_pos_ur[1] *= -1.0

        delta_rot_ur = self._quest2ur @ delta_rot @ self._ur2quest
        delta_rotvec_ur = R.from_matrix(delta_rot_ur).as_rotvec()
        delta_rotvec_ur *= -1.0
        delta_rot_ur = R.from_rotvec(delta_rotvec_ur).as_matrix()

        next_tcp = np.zeros((tcp_dim,), dtype=np.float64)
        next_tcp[:3] = arm.reference_tcp_pos + delta_pos_ur
        next_tcp[3:6] = R.from_matrix(delta_rot_ur @ arm.reference_tcp_rot).as_rotvec()
        if self.use_gripper and tcp_dim == 7:
            next_tcp[6] = arm.gripper_target
        return next_tcp

    def _update_gripper_target(self, arm: _QuestArmState, button_data: Optional[Dict[str, Any]]) -> None:
        if not self.use_gripper or not button_data:
            return
        if bool(button_data.get(arm.gripper_open_key, False)):
            arm.gripper_target = 1.0
        if bool(button_data.get(arm.gripper_close_key, False)):
            arm.gripper_target = 0.0
        arm.gripper_target = float(np.clip(arm.gripper_target, 0.0, 1.0))

    def _trigger_value(self, arm: _QuestArmState, button_data: Optional[Dict[str, Any]]) -> float:
        if not button_data:
            return 0.0
        state = button_data.get(arm.trigger_key, (0.0,))
        if isinstance(state, (list, tuple, np.ndarray)):
            return float(state[0]) if len(state) > 0 else 0.0
        return float(state)
