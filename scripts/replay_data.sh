#!/bin/bash
set -euo pipefail

DATASET_PATH=${1:?Usage: bash scripts/replay_data.sh <dataset_path_or_raw_dir> <episode_index>}
EPISODE=${2:?Usage: bash scripts/replay_data.sh <dataset_path_or_raw_dir> <episode_index>}

ACTION_MODE=${UR5E_ACTION_MODE:-}
BIMANUAL=${UR5E_BIMANUAL:-false}
ROBOT_IP=${UR5E_ROBOT_IP:-192.168.1.5}
LEFT_IP=${UR5E_LEFT_IP:-192.168.1.3}
RIGHT_IP=${UR5E_RIGHT_IP:-192.168.1.5}
FREQUENCY=${UR5E_REPLAY_FREQUENCY:-10}
SPEED_SLIDER=${UR5E_REPLAY_SPEED_SLIDER:-0.6}
USE_GRIPPER=${UR5E_USE_GRIPPER:-true}
FILTER_STATIC=${UR5E_FILTER_STATIC:-false}
DRY_RUN=${UR5E_DRY_RUN:-false}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    PYTHON_BIN_DEFAULT="${CONDA_PREFIX}/bin/python"
else
    PYTHON_BIN_DEFAULT=python
fi
PYTHON_BIN=${PYTHON_BIN:-${PYTHON_BIN_DEFAULT}}

CMD=("$PYTHON_BIN" scripts/replay_dataset.py "$DATASET_PATH" --episode "$EPISODE" --frequency "$FREQUENCY" --speed_slider_value "$SPEED_SLIDER")

if [[ -n "${ACTION_MODE}" ]]; then
    CMD+=(--control_mode "$ACTION_MODE")
fi

if [[ "${BIMANUAL}" == "true" ]]; then
    CMD+=(--bimanual --robot_left_ip "$LEFT_IP" --robot_right_ip "$RIGHT_IP")
else
    CMD+=(--single_arm --robot_ip "$ROBOT_IP")
fi

if [[ "${USE_GRIPPER}" == "true" ]]; then
    CMD+=(--use_gripper)
else
    CMD+=(--no-use_gripper)
fi

if [[ "${FILTER_STATIC}" == "true" ]]; then
    CMD+=(--filter_static)
else
    CMD+=(--no-filter_static)
fi

if [[ "${DRY_RUN}" == "true" ]]; then
    CMD+=(--dry_run)
fi

"${CMD[@]}"
