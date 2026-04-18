#!/bin/bash
set -euo pipefail

OUTPUT_DIR=${1:-data/ur5e_raw}
CAMERA_IDX=${UR5E_CAMERA_IDX:-0}
ACTION_MODE=${UR5E_ACTION_MODE:-eef}
INPUT_DEVICE=${UR5E_INPUT_DEVICE:-spacemouse}
BIMANUAL=${UR5E_BIMANUAL:-false}
LEFT_IP=${UR5E_LEFT_IP:-192.168.1.3}
RIGHT_IP=${UR5E_RIGHT_IP:-192.168.1.5}
QUEST_SINGLE_HAND=${UR5E_QUEST_SINGLE_HAND:-r}
QUEST_TRANSLATION_SCALE=${UR5E_QUEST_TRANSLATION_SCALE:-3.0}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

python scripts/collect_data.py \
    --output "$OUTPUT_DIR" \
    --robot_ip "192.168.1.5" \
    --camera_idx "$CAMERA_IDX" \
    --control_mode "$ACTION_MODE" \
    --input_device "$INPUT_DEVICE" \
    --robot_left_ip "$LEFT_IP" \
    --robot_right_ip "$RIGHT_IP" \
    --quest_single_hand "$QUEST_SINGLE_HAND" \
    --quest_translation_scale "$QUEST_TRANSLATION_SCALE" \
    $( [ "$BIMANUAL" = "true" ] && printf '%s' '--bimanual' || printf '%s' '--single_arm' )
