#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
lerobot_root="${LEROBOT_ROOT:-/home/yechen/UR5e-LeRobot}"

show_help() {
    cat <<'EOF'
Usage:
  bash scripts/convert_to_lerobot.sh [options]

Options are forwarded to scripts/convert_to_lerobot.py. Common options:
  --input PATH              Input raw replay_buffer.zarr or raw directory
  --output PATH             Output LeRobot dataset root
  --action-mode joint|eef   Conversion mode, default: joint
  --repo-id REPO_ID         LeRobot dataset repo id metadata
  --fps FPS                 Dataset fps, default: 10
  --task TEXT               Task description saved with every frame
  --use-videos true|false   Store visual feature as video or image, default: true
  --max-episodes N          Convert only the first N episodes
  --overwrite               Replace output directory if it already exists

Environment:
  PYTHON=/path/to/python    Python with zarr and LeRobot dependencies
  LEROBOT_ROOT=/path/repo   LeRobot repo, default: /home/yechen/UR5e-LeRobot
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_help
    exit 0
fi

if [[ -n "${PYTHON:-}" ]]; then
    python_bin="${PYTHON}"
else
    python_bin="$(command -v python3 || command -v python)"
fi

if [[ ! -d "${lerobot_root}/src" ]]; then
    echo "LeRobot source directory not found: ${lerobot_root}/src" >&2
    echo "Set LEROBOT_ROOT=/path/to/UR5e-LeRobot and retry." >&2
    exit 1
fi

export PYTHONPATH="${lerobot_root}/src:${repo_root}/dp-family:${PYTHONPATH:-}"

cd "${repo_root}"
exec "${python_bin}" scripts/convert_to_lerobot.py "$@"
