#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./run_rover_camera_only_inspection.sh X_MIN X_MAX Y_MIN Y_MAX [MODEL] [HINT]

Example:
  ./run_rover_camera_only_inspection.sh -1 1 0 1 gemini-2.5-pro easy

Environment overrides:
  UGV_AGENT_MODEL, UGV_AGENT_HINT, UGV_CAPTURE_DIR, UGV_METRICS_CSV
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 4 || $# -gt 6 ]]; then
  usage >&2
  exit 2
fi

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_FILE="${WORKSPACE_DIR}/install/setup.bash"

if [[ -f "${SETUP_FILE}" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "${SETUP_FILE}"
  set -u
fi

export PYTHONPATH="${WORKSPACE_DIR}/src/ugv_main/ugv_tools:${PYTHONPATH:-}"

export UGV_INSPECTION_MODE="ROVER_CAMERA_ONLY"
export UGV_PLATFORM="ROVER"
export PLATFORM="ROVER"

export UGV_CAMERA_ONLY_X_MIN="$1"
export UGV_CAMERA_ONLY_X_MAX="$2"
export UGV_CAMERA_ONLY_X_MX="$2"
export UGV_CAMERA_ONLY_Y_MIN="$3"
export UGV_CAMERA_ONLY_Y_MAX="$4"

export UGV_AGENT_MODEL="${5:-${UGV_AGENT_MODEL:-gemini-2.5-pro}}"
export UGV_AGENT_HINT="${6:-${UGV_AGENT_HINT:-}}"
export UGV_GREEDY="$([[ "${UGV_AGENT_MODEL}" == "greedy" ]] && echo true || echo false)"
export UGV_CODE_AGENT="$([[ "${UGV_AGENT_MODEL}" == "code" || "${UGV_AGENT_MODEL}" == code-* ]] && echo true || echo false)"

echo "Starting ROVER_CAMERA_ONLY inspection:"
echo "  target area: x=[${UGV_CAMERA_ONLY_X_MIN},${UGV_CAMERA_ONLY_X_MAX}] y=[${UGV_CAMERA_ONLY_Y_MIN},${UGV_CAMERA_ONLY_Y_MAX}]"
echo "  model: ${UGV_AGENT_MODEL}"
echo "  hint: ${UGV_AGENT_HINT:-<none>}"

ros2 run ugv_tools inspection_pipeline
