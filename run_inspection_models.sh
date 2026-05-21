#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_FILE="${WORKSPACE_DIR}/install/setup.bash"

if [[ -f "${SETUP_FILE}" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "${SETUP_FILE}"
  set -u
fi

export PYTHONPATH="${WORKSPACE_DIR}/src/ugv_main/ugv_tools:${PYTHONPATH:-}"

RUNS=(
#   "greedy|"
#   "code-gemini-2.5-pro|easy"
  "gemini-2.5-flash-lite|easy"
#   "gemini-2.5-pro|easy"
)

MODELS=(
#   "greedy"
#   "code-gemini-2.5-pro"
  "gemini-2.5-flash-lite"
#   "gemini-2.5-pro"
)

echo "Testing ${#MODELS[@]} selected inspection model(s) before starting ROS runs..."
python3 -m ugv_tools.model_smoke_test "${MODELS[@]}"

echo
echo "Starting inspection loop with ${#RUNS[@]} selected run(s)..."
for run in "${RUNS[@]}"; do
  IFS='|' read -r model_name hint_name <<< "${run}"
  hint_label="${hint_name:-<none>}"

  echo
  echo "============================================================"
  echo "Running inspection with model: ${model_name}, hint: ${hint_label}"
  echo "============================================================"
  UGV_AGENT_MODEL="${model_name}" UGV_AGENT_HINT="${hint_name}" python3 -m ugv_tools.run_inspection --no-debug --model "${model_name}" --hint "${hint_name}"
done

echo
echo "Inspection loop complete."
