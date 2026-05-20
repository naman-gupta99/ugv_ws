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

mapfile -t MODELS < <(
  python3 - <<'PY'
from ugv_tools.agent.models import Models

for model_name in Models().available_model_names():
    print(model_name)
PY
)

mapfile -t HINTS < <(
  python3 - <<'PY'
from ugv_tools.agent.hints import hints

for hint_name in hints.keys():
    print(hint_name)
PY
)

if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "No models found in ugv_tools.agent.models.Models." >&2
  exit 1
fi

echo "Testing ${#MODELS[@]} inspection model(s) before starting ROS runs..."
python3 -m ugv_tools.model_smoke_test "${MODELS[@]}"

echo
echo "Starting inspection loop with ${#HINTS[@]} hint(s)..."
for model_name in "${MODELS[@]}"; do
  if [[ "${model_name}" == "greedy" ]]; then
    echo
    echo "============================================================"
    echo "Running inspection with model: ${model_name}, hint: <none>"
    echo "============================================================"
    UGV_AGENT_MODEL="${model_name}" UGV_AGENT_HINT="" python3 -m ugv_tools.run_inspection --no-debug --model "${model_name}" --hint ""
    continue
  fi

  for hint_name in "${HINTS[@]}"; do
    echo
    echo "============================================================"
    echo "Running inspection with model: ${model_name}, hint: ${hint_name}"
    echo "============================================================"
    UGV_AGENT_MODEL="${model_name}" UGV_AGENT_HINT="${hint_name}" python3 -m ugv_tools.run_inspection --no-debug --model "${model_name}" --hint "${hint_name}"
  done
done

echo
echo "Inspection loop complete."
