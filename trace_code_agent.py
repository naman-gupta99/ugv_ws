import os
import time
from ugv_tools.ugv_tools.agent.code_execution_agent import execute_code_audit

metrics = {
    "agent_turns": 0,
    "tools_transitions": 0,
    "missions_completed": 0,
    "start_time": None,
    "end_time": None,
    "duration_sec": None,
}

def print_agent_metrics():
    print("[Code Agent] metrics:", metrics)

# Keep the run short for a quick trace check.
os.environ["UGV_CODE_AGENT_MAX_STEPS"] = "6"
# Make sure tracing envs are set
os.environ.setdefault('LANGSMITH_TRACING', 'true')
os.environ.setdefault('LANGSMITH_PROJECT', 'UGV Testing')

print("Starting code execution audit (trace test)")
execute_code_audit(metrics, print_agent_metrics, llm_model_name=None, hint='')
print("Finished code execution audit (trace test)")
