#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the full UGV inspection launch."
    )
    debug_group = parser.add_mutually_exclusive_group()
    debug_group.add_argument(
        "--debug",
        action="store_true",
        help="Wait for debugpy to attach to inspection_pipeline.",
    )
    debug_group.add_argument(
        "--no-debug",
        action="store_true",
        help="Run inspection_pipeline normally.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("UGV_AGENT_MODEL", "gemini-2.5-pro"),
        help="Agent mode or LLM model name. Use 'greedy', 'code', or 'code-<llm-name>' for controller-level agents.",
    )
    parser.add_argument(
        "--nav-ready-delay",
        type=float,
        help="Seconds to wait before starting behavior_ctrl.",
    )
    parser.add_argument(
        "--pipeline-delay",
        type=float,
        help="Seconds to wait before starting inspection_pipeline.",
    )
    parser.add_argument(
        "--use-sim-time",
        default="true",
        choices=("true", "false"),
        help="Pass use_sim_time to the inspection launch file.",
    )
    parser.add_argument(
        "--hint",
        default=os.environ.get("UGV_AGENT_HINT", ""),
        help="Strategy hint for agents (e.g., 'easy', 'medium', 'hard'). Empty string means no hint.",
    )
    args = parser.parse_args(argv)

    cmd = [
        "ros2",
        "launch",
        "ugv_tools",
        "inspection.launch.py",
        f"debug_inspection_pipeline:={'true' if args.debug else 'false'}",
        f"agent_model:={args.model}",
        f"use_sim_time:={args.use_sim_time}",
    ]
    if args.nav_ready_delay is not None:
        cmd.append(f"nav_ready_delay:={args.nav_ready_delay}")
    if args.pipeline_delay is not None:
        cmd.append(f"pipeline_delay:={args.pipeline_delay}")

    env = os.environ.copy()
    env["UGV_AGENT_MODEL"] = args.model
    env["UGV_GREEDY"] = "true" if args.model == "greedy" else "false"
    env["UGV_CODE_AGENT"] = "true" if args.model == "code" or args.model.startswith("code-") else "false"
    env["UGV_AGENT_HINT"] = args.hint
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
