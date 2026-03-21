#!/usr/bin/env python
# encoding: utf-8

"""
Multi-waypoint inspection pipeline.

For each goal in INSPECTION_GOALS the rover will:
  1. Navigate to the waypoint (Nav2).
  2. Align perpendicular to the nearest wall.
  3. Capture a camera image, detect objects via REST API, and laterally
     shift until the highest-confidence detection is centred in frame.
  4. Find the ideal inspection distance from the wall.
  5. Align parallel to the wall.
  6. Run the LLM pan-tilt inspection agent.
"""

import os
import threading
import time
import traceback

import requests

CONTINUE_SENTINEL = '/tmp/ugv_continue'

import rclpy
from rclpy.executors import SingleThreadedExecutor

from .align_ctrl import AlignCtrl
from .distance_ctrl import DistanceCtrl
from .llm_pt_ctrl import LlmPtCtrl
from .nav_ctrl import NavCtrl
from .wall_centering import WallCenteringCtrl


# ---------------------------------------------------------------------------
# Inspection waypoints
# Capture these values using goal_spy.py while setting goals in RViz.
# orientation qz/qw: for heading angle θ, qz = sin(θ/2), qw = cos(θ/2).
# ---------------------------------------------------------------------------
INSPECTION_GOALS = [
    {'label': 'Waypoint 1', 'x': 2.8371, 'y': 2.9142, 'qz': 0.1, 'qw': 1.0},
    {'label': 'Waypoint 2', 'x': 2.5260, 'y': -2.6412, 'qz': 0.1, 'qw': 1.0},
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_phase(node, work_fn, startup_delay: float = 0.5) -> None:
    """
    Spin *node* while *work_fn()* runs in a background thread.

    Destroys the node when work is done (or if an exception is raised).
    Re-raises KeyboardInterrupt so the top-level loop can shut down cleanly.
    """
    done = threading.Event()

    def _worker():
        time.sleep(startup_delay)
        try:
            work_fn()
        except Exception:
            traceback.print_exc()
        finally:
            done.set()

    threading.Thread(target=_worker, daemon=True).start()

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        while not done.is_set():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        raise
    finally:
        executor.remove_node(node)
        node.destroy_node()


def _wait_for_continue(phase_name: str) -> None:
    """Pause until the user creates the sentinel file to signal continuation."""
    # Remove any leftover sentinel from a previous phase
    try:
        os.remove(CONTINUE_SENTINEL)
    except FileNotFoundError:
        pass
    # curl -H "Title: Test Alert" -d "If you see this, your ntfy setup is ready for Claude." ntfy.sh/naman_claude_ugv
    requests.post('https://ntfy.sh/naman_claude_ugv', data=f'Phase "{phase_name}" complete. Ready for next phase.')

    print(f'\n  >>> Phase "{phase_name}" complete.')
    print(f'      Run in another terminal to continue:  touch {CONTINUE_SENTINEL}\n')

    while not os.path.exists(CONTINUE_SENTINEL):
        time.sleep(0.5)

    try:
        os.remove(CONTINUE_SENTINEL)
    except FileNotFoundError:
        pass


def _run_inspection_at_goal(goal: dict) -> None:
    """Run the full six-phase inspection pipeline at one waypoint."""
    label = goal.get('label', f"({goal['x']:.2f}, {goal['y']:.2f})")

    # ------------------------------------------------------------------
    # Phase 1: Navigate to waypoint
    # ------------------------------------------------------------------
    print(f'  [Phase 1] Navigating to {label}')
    nav_node = NavCtrl()
    _run_phase(
        nav_node,
        lambda: nav_node.navigate_to(goal['x'], goal['y'], goal['qz'], goal['qw']),
        startup_delay=1.0,
    )
    time.sleep(2.0)

    # ------------------------------------------------------------------
    # Phase 2: Align perpendicular to wall
    # ------------------------------------------------------------------
    print(f'  [Phase 2] Aligning perpendicular to wall')
    align_node = AlignCtrl('perpendicular')
    _run_phase(align_node, align_node.align, startup_delay=1.0)
    time.sleep(5.0)

    # ------------------------------------------------------------------
    # Phase 3: Centre on highest-confidence detection
    #   - Captures a camera image and posts it to the detection API.
    #   - If the best detection is off-centre, the rover shifts laterally
    #     until the target is within CENTERING_MARGIN_PX of image centre.
    # ------------------------------------------------------------------
    print(f'  [Phase 3] Centering on wall detection')
    centering_node = WallCenteringCtrl()
    _run_phase(centering_node, centering_node.run, startup_delay=1.0)
    time.sleep(2.0)
    
    # ------------------------------------------------------------------
    # Phase 4: Align perpendicular to wall
    # ------------------------------------------------------------------
    print(f'  [Phase 4] Aligning perpendicular to wall')
    align_node = AlignCtrl('perpendicular')
    _run_phase(align_node, align_node.align, startup_delay=1.0)
    time.sleep(5.0)

    # ------------------------------------------------------------------
    # Phase 5: Find ideal inspection distance from wall
    # ------------------------------------------------------------------
    print(f'  [Phase 5] Finding ideal inspection distance')
    dist_node = DistanceCtrl()

    def _find_distance():
        best = dist_node.find_accessible_distance()
        if best is None:
            dist_node.get_logger().error('No accessible distance found — continuing.')

    _run_phase(dist_node, _find_distance, startup_delay=0.5)
    time.sleep(5.0)

    # ------------------------------------------------------------------
    # Phase 6: Align parallel to wall
    # ------------------------------------------------------------------
    print(f'  [Phase 6] Aligning parallel to wall')
    align_node = AlignCtrl('parallel')
    _run_phase(align_node, align_node.align, startup_delay=1.0)
    time.sleep(10.0)

    # ------------------------------------------------------------------
    # Phase 7: LLM pan-tilt inspection (runs until agent finishes)
    # ------------------------------------------------------------------
    print(f'  [Phase 7] Running LLM inspection agent')
    pt_ctrl = LlmPtCtrl('llm_pt_ctrl')
    executor = SingleThreadedExecutor()
    executor.add_node(pt_ctrl)
    try:
        while pt_ctrl.validation_agent_thread.is_alive():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        raise
    finally:
        executor.remove_node(pt_ctrl)
        pt_ctrl.on_shutdown()
        pt_ctrl.destroy_node()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)

    total = len(INSPECTION_GOALS)
    try:
        for i, goal in enumerate(INSPECTION_GOALS):
            label = goal.get('label', f"({goal['x']:.2f}, {goal['y']:.2f})")
            print(f'\n[{i + 1}/{total}] Starting inspection at: {label}')
            _run_inspection_at_goal(goal)
            print(f'[{i + 1}/{total}] Inspection complete at: {label}')
            # _wait_for_continue(f'Phase 7: LLM inspection at {label}')

        print('\nAll waypoints inspected.')
    except KeyboardInterrupt:
        print('\nPipeline interrupted by user.')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
