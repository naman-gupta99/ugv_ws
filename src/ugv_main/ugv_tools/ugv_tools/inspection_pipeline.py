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
import csv
import json
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import requests

CONTINUE_SENTINEL = '/tmp/ugv_continue'

import rclpy
from rclpy.executors import SingleThreadedExecutor

from .align_ctrl import AlignCtrl
from .distance_ctrl import DistanceCtrl, audit_state_instance
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

METRICS_CSV_PATH = os.environ.get(
    'UGV_METRICS_CSV',
    '/home/ws/ugv_ws/inspection_metrics.csv',
)

METRICS_FIELDS = [
    'entry_time',
    'greedy',
    'llm_used',
    'hint',
    'capture_folders',
    'inspection_duration_sec',
    'goal_1_phase_1_duration_sec',
    'goal_1_phase_2_duration_sec',
    'goal_1_phase_3_duration_sec',
    'goal_1_phase_4_duration_sec',
    'goal_1_phase_5_duration_sec',
    'goal_1_phase_6_duration_sec',
    'goal_1_phase_7_duration_sec',
    'goal_1_pictures_taken',
    'goal_2_phase_1_duration_sec',
    'goal_2_phase_2_duration_sec',
    'goal_2_phase_3_duration_sec',
    'goal_2_phase_4_duration_sec',
    'goal_2_phase_5_duration_sec',
    'goal_2_phase_6_duration_sec',
    'goal_2_phase_7_duration_sec',
    'goal_2_pictures_taken',
    'goal_1_agent_metrics',
    'goal_2_agent_metrics',
    'goal_1_langsmith_tokens',
    'goal_1_langsmith_cost',
    'goal_2_langsmith_tokens',
    'goal_2_langsmith_cost',
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_cell(value) -> str:
    if value in (None, ''):
        return ''
    return json.dumps(value, sort_keys=True)


def _collect_langsmith_usage(start_time: datetime, end_time: datetime, thread_id: str = None) -> dict:
    """Aggregate token and cost fields from LangSmith LLM runs in this window."""
    usage = {
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0,
        'prompt_cost': 0.0,
        'completion_cost': 0.0,
        'total_cost': 0.0,
        'llm_runs': 0,
        'error': '',
    }
    timeout_s = float(os.getenv('UGV_LANGSMITH_METRICS_TIMEOUT', '60'))
    poll_s = float(os.getenv('UGV_LANGSMITH_METRICS_POLL_SEC', '3'))

    if os.getenv('LANGSMITH_TRACING', 'false').lower() != 'true':
        usage['error'] = 'LANGSMITH_TRACING is not enabled'
        return usage

    project_name = os.getenv('LANGSMITH_PROJECT')
    if not project_name:
        usage['error'] = 'LANGSMITH_PROJECT is not set'
        return usage

    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers
        from langsmith import Client

        wait_for_all_tracers()
        client = Client()
        deadline = time.monotonic() + timeout_s

        while True:
            current_usage = {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'prompt_cost': 0.0,
                'completion_cost': 0.0,
                'total_cost': 0.0,
                'llm_runs': 0,
                'error': '',
            }

            for run in client.list_runs(
                project_name=project_name,
                run_type='llm',
                start_time=start_time,
                select=[
                    'start_time',
                    'extra',
                    'prompt_tokens',
                    'completion_tokens',
                    'total_tokens',
                    'prompt_cost',
                    'completion_cost',
                    'total_cost',
                ],
                limit=100,
            ):
                run_start = run.start_time
                if run_start is None:
                    continue
                if run_start.tzinfo is None:
                    run_start = run_start.replace(tzinfo=timezone.utc)
                if run_start > end_time:
                    continue

                metadata = (run.extra or {}).get('metadata', {})
                if thread_id and metadata.get('ugv_agent_thread_id') != thread_id:
                    continue

                current_usage['llm_runs'] += 1
                for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                    value = getattr(run, field, None)
                    if value is not None:
                        current_usage[field] += int(value)
                for field in ('prompt_cost', 'completion_cost', 'total_cost'):
                    value = getattr(run, field, None)
                    if value is not None:
                        current_usage[field] += float(value)

            usage = current_usage
            if usage['llm_runs'] and (usage['total_tokens'] or usage['total_cost']):
                break
            if time.monotonic() >= deadline:
                if not usage['llm_runs']:
                    usage['error'] = 'No LangSmith LLM runs found for phase 7 window'
                else:
                    usage['error'] = 'LangSmith LLM runs found but token/cost fields were not populated before timeout'
                break
            time.sleep(poll_s)
    except Exception as exc:
        usage['error'] = str(exc)

    usage['prompt_cost'] = round(usage['prompt_cost'], 8)
    usage['completion_cost'] = round(usage['completion_cost'], 8)
    usage['total_cost'] = round(usage['total_cost'], 8)
    return usage


def _write_metrics_row(row: dict) -> None:
    path = Path(METRICS_CSV_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    if path.exists():
        with path.open('r', newline='', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            existing_fields = reader.fieldnames or []
            if not existing_fields:
                write_header = True
            if existing_fields and existing_fields != METRICS_FIELDS:
                rows = list(reader)
                for field in METRICS_FIELDS:
                    if field not in existing_fields:
                        for existing_row in rows:
                            existing_row[field] = ''
                with path.open('w', newline='', encoding='utf-8') as rewrite_file:
                    writer = csv.DictWriter(rewrite_file, fieldnames=METRICS_FIELDS)
                    writer.writeheader()
                    for existing_row in rows:
                        writer.writerow({field: existing_row.get(field, '') for field in METRICS_FIELDS})

    with path.open('a', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=METRICS_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, '') for field in METRICS_FIELDS})


def _run_phase(node, work_fn, startup_delay: float = 0.5):
    """
    Spin *node* while *work_fn()* runs in a background thread.

    Destroys the node when work is done (or if an exception is raised),
    and returns work_fn()'s return value.
    Re-raises KeyboardInterrupt so the top-level loop can shut down cleanly.
    """
    done = threading.Event()
    outcome = {'result': None, 'exc': None}

    def _worker():
        time.sleep(startup_delay)
        try:
            outcome['result'] = work_fn()
        except Exception as exc:
            outcome['exc'] = exc
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

    if outcome['exc'] is not None:
        raise outcome['exc']

    return outcome['result']


def _record_phase_duration(goal_metrics: dict, phase_number: int, start_time: float) -> None:
    goal_metrics['phase_durations_sec'][f'phase_{phase_number}'] = round(
        time.monotonic() - start_time,
        3,
    )


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


def _run_inspection_at_goal(goal: dict) -> dict:
    """Run the full six-phase inspection pipeline at one waypoint."""
    label = goal.get('label', f"({goal['x']:.2f}, {goal['y']:.2f})")
    goal_metrics = {
        'label': label,
        'success': False,
        'failure_reason': '',
        'capture_folder': '',
        'phase_durations_sec': {},
        'agent_metrics': {},
        'langsmith_usage': {},
        'llm_used': '',
        'pictures_taken': '',
    }
    
    # Reset shared mission state so every controller invocation starts fresh.
    audit_state_instance.reset()

    # ------------------------------------------------------------------
    # Phase 1: Navigate to waypoint
    # ------------------------------------------------------------------
    print(f'  [Phase 1] Navigating to {label}')
    phase_start = time.monotonic()
    nav_node = NavCtrl()
    try:
        if not _run_phase(
            nav_node,
            lambda: nav_node.navigate_to(goal['x'], goal['y'], goal['qz'], goal['qw']),
            startup_delay=1.0,
        ):
            goal_metrics['failure_reason'] = 'Navigation goal failed.'
            return goal_metrics
    finally:
        _record_phase_duration(goal_metrics, 1, phase_start)
    time.sleep(2.0)

    # ------------------------------------------------------------------
    # Phase 2: Align perpendicular to wall
    # ------------------------------------------------------------------
    print(f'  [Phase 2] Aligning perpendicular to wall')
    phase_start = time.monotonic()
    align_node = AlignCtrl('perpendicular')
    try:
        if not _run_phase(align_node, align_node.align, startup_delay=1.0):
            return goal_metrics
    finally:
        _record_phase_duration(goal_metrics, 2, phase_start)
    time.sleep(5.0)

    # ------------------------------------------------------------------
    # Phase 3: Centre on highest-confidence detection
    #   - Captures a camera image and posts it to the detection API.
    #   - If the best detection is off-centre, the rover shifts laterally
    #     until the target is within CENTERING_MARGIN_PX of image centre.
    # ------------------------------------------------------------------
    print(f'  [Phase 3] Centering on wall detection')
    phase_start = time.monotonic()
    centering_node = WallCenteringCtrl()
    try:
        if not _run_phase(centering_node, centering_node.run, startup_delay=1.0):
            return goal_metrics
    finally:
        _record_phase_duration(goal_metrics, 3, phase_start)
    time.sleep(2.0)
    
    # ------------------------------------------------------------------
    # Phase 4: Align perpendicular to wall
    # ------------------------------------------------------------------
    print(f'  [Phase 4] Aligning perpendicular to wall')
    phase_start = time.monotonic()
    align_node = AlignCtrl('perpendicular')
    try:
        if not _run_phase(align_node, align_node.align, startup_delay=1.0):
            return goal_metrics
    finally:
        _record_phase_duration(goal_metrics, 4, phase_start)
    time.sleep(5.0)

    # ------------------------------------------------------------------
    # Phase 5: Find ideal inspection distance from wall
    # ------------------------------------------------------------------
    print(f'  [Phase 5] Finding ideal inspection distance')
    phase_start = time.monotonic()
    dist_node = DistanceCtrl()

    def _find_distance():
        best = dist_node.find_accessible_distance()
        if best is None:
            dist_node.get_logger().error('No accessible distance found — continuing.')
        return best

    try:
        if _run_phase(dist_node, _find_distance, startup_delay=0.5) is None:
            return goal_metrics
    finally:
        _record_phase_duration(goal_metrics, 5, phase_start)
    time.sleep(5.0)

    # ------------------------------------------------------------------
    # Phase 6: Align parallel to wall
    # ------------------------------------------------------------------
    print(f'  [Phase 6] Aligning parallel to wall')
    phase_start = time.monotonic()
    align_node = AlignCtrl('parallel')
    try:
        if not _run_phase(align_node, align_node.align, startup_delay=1.0):
            return goal_metrics
    finally:
        _record_phase_duration(goal_metrics, 6, phase_start)
    time.sleep(10.0)

    # ------------------------------------------------------------------
    # Phase 7: LLM pan-tilt inspection (runs until agent finishes)
    # ------------------------------------------------------------------
    print(f'  [Phase 7] Running LLM inspection agent')
    phase_7_start_monotonic = time.monotonic()
    phase_7_start_dt = _utc_now()
    pt_ctrl = LlmPtCtrl('llm_pt_ctrl')
    goal_metrics['capture_folder'] = os.path.basename(pt_ctrl._run_dir)
    executor = SingleThreadedExecutor()
    executor.add_node(pt_ctrl)
    try:
        while pt_ctrl.validation_agent_thread.is_alive():
            if pt_ctrl.motion_failed_event.is_set():
                pt_ctrl.get_logger().warn(
                    f'LLM inspection motion failed: {pt_ctrl.motion_failed_reason}. Aborting current goal.'
                )
                return goal_metrics
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        raise
    finally:
        phase_7_end_dt = _utc_now()
        _record_phase_duration(goal_metrics, 7, phase_7_start_monotonic)
        goal_metrics['agent_metrics'] = pt_ctrl.validation_agent_metrics or {}
        goal_metrics['llm_used'] = pt_ctrl.validation_agent_model_name or ''
        goal_metrics['pictures_taken'] = pt_ctrl.pictures_taken
        goal_metrics['langsmith_usage'] = _collect_langsmith_usage(
            phase_7_start_dt,
            phase_7_end_dt,
            thread_id=pt_ctrl.validation_agent_thread_id,
        )
        executor.remove_node(pt_ctrl)
        pt_ctrl.on_shutdown()
        pt_ctrl.destroy_node()

    if pt_ctrl.motion_failed_event.is_set():
        return goal_metrics

    goal_metrics['success'] = True
    return goal_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    debugpy_port = os.environ.get('UGV_DEBUGPY_PORT')
    if debugpy_port:
        import debugpy

        debugpy.listen(('0.0.0.0', int(debugpy_port)))

        # Spawn a background waiter so we can keep logging while we block
        # waiting for the debugger to attach. This prints the waiting
        # message every second until the client connects.
        attached_event = threading.Event()

        def _wait_for_debugger():
            try:
                debugpy.wait_for_client()
            finally:
                attached_event.set()

        threading.Thread(target=_wait_for_debugger, daemon=True).start()

        while not attached_event.wait(timeout=1.0):
            print(f'Waiting for debugger attach on port {debugpy_port}...')

        print(f'Debugger attached on port {debugpy_port}.')

    rclpy.init(args=args)

    total = len(INSPECTION_GOALS)
    inspection_start = time.monotonic()
    goal_results = []
    try:
        for i, goal in enumerate(INSPECTION_GOALS):
            label = goal.get('label', f"({goal['x']:.2f}, {goal['y']:.2f})")
            print(f'\n[{i + 1}/{total}] Starting inspection at: {label}')
            goal_result = _run_inspection_at_goal(goal)
            goal_results.append(goal_result)
            if goal_result.get('success'):
                print(f'[{i + 1}/{total}] Inspection complete at: {label}')
            else:
                reason = goal_result.get('failure_reason') or 'goal failed'
                print(f'[{i + 1}/{total}] Inspection failed at: {label} — {reason}. Continuing to next waypoint.')
            # _wait_for_continue(f'Phase 7: LLM inspection at {label}')

        successful_goals = sum(1 for result in goal_results if result.get('success'))
        failed_goals = len(goal_results) - successful_goals
        if len(goal_results) == total and failed_goals == 0:
            print('\nAll waypoints inspected.')
        elif len(goal_results) == total:
            print(f'\nInspection pipeline complete: {successful_goals}/{total} waypoint(s) succeeded, {failed_goals} failed.')
        # _wait_for_continue(f'Complete: All waypoints inspected')
        
    except KeyboardInterrupt:
        print('\nPipeline interrupted by user.')
    finally:
        inspection_duration_sec = round(time.monotonic() - inspection_start, 3)
        goal_1 = goal_results[0] if len(goal_results) > 0 else {}
        goal_2 = goal_results[1] if len(goal_results) > 1 else {}
        capture_folders = [
            result.get('capture_folder', '')
            for result in goal_results
            if result.get('capture_folder')
        ]
        llm_used = next(
            (result.get('llm_used') for result in goal_results if result.get('llm_used')),
            os.environ.get('UGV_AGENT_MODEL', 'gemini-2.5-pro'),
        )
        hint_used = next(
            (result.get('agent_metrics', {}).get('hint_used', '') for result in goal_results if result.get('agent_metrics', {}).get('hint_used')),
            os.environ.get('UGV_AGENT_HINT', ''),
        )
        goal_1_phases = goal_1.get('phase_durations_sec') or {}
        goal_2_phases = goal_2.get('phase_durations_sec') or {}

        metrics_row = {
            'entry_time': _utc_now().isoformat(),
            'greedy': 'Yes' if os.getenv('UGV_GREEDY', 'false').lower() in ('1', 'true', 'yes') else 'No',
            'llm_used': llm_used,
            'hint': hint_used,
            'capture_folders': ';'.join(capture_folders),
            'inspection_duration_sec': inspection_duration_sec,
            'goal_1_phase_1_duration_sec': goal_1_phases.get('phase_1', ''),
            'goal_1_phase_2_duration_sec': goal_1_phases.get('phase_2', ''),
            'goal_1_phase_3_duration_sec': goal_1_phases.get('phase_3', ''),
            'goal_1_phase_4_duration_sec': goal_1_phases.get('phase_4', ''),
            'goal_1_phase_5_duration_sec': goal_1_phases.get('phase_5', ''),
            'goal_1_phase_6_duration_sec': goal_1_phases.get('phase_6', ''),
            'goal_1_phase_7_duration_sec': goal_1_phases.get('phase_7', ''),
            'goal_1_pictures_taken': goal_1.get('pictures_taken', ''),
            'goal_2_phase_1_duration_sec': goal_2_phases.get('phase_1', ''),
            'goal_2_phase_2_duration_sec': goal_2_phases.get('phase_2', ''),
            'goal_2_phase_3_duration_sec': goal_2_phases.get('phase_3', ''),
            'goal_2_phase_4_duration_sec': goal_2_phases.get('phase_4', ''),
            'goal_2_phase_5_duration_sec': goal_2_phases.get('phase_5', ''),
            'goal_2_phase_6_duration_sec': goal_2_phases.get('phase_6', ''),
            'goal_2_phase_7_duration_sec': goal_2_phases.get('phase_7', ''),
            'goal_2_pictures_taken': goal_2.get('pictures_taken', ''),
            'goal_1_agent_metrics': _json_cell(goal_1.get('agent_metrics')),
            'goal_2_agent_metrics': _json_cell(goal_2.get('agent_metrics')),
            'goal_1_langsmith_tokens': (goal_1.get('langsmith_usage') or {}).get('total_tokens', ''),
            'goal_1_langsmith_cost': (goal_1.get('langsmith_usage') or {}).get('total_cost', ''),
            'goal_2_langsmith_tokens': (goal_2.get('langsmith_usage') or {}).get('total_tokens', ''),
            'goal_2_langsmith_cost': (goal_2.get('langsmith_usage') or {}).get('total_cost', ''),
        }
        try:
            _write_metrics_row(metrics_row)
            print(f'Metrics written to {METRICS_CSV_PATH}')
        finally:
            rclpy.shutdown()


if __name__ == '__main__':
    main()
