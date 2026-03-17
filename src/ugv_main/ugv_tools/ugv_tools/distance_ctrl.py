#!/usr/bin/env python
# encoding: utf-8

import json
import math
import threading
import time

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ugv_interface.action import Behavior

# Search range boundaries
IDEAL_DISTANCE = 1.2   # m — target distance from wall
MIN_DISTANCE   = 0.6   # m — inner boundary (too close)
MAX_DISTANCE   = 2.5   # m — outer boundary (too far)

# Accessibility thresholds
SIDE_CLEARANCE  = 1.0  # m — minimum lateral clearance on each side
FRONT_HALF_DEG  = 10.0 # ° — half-width for front distance measurement
SIDE_HALF_DEG   = 10.0 # ° — half-width for left/right clearance sectors

# Exploration
PROBE_STEP      = 0.1   # m — spacing between probe points during sweep
POSITION_TOL    = 0.05  # m — acceptable error when moving to final position
SAFETY_MARGIN   = 0.3  # m — buffer kept from any obstacle when moving

# Midpoint of search range: rover goes to the nearer endpoint first
_SWEEP_MIDPOINT = (MIN_DISTANCE + MAX_DISTANCE) / 2.0  # 1.55 m


class DistanceCtrl(Node):

    def __init__(self):
        super().__init__('distance_ctrl')
        self._scan = None
        self._scan_event = threading.Event()
        self.create_subscription(LaserScan, 'scan', self._scan_cb, 10)
        self._behavior_client = ActionClient(self, Behavior, 'behavior')

    # ------------------------------------------------------------------
    # Scan callbacks & helpers
    # ------------------------------------------------------------------

    def _scan_cb(self, msg):
        self._scan = msg
        self._scan_event.set()

    def _fresh_scan(self):
        """Wait for the next LiDAR scan and return it."""
        self._scan_event.clear()
        self._scan_event.wait()
        return self._scan

    def _sector_median(self, scan, center_deg: float, half_deg: float):
        """
        Median range of valid returns in the angular sector
        [center_deg ± half_deg].  Returns None if the sector is empty.
        """
        center_rad = math.radians(center_deg)
        half_rad   = math.radians(half_deg)
        dists = []
        for i, r in enumerate(scan.ranges):
            if not math.isfinite(r) or not (scan.range_min <= r <= scan.range_max):
                continue
            angle = scan.angle_min + i * scan.angle_increment
            # Normalize difference to [-π, π] to handle wrap-around
            diff = (angle - center_rad + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) <= half_rad:
                dists.append(r)
        return float(np.median(dists)) if dists else None

    def front_distance(self):
        """Median distance straight ahead (0° ± FRONT_HALF_DEG)."""
        return self._sector_median(self._fresh_scan(), 0.0, FRONT_HALF_DEG)

    def rear_distance(self):
        """Median distance directly behind (180° ± FRONT_HALF_DEG)."""
        return self._sector_median(self._fresh_scan(), 180.0, FRONT_HALF_DEG)

    def is_accessible(self):
        """
        Read one fresh scan and check both lateral clearances.

        Accessible = left clearance (90° ± 10°) ≥ 1 m
                   AND right clearance (−90° ± 10°) ≥ 1 m
        Returns True/False; logs the measured clearances.
        """
        scan = self._fresh_scan()
        lc = self._sector_median(scan,  90.0, SIDE_HALF_DEG)
        rc = self._sector_median(scan, -90.0, SIDE_HALF_DEG)

        if lc is None or rc is None:
            self.get_logger().warn('  Side sector returned no valid readings — marking inaccessible.')
            return False

        self.get_logger().info(f'    left: {lc:.2f} m   right: {rc:.2f} m')
        return lc >= SIDE_CLEARANCE and rc >= SIDE_CLEARANCE

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def _send_move(self, toward_wall_m: float) -> float:
        """
        Move toward the wall by *toward_wall_m* metres (negative = away).

        Before moving, reads the relevant clearance sector and clamps the
        requested distance to (clearance - SAFETY_MARGIN) so the rover
        never drives into an obstacle.

        Returns the actual distance moved (≥ 0).  Returns 0.0 if the path
        is fully blocked or the requested distance is negligible.
        """
        if abs(toward_wall_m) < 1e-3:
            return 0.0

        requested = abs(toward_wall_m)

        if toward_wall_m > 0:
            # Moving toward wall — check front clearance
            clearance = self.front_distance()
            cmd_type, label = 'drive_on_heading', 'toward wall'
        else:
            # Moving away from wall — check rear clearance
            clearance = self.rear_distance()
            cmd_type, label = 'back_up', 'away from wall'

        if clearance is None:
            self.get_logger().warn(
                f'No {label.split()[0]} distance reading — aborting move.')
            return 0.0

        available = clearance - SAFETY_MARGIN
        if available <= 1e-3:
            self.get_logger().warn(
                f'Path {label} fully blocked (clearance {clearance:.2f} m, '
                f'margin {SAFETY_MARGIN} m) — skipping move.')
            return 0.0

        actual = min(requested, available)
        if actual < requested - 1e-3:
            self.get_logger().warn(
                f'Clamping move {label}: requested {requested:.2f} m, '
                f'available {available:.2f} m (clearance {clearance:.2f} m).')

        goal = Behavior.Goal()
        goal.command = json.dumps([{'type': cmd_type, 'data': actual}])
        self.get_logger().info(f'Moving {label} by {actual:.2f} m')

        done = threading.Event()

        def _result_cb(_):
            done.set()

        def _goal_cb(future):
            future.result().get_result_async().add_done_callback(_result_cb)

        self._behavior_client.send_goal_async(goal).add_done_callback(_goal_cb)
        done.wait()
        time.sleep(1)
        return actual

    # ------------------------------------------------------------------
    # Main exploration controller
    # ------------------------------------------------------------------

    def find_accessible_distance(self):
        """
        Explore [MIN_DISTANCE, MAX_DISTANCE] in a single continuous pass
        (zero overlap) to find every accessible position, then move to
        the one closest to IDEAL_DISTANCE.

        Cases handled
        -------------
        1. d > MAX_DISTANCE (2.5 m) : approach wall until d == 2.5 m
        2. MAX_DISTANCE ≥ d > IDEAL  : already in range, above ideal
        3. IDEAL ≥ d > MIN_DISTANCE  : already in range, below ideal
        4. d < MIN_DISTANCE (0.6 m)  : back away until d == 0.6 m

        For cases 2–3 the sweep direction is chosen so the rover reaches
        the nearer range endpoint first, giving the shortest total travel
        with no revisited positions.

        Returns the selected wall distance (float), or None if no
        accessible position was found.
        """
        # ── 1. Read initial distance ───────────────────────────────────
        d = self.front_distance()
        if d is None:
            self.get_logger().error('Cannot read front distance — aborting.')
            return None
        self.get_logger().info(f'Initial wall distance: {d:.2f} m')

        # ── 2. Enter the search range [MIN, MAX] ───────────────────────
        if d > MAX_DISTANCE:
            # Case 1 — too far: drive toward wall to MAX_DISTANCE
            self.get_logger().info(
                f'[Case 1] d={d:.2f} m > {MAX_DISTANCE} m — approaching wall.')
            self._send_move(d - MAX_DISTANCE)
            d = MAX_DISTANCE

        elif d < MIN_DISTANCE:
            # Case 4 — too close: back away to MIN_DISTANCE
            self.get_logger().info(
                f'[Case 4] d={d:.2f} m < {MIN_DISTANCE} m — backing away.')
            self._send_move(-(MIN_DISTANCE - d))
            d = MIN_DISTANCE

        elif d > IDEAL_DISTANCE:
            self.get_logger().info(
                f'[Case 2] d={d:.2f} m in ({IDEAL_DISTANCE}, {MAX_DISTANCE}] — in range, above ideal.')
        else:
            self.get_logger().info(
                f'[Case 3] d={d:.2f} m in [{MIN_DISTANCE}, {IDEAL_DISTANCE}] — in range, below ideal.')

        # d is now guaranteed ∈ [MIN_DISTANCE, MAX_DISTANCE]

        # ── 3. Choose sweep direction for minimum-travel, zero-overlap ─
        #
        #   Rover goes to the nearer endpoint first, then sweeps in one
        #   continuous pass to the far endpoint.  The midpoint (1.55 m)
        #   is the crossover: below it we go to 0.6 first, above it to 2.5.
        #
        #   Travel cost  = |d - near_end| + (MAX - MIN)
        #   Choosing the nearer end minimises that first segment.
        #
        if d <= _SWEEP_MIDPOINT:
            sweep_start, sweep_end = MIN_DISTANCE, MAX_DISTANCE
            sweep_dir = +1   # moving away from wall (+distance) each step
        else:
            sweep_start, sweep_end = MAX_DISTANCE, MIN_DISTANCE
            sweep_dir = -1   # moving toward wall (−distance) each step

        self.get_logger().info(
            f'Sweep: {sweep_start:.1f} m → {sweep_end:.1f} m '
            f'(step {PROBE_STEP} m, {sweep_dir:+d} direction)')

        # Move from current d to sweep start
        delta_to_start = d - sweep_start   # >0 → toward wall, <0 → away
        actual = self._send_move(delta_to_start)
        # Derive actual position: movement reduces/increases distance from wall
        current_d = d - math.copysign(actual, delta_to_start)

        # ── 4. Single-pass sweep — probe every PROBE_STEP ─────────────
        n_steps = int(round(abs(sweep_end - sweep_start) / PROBE_STEP))
        accessible_positions = []   # wall distances where rover is accessible

        for step_idx in range(n_steps + 1):
            # Clamp to avoid floating-point overshoot at boundaries
            probe_d = max(MIN_DISTANCE, min(MAX_DISTANCE, current_d))

            self.get_logger().info(f'Probing {probe_d:.2f} m ...')
            if self.is_accessible():
                accessible_positions.append(probe_d)
                self.get_logger().info(f'  -> ACCESSIBLE at {probe_d:.2f} m')
            else:
                self.get_logger().info(f'  -> blocked at {probe_d:.2f} m')

            # Advance to next probe point (skip after last step)
            if step_idx < n_steps:
                # sweep_dir +1 → distance increases → move away from wall → _send_move < 0
                actual = self._send_move(-sweep_dir * PROBE_STEP)
                if actual < PROBE_STEP - 1e-3:
                    # Obstacle blocked the step — record where we stopped and abort sweep
                    current_d += sweep_dir * actual
                    self.get_logger().warn(
                        f'Sweep blocked at {current_d:.2f} m after {step_idx + 1} probe(s) '
                        f'— stopping early.')
                    break
                current_d += sweep_dir * actual

        # ── 5. Select the accessible position closest to IDEAL_DISTANCE ─
        if not accessible_positions:
            self.get_logger().warn(
                f'No accessible position found in [{MIN_DISTANCE}, {MAX_DISTANCE}] m. '
                'Inspection cannot proceed.')
            return None

        best_d = min(accessible_positions, key=lambda x: abs(x - IDEAL_DISTANCE))
        delta  = best_d - IDEAL_DISTANCE
        self.get_logger().info(
            f'Best accessible distance: {best_d:.2f} m '
            f'(Δ from ideal: {delta:+.2f} m)  '
            f'[{len(accessible_positions)} accessible positions found]')

        # ── 6. Move to the chosen position ───────────────────────────
        # Re-read actual position before final move to absorb any clamping drift
        current_d = self.front_distance() or current_d
        move_needed = current_d - best_d
        if abs(move_needed) > POSITION_TOL:
            self._send_move(move_needed)

        final = self.front_distance()
        self.get_logger().info(
            f'Positioned at {final:.2f} m from wall '
            f'(target {best_d:.2f} m).')
        return best_d


def main(args=None):
    rclpy.init(args=args)
    node = DistanceCtrl()
    result = [None]
    done_event = threading.Event()

    def _run():
        time.sleep(1.0)  # Allow subscriptions / action server to connect
        result[0] = node.find_accessible_distance()
        done_event.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
