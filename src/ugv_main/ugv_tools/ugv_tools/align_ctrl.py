#!/usr/bin/env python
# encoding: utf-8

import argparse
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

from .lidar_scan_utils import process_scan_for_rover

# Angular half-width of the LiDAR sectors used for wall detection (degrees)
FRONT_HALF_ANGLE_DEG = 10.0
RIGHT_HALF_ANGLE_DEG = 10.0

# Iterative refinement: stop correcting when the residual error is below this
ALIGN_THRESHOLD_DEG = 0.5
# Maximum correction iterations before giving up
MAX_ALIGN_ITER = 10
# Number of consecutive scans to average for a stable wall-angle measurement
NUM_SCANS_TO_AVG = 3
# Calibrated time for a 90° spin — used to derive proportional waits
SPIN_90_S = 12.0


class AlignCtrl(Node):

    def __init__(self, mode: str):
        """
        Args:
            mode: 'parallel'      – align body axis parallel to wall (wall on right)
                  'perpendicular' – face directly into the wall
        """
        super().__init__('align_ctrl')
        if mode not in ('parallel', 'perpendicular'):
            raise ValueError(f"mode must be 'parallel' or 'perpendicular', got '{mode}'")
        self._mode = mode
        self._scan = None
        self._scan_event = threading.Event()
        self.create_subscription(LaserScan, 'scan', self._scan_cb, 10)
        self._behavior_client = ActionClient(self, Behavior, 'behavior')

    def _scan_cb(self, msg):
        self._scan = process_scan_for_rover(msg)
        self._scan_event.set()

    def align(self):
        """Dispatch to the selected alignment mode."""
        if self._mode == 'parallel':
            return self._align_parallel()
        else:
            return self._align_perpendicular()

    def _align_parallel(self):
        """Spin the rover to be parallel to the wall.

        The first correction uses the wall directly in front to get onto the
        wall side. After the rover has turned, later iterations use wall points
        on the right side to refine the heading.
        """
        self.get_logger().info('Waiting for initial LiDAR scan...')
        self._scan_event.wait()

        for iteration in range(MAX_ALIGN_ITER):
            if iteration == 0:
                wall_angle = self._measure_wall_angle_averaged(0.0, FRONT_HALF_ANGLE_DEG)
                sector_label = 'front'
            else:
                wall_angle = self._measure_wall_angle_averaged(-90.0, RIGHT_HALF_ANGLE_DEG)
                sector_label = 'right'

            if wall_angle is None:
                self.get_logger().error(f'Not enough wall points in {sector_label} sector — aborting.')
                return False

            if iteration == 0:
                # After rotating by `rot`, the wall centre (currently at roughly +x) ends up
                # at (d·cos(rot), −d·sin(rot)). For wall on the right: −d·sin(rot) < 0,
                # i.e. sin(rot) > 0, so 0 < rot < π. The wall line has two antipodal
                # directions; pick the one that satisfies this constraint.
                rot = wall_angle
                if math.sin(rot) <= 0:
                    rot += math.pi
                rot = (rot + math.pi) % (2 * math.pi) - math.pi
            else:
                # Once the wall is on the right, align the rover with the wall axis.
                # The wall direction is undirected, so choose the smallest equivalent
                # correction in [-90°, +90°].
                rot = (wall_angle + math.pi / 2) % math.pi - math.pi / 2
            rot_deg = math.degrees(rot)

            self.get_logger().info(
                f'[iter {iteration + 1}/{MAX_ALIGN_ITER}] '
                f'Wall angle: {math.degrees(wall_angle):.2f}°  →  correction: {rot_deg:.2f}° '
                f'(parallel, sampled from {sector_label} sector)'
            )

            if abs(rot_deg) < ALIGN_THRESHOLD_DEG:
                self.get_logger().info(
                    f'Residual error {abs(rot_deg):.2f}° < {ALIGN_THRESHOLD_DEG}° threshold — alignment complete.'
                )
                return True

            spin_wait_s = max(3.0, abs(rot_deg) / 90.0 * SPIN_90_S)
            self.get_logger().info(f'Spinning {rot_deg:.2f}° — waiting {spin_wait_s:.1f} s for physical completion.')
            if not self._send_spin(rot_deg):
                return False
            time.sleep(spin_wait_s)

        self.get_logger().warn(
            f'Reached max iterations ({MAX_ALIGN_ITER}) — alignment may have residual error.'
        )
        return True

    def _align_perpendicular(self):
        """Spin the rover to face directly into the wall in front."""
        self.get_logger().info('Waiting for initial LiDAR scan...')
        self._scan_event.wait()

        for iteration in range(MAX_ALIGN_ITER):
            wall_angle = self._measure_wall_angle_averaged()
            if wall_angle is None:
                self.get_logger().error('Not enough wall points in front sector — aborting.')
                return False

            # Rotate so the wall runs at ±90° in the new robot frame (left-right),
            # making the robot face the wall.  Two candidates; pick the smaller rotation.
            rot_a = (wall_angle - math.pi / 2 + math.pi) % (2 * math.pi) - math.pi
            rot_b = (wall_angle + math.pi / 2 + math.pi) % (2 * math.pi) - math.pi
            rot = rot_a if abs(rot_a) <= abs(rot_b) else rot_b
            rot_deg = math.degrees(rot)

            self.get_logger().info(
                f'[iter {iteration + 1}/{MAX_ALIGN_ITER}] '
                f'Wall angle: {math.degrees(wall_angle):.2f}°  →  correction: {rot_deg:.2f}° (perpendicular)'
            )

            if abs(rot_deg) < ALIGN_THRESHOLD_DEG:
                self.get_logger().info(
                    f'Residual error {abs(rot_deg):.2f}° < {ALIGN_THRESHOLD_DEG}° threshold — alignment complete.'
                )
                return True

            spin_wait_s = max(3.0, abs(rot_deg) / 90.0 * SPIN_90_S)
            self.get_logger().info(f'Spinning {rot_deg:.2f}° — waiting {spin_wait_s:.1f} s for physical completion.')
            if not self._send_spin(rot_deg):
                return False
            time.sleep(spin_wait_s)

        self.get_logger().warn(
            f'Reached max iterations ({MAX_ALIGN_ITER}) — alignment may have residual error.'
        )
        return True

    def _get_fresh_scan(self):
        """Block until a new scan arrives (clears any cached scan first)."""
        self._scan_event.clear()
        self._scan_event.wait()
        return self._scan

    def _measure_wall_angle_averaged(self, center_deg: float = 0.0, half_deg: float = FRONT_HALF_ANGLE_DEG):
        """
        Collect NUM_SCANS_TO_AVG fresh scans and return a circular-mean wall
        angle for noise reduction from a specific LiDAR sector. Returns None
        if there are never enough points.
        """
        angles = []
        for _ in range(NUM_SCANS_TO_AVG):
            scan = self._get_fresh_scan()
            points = self._sector_wall_points(scan, center_deg, half_deg)
            if len(points) >= 5:
                angles.append(self._wall_angle(points))
        if not angles:
            return None
        # Circular mean — correct across the ±π wrap.
        sin_mean = np.mean(np.sin(angles))
        cos_mean = np.mean(np.cos(angles))
        return math.atan2(sin_mean, cos_mean)

    def _wall_angle(self, points):
        """Return the angle (radians) of the dominant wall direction via SVD."""
        pts = np.array(points)
        centered = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        wall_dir = vt[0]
        return math.atan2(wall_dir[1], wall_dir[0])

    def _sector_wall_points(self, scan, center_deg: float, half_deg: float):
        """Return (x, y) Cartesian points from a LiDAR sector centered at center_deg."""
        center_rad = math.radians(center_deg)
        half_rad = math.radians(half_deg)
        points = []
        for i, r in enumerate(scan.ranges):
            if not math.isfinite(r) or not (scan.range_min <= r <= scan.range_max):
                continue
            angle = scan.angle_min + i * scan.angle_increment
            delta = (angle - center_rad + math.pi) % (2 * math.pi) - math.pi
            if abs(delta) <= half_rad:
                points.append((r * math.cos(angle), r * math.sin(angle)))
        return points

    def _send_spin(self, angle_deg):
        """Send a spin goal to behavior_ctrl and block until it completes."""
        goal = Behavior.Goal()
        goal.command = json.dumps([{'type': 'spin', 'data': angle_deg}])

        done_event = threading.Event()
        outcome = {'success': False, 'message': ''}

        def _result_cb(_):
            result = _.result()
            behavior_result = result.result
            outcome['success'] = bool(getattr(behavior_result, 'result', False))
            outcome['message'] = getattr(behavior_result, 'message', '')
            done_event.set()

        def _goal_cb(future):
            handle = future.result()
            if not handle.accepted:
                outcome['message'] = 'Spin goal was rejected.'
                done_event.set()
                return
            handle.get_result_async().add_done_callback(_result_cb)

        self._behavior_client.send_goal_async(goal).add_done_callback(_goal_cb)
        done_event.wait()
        if not outcome['success']:
            self.get_logger().warn(f'Spin behavior failed: {outcome["message"]}')
        return outcome['success']


def main(args=None):
    # parse_known_args lets ROS2 remapping args pass through untouched.
    parser = argparse.ArgumentParser(description='Align rover to wall')
    parser.add_argument(
        'mode',
        choices=['parallel', 'perpendicular'],
        help="'parallel': body axis alongside wall (wall on right); "
             "'perpendicular': face directly into wall",
    )
    known, _ = parser.parse_known_args(args)

    rclpy.init(args=args)
    node = AlignCtrl(known.mode)

    def _run():
        time.sleep(1.0)  # Allow subscriptions / action server to connect
        node.align()

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
