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

# Angular half-width of the front sector used for wall detection (degrees)
FRONT_HALF_ANGLE_DEG = 10.0


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
        self._scan = msg
        self._scan_event.set()

    def align(self):
        """Dispatch to the selected alignment mode."""
        if self._mode == 'parallel':
            self._align_parallel()
        else:
            self._align_perpendicular()

    def _align_parallel(self):
        """Spin the rover to be parallel to the wall in front (wall on the right)."""
        self.get_logger().info('Waiting for LiDAR scan...')
        self._scan_event.wait()

        points = self._front_wall_points(self._scan)
        if len(points) < 5:
            self.get_logger().error('Not enough wall points in front sector — aborting.')
            return

        wall_angle = self._wall_angle(points)

        # After rotating by `rot`, the wall centre (currently at roughly +x) ends up
        # at (d·cos(rot), −d·sin(rot)).  For wall on the right: −d·sin(rot) < 0,
        # i.e. sin(rot) > 0, so 0 < rot < π.  The wall line has two antipodal
        # directions; pick the one that satisfies this constraint.
        rot = wall_angle
        if math.sin(rot) <= 0:
            rot += math.pi
        rot = (rot + math.pi) % (2 * math.pi) - math.pi
        rot_deg = math.degrees(rot)

        self.get_logger().info(
            f'Wall angle: {math.degrees(wall_angle):.1f}°  →  spinning {rot_deg:.1f}° (parallel, wall on right)'
        )
        self._send_spin(rot_deg)
        self.get_logger().info('Spin complete — rover is now parallel to wall (wall on right).')

    def _align_perpendicular(self):
        """Spin the rover to face directly into the wall in front."""
        self.get_logger().info('Waiting for LiDAR scan...')
        self._scan_event.wait()

        points = self._front_wall_points(self._scan)
        if len(points) < 5:
            self.get_logger().error('Not enough wall points in front sector — aborting.')
            return

        wall_angle = self._wall_angle(points)

        # Rotate so the wall runs at ±90° in the new robot frame (left-right),
        # making the robot face the wall.  Two candidates; pick the smaller rotation.
        rot_a = (wall_angle - math.pi / 2 + math.pi) % (2 * math.pi) - math.pi
        rot_b = (wall_angle + math.pi / 2 + math.pi) % (2 * math.pi) - math.pi
        rot = rot_a if abs(rot_a) <= abs(rot_b) else rot_b
        rot_deg = math.degrees(rot)

        self.get_logger().info(
            f'Wall angle: {math.degrees(wall_angle):.1f}°  →  spinning {rot_deg:.1f}° (perpendicular)'
        )
        self._send_spin(rot_deg)
        self.get_logger().info('Spin complete — rover is now perpendicular to wall (facing it).')

    def _wall_angle(self, points):
        """Return the angle (radians) of the dominant wall direction via SVD."""
        pts = np.array(points)
        centered = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        wall_dir = vt[0]
        return math.atan2(wall_dir[1], wall_dir[0])

    def _front_wall_points(self, scan):
        """Return (x, y) Cartesian points from the front ±FRONT_HALF_ANGLE_DEG sector."""
        half_rad = math.radians(FRONT_HALF_ANGLE_DEG)
        points = []
        for i, r in enumerate(scan.ranges):
            if not math.isfinite(r) or not (scan.range_min <= r <= scan.range_max):
                continue
            angle = scan.angle_min + i * scan.angle_increment
            if abs(angle) <= half_rad:
                points.append((r * math.cos(angle), r * math.sin(angle)))
        return points

    def _send_spin(self, angle_deg):
        """Send a spin goal to behavior_ctrl and block until it completes."""
        goal = Behavior.Goal()
        goal.command = json.dumps([{'type': 'spin', 'data': angle_deg}])

        done_event = threading.Event()

        def _result_cb(_):
            done_event.set()

        def _goal_cb(future):
            future.result().get_result_async().add_done_callback(_result_cb)

        self._behavior_client.send_goal_async(goal).add_done_callback(_goal_cb)
        done_event.wait()


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
