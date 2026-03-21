#!/usr/bin/env python
# encoding: utf-8

"""
NavCtrl: navigate to a map pose via Nav2's navigate_to_pose action server.
"""

import threading

from rclpy.action import ActionClient
from rclpy.node import Node

from nav2_msgs.action import NavigateToPose


class NavCtrl(Node):

    def __init__(self):
        super().__init__('nav_ctrl')
        self._client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

    def navigate_to(self, x: float, y: float, qz: float, qw: float) -> bool:
        """
        Send a NavigateToPose goal and block until the robot arrives.

        Args:
            x, y : target position in the map frame.
            qz, qw: quaternion orientation components (heading).
                    For a heading angle θ: qz = sin(θ/2), qw = cos(θ/2).

        Returns:
            True if the goal was accepted and completed, False otherwise.
        """
        self.get_logger().info(f'Navigating to x={x:.3f}, y={y:.3f}')

        if not self._client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('navigate_to_pose action server not available.')
            return False

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation.z = float(qz)
        goal.pose.pose.orientation.w = float(qw)

        done = threading.Event()
        success = [False]

        def _result_cb(future):
            success[0] = True
            done.set()

        def _goal_cb(future):
            handle = future.result()
            if not handle.accepted:
                self.get_logger().error('Navigation goal was rejected.')
                done.set()
                return
            self.get_logger().info('Navigation goal accepted — robot is moving.')
            handle.get_result_async().add_done_callback(_result_cb)

        self._client.send_goal_async(goal).add_done_callback(_goal_cb)
        done.wait()

        if success[0]:
            self.get_logger().info('Navigation complete.')
        return success[0]
