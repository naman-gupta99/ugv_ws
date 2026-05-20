#!/usr/bin/env python
# encoding: utf-8

"""
NavCtrl: navigate to a map pose via Nav2's navigate_to_pose action server.
"""

import threading

from action_msgs.msg import GoalStatus
from rcl_interfaces.msg import Log
from rclpy.action import ActionClient
from rclpy.node import Node

from nav2_msgs.action import NavigateToPose


class NavCtrl(Node):

    def __init__(self):
        super().__init__('nav_ctrl')
        self._client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._nav_failed_event = threading.Event()
        self._nav_failed_reason = ''
        self.create_subscription(Log, '/rosout', self._rosout_callback, 10)

    def _rosout_callback(self, msg: Log):
        if msg.name == 'bt_navigator' and 'Goal failed' in msg.msg:
            self._nav_failed_reason = msg.msg
            self._nav_failed_event.set()

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
        self._nav_failed_event.clear()
        self._nav_failed_reason = ''

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
        failure_reason = ['']

        def _result_cb(future):
            result = future.result()
            status = getattr(result, 'status', GoalStatus.STATUS_UNKNOWN)
            if status == GoalStatus.STATUS_SUCCEEDED:
                success[0] = True
            else:
                failure_reason[0] = f'Navigation action ended with status {status}.'
            done.set()

        def _goal_cb(future):
            handle = future.result()
            if not handle.accepted:
                failure_reason[0] = 'Navigation goal was rejected.'
                self.get_logger().error(failure_reason[0])
                done.set()
                return
            self.get_logger().info('Navigation goal accepted — robot is moving.')
            handle.get_result_async().add_done_callback(_result_cb)

        self._client.send_goal_async(goal).add_done_callback(_goal_cb)
        while not done.wait(timeout=0.1):
            if self._nav_failed_event.is_set():
                failure_reason[0] = self._nav_failed_reason or 'bt_navigator reported: Goal failed.'
                done.set()
                break

        if success[0]:
            self.get_logger().info('Navigation complete.')
        else:
            self.get_logger().error(failure_reason[0] or 'Navigation failed.')
        return success[0]
