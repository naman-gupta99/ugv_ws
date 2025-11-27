#!/usr/bin/env python
# encoding: utf-8

import cv2
from cv_bridge import CvBridge
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
import threading
import queue
import time

from .agent.audit_toolset import audit_state_instance
from .agent.validation_react_agent import ValidationReactAgent
import os


class LlmPtCtrl(Node):

    def set_new_angles(self, x_rad, y_rad):
        self._publish_event.clear()
        with self._angle_lock:
            self.x_rad = x_rad
            self.y_rad = y_rad

        # Block until the publisher loop confirms the joint state broadcast.
        waited = 0.0
        self.publish_joint_state()
        while not self._publish_event.wait(timeout=0.5):
            waited += 0.5
            if waited >= 2.0:
                self.get_logger().warn(
                    "Still waiting for joint state publish after angle update."
                )
                waited = 0.0

        # Actuation Delay
        """
        TODO: There should either be a feedback mechanism confirming when the actuation is complete,
        or a delay based on the difference in angles and known actuation speed.
        """
        time.sleep(4.0)
        
        self.capture_image()

        return True

    def __init__(self, name):
        super().__init__(name)

        # Subscriptions
        self.image_raw_sub = self.create_subscription(
            Image, "image_raw", self.image_raw_callback, 10
        )

        # Publisher
        self.pub_cmdJoint = self.create_publisher(JointState, "ugv/joint_states", 10)

        # Pan-Tilt Camera State
        self.x_rad = 0.0
        self.y_rad = 0.0

        # Image saving state
        self.curr = 0
        self.bridge = CvBridge()
        self.curr_image_raw = None
        self.image_lock = threading.Lock()
        self._angle_lock = threading.Lock()
        self._publish_event = threading.Event()
        self._publish_event.set()

        # Timer to periodically call publish_joint_state
        # self.timer = self.create_timer(0.1, self.publish_joint_state)

        # Agent Controller
        audit_state_instance.update_rover_state_func = self.set_new_angles
        self.validation_agent_thread = threading.Thread(
            target=self._run_validation_agent, daemon=True
        )
        self.validation_agent_thread.start()

    # Callback for image_raw
    def image_raw_callback(self, msg):
        with self.image_lock:
            self.curr_image_raw = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )

    def capture_image(self):
        # Process image if available
        image_to_save = None
        with self.image_lock:
            if self.curr_image_raw is not None:
                image_to_save = self.curr_image_raw.copy()

        if image_to_save is not None:
            date_folder = time.strftime("%Y%m%d%H%M")
            capture_dir = f"/home/ws/ugv_ws/captures/{date_folder}"
            os.makedirs(capture_dir, exist_ok=True)
            cv2.imwrite(
                f"/home/ws/ugv_ws/captures/{date_folder}/decision{self.curr}.png",
                image_to_save,
            )
            self.get_logger().info(
                f"Image saved successfully as /{date_folder}/decision{self.curr}.png"
            )
            self.curr += 1
        elif image_to_save is None:
            self.get_logger().warn(
                "No image_raw received yet, delaying image capture but publishing joint state."
            )

    def publish_joint_state(self):
        try:

            with self._angle_lock:
                x_rad = self.x_rad
                y_rad = self.y_rad
                print(f"Publishing joint state with x_rad: {x_rad}, y_rad: {y_rad}")

            # Always publish the joint state with current values
            joint_state = JointState()
            joint_state.header.frame_id = "ugv_joint_state"
            joint_state.header.stamp = self.get_clock().now().to_msg()

            # Assuming the robot has two joints for x and y rotation
            joint_state.name = [
                "left_up_wheel_link_joint",
                "left_down_wheel_link_joint",
                "right_up_wheel_link_joint",
                "right_down_wheel_link_joint",
                "pt_base_link_to_pt_link1",
                "pt_link1_to_pt_link2",
            ]
            joint_state.position = [0.0, 0.0, 0.0, 0.0, x_rad, y_rad]

            self.pub_cmdJoint.publish(joint_state)
        except Exception as exc:
            self.get_logger().error(f"Failed to publish joint state: {exc}")
        finally:
            self._publish_event.set()

    def _run_validation_agent(self):
        try:
            ValidationReactAgent().execute()
        except Exception as exc:
            self.get_logger().error(f"ValidationReactAgent failed: {exc}")

    def on_shutdown(self):
        with self._angle_lock:
            self.x_rad = 0.0
            self.y_rad = 0.0
            self.capture_image = False
        self.publish_joint_state()
        self.get_logger().info("Shutting down LlmPtCtrl node.")


def main(args=None):
    rclpy.init(args=args)

    pt_ctrl = LlmPtCtrl("llm_pt_ctrl")

    try:
        rclpy.spin(pt_ctrl)
    except KeyboardInterrupt:
        pass
    finally:
        pt_ctrl.on_shutdown()
        pt_ctrl.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
