#!/usr/bin/env python
# encoding: utf-8

import cv2
from cv_bridge import CvBridge
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image, LaserScan
import threading
import queue
import time

from .agent.audit_toolset import audit_state_instance
from .agent.validation_react_agent import ValidationReactAgent
import os
import pathlib
import tempfile
import traceback


class LlmPtCtrl(Node):

    def set_new_angles(self, x_rad, y_rad):
        self.x_rad = x_rad
        self.y_rad = y_rad
            
        self.publish_joint_state()

        # Actuation Delay
        """
        TODO: There should either be a feedback mechanism confirming when the actuation is complete,
        or a delay based on the difference in angles and known actuation speed.
        """
        time.sleep(2.0)

        return True
    
    def get_laser_scan(self):
        return self.curr_lidar_scan
    
    def get_current_angles(self):
        return self.x_rad, self.y_rad

    def __init__(self, name):
        super().__init__(name)

        # Subscriptions
        self.image_raw_sub = self.create_subscription(
            Image, "image_raw", self.image_raw_callback, 10
        )
        self.lidar_sub = self.create_subscription(
            LaserScan, "scan", self.lidar_callback, 10
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
        
        # Laser Scan state
        self.curr_lidar_scan = None

        # Capture directory (configurable via env var). Prefer mounted workspace if present,
        # otherwise fall back to user's home directory under ~/ugv_ws/captures.
        env_dir = os.environ.get("UGV_CAPTURE_DIR")
        default_workspace_dir = "/home/ws/ugv_ws/captures"
        home_fallback = os.path.expanduser("~/ugv_ws/captures")
        if env_dir:
            self.capture_base_dir = env_dir
        elif os.path.isdir(os.path.dirname(default_workspace_dir)):
            # If /home/ws/ugv_ws exists on the system, use it (common for your setup)
            self.capture_base_dir = default_workspace_dir
        else:
            self.capture_base_dir = home_fallback

        # Timer to periodically call publish_joint_state
        # self.timer = self.create_timer(0.1, self.publish_joint_state)

        # Agent Controller
        audit_state_instance.update_rover_state_func = self.set_new_angles
        audit_state_instance.get_laser_scan_func = self.get_laser_scan
        audit_state_instance.get_current_angles_func = self.get_current_angles
        audit_state_instance.capture_image_func = self.capture_image
        
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
            
    def lidar_callback(self, msg):
        self.curr_lidar_scan = msg

    def capture_image(self):
        # Process image if available
        image_to_save = None
        with self.image_lock:
            if self.curr_image_raw is not None:
                image_to_save = self.curr_image_raw.copy()

        if image_to_save is not None:
            date_folder = time.strftime("%Y%m%d%H%M")
            capture_dir = os.path.join(self.capture_base_dir, date_folder)
            try:
                # Use pathlib to ensure parent directories are created
                pathlib.Path(capture_dir).mkdir(parents=True, exist_ok=True)
                filename = os.path.join(capture_dir, f"decision{self.curr}.png")
                ok = cv2.imwrite(filename, image_to_save)
                if ok:
                    self.get_logger().info(
                        f"Image saved successfully as {filename}"
                    )
                else:
                    # cv2 can fail silently; log and fallback to temp
                    raise IOError("cv2.imwrite returned False")
            except Exception as exc:
                # Log full traceback to help debugging permission or filesystem issues
                tb = traceback.format_exc()
                self.get_logger().error(
                    f"Failed to save image to {capture_dir}: {exc}\n{tb}"
                )
                # Fallback: create a temp directory we can write to
                try:
                    tmpdir = tempfile.mkdtemp(prefix="ugv_captures_")
                    tmpfile = os.path.join(tmpdir, f"decision{self.curr}.png")
                    cv2.imwrite(tmpfile, image_to_save)
                    self.get_logger().info(
                        f"Image saved to fallback location {tmpfile}"
                    )
                except Exception as exc2:
                    self.get_logger().error(
                        f"Fallback save also failed: {exc2}\n{traceback.format_exc()}"
                    )
            self.curr += 1
        elif image_to_save is None:
            self.get_logger().warn(
                "No image_raw received yet, delaying image capture but publishing joint state."
            )

    def publish_joint_state(self):
        try:

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
