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

class LlmPtCtrl(Node):
    def __init__(self, name):
        super().__init__(name)

        # Subscribe to image_raw (/image_raw topic topic)
        self.image_raw_sub = self.create_subscription(Image, 'image_raw', self.image_raw_callback, 10)

        self.curr = 0
        self.coord_queue = queue.Queue()

        self.pub_cmdJoint = self.create_publisher(
            JointState, 'ugv/joint_states', 10
        )

        # Get initial values without blocking the main thread
        self.x_rad = 0.0
        self.y_rad = 0.0
        
        self.get_initial_values()

        self.bridge = CvBridge()
        self.curr_image_raw = None
        self.image_lock = threading.Lock()

        # Timer to periodically call publish_joint_state
        self.timer = self.create_timer(0.1, self.publish_joint_state)
        
        # Flag to track if we're getting coordinates
        self.waiting_for_coords = False

        # Time tracking for delays
        self.coords_processed_time = 0
        self.delay_duration = 2.0  # 2 second delay
        
    def get_initial_values(self):
        """Get initial x_rad and y_rad values in a separate thread"""
        try:
            x_rad = float(input("x_rad:"))
            y_rad = float(input("y_rad:"))
            self.x_rad = x_rad
            self.y_rad = y_rad
            self.get_logger().info(f"Initial values set: x_rad={self.x_rad}, y_rad={self.y_rad}")
        except ValueError:
            self.get_logger().error("Invalid input for initial values, using defaults.")
    
    # Callback for image_raw
    def image_raw_callback(self, msg):
        with self.image_lock:
            self.curr_image_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def publish_joint_state(self):
        current_time = time.time()

        # Process image if available
        image_to_save = None
        with self.image_lock:
            if self.curr_image_raw is not None:
                image_to_save = self.curr_image_raw.copy()

        if image_to_save is not None:
            if not self.waiting_for_coords and current_time - self.coords_processed_time >= self.delay_duration:
                date_folder = time.strftime('%Y-%m-%d')
                cv2.imwrite(f'/home/ws/ugv_ws/captures/{date_folder}/decision{self.curr}.png', image_to_save)
                self.get_logger().info(f"Image saved successfully as decision{self.curr}.png")
                self.curr += 1
        else:
            self.get_logger().warn("No image_raw received yet, cannot save image.")
            return

        # Request coordinates if we're not already waiting
        if not self.waiting_for_coords and current_time - self.coords_processed_time >= self.delay_duration:
            self.waiting_for_coords = True
            # Signal the input thread to get coordinates
            self.input_queue.put("get_coords")

        # Try to get coordinates from the queue (non-blocking)
        try:
            coords = self.coord_queue.get_nowait()
            # Record the time when coordinates were processed
            self.coords_processed_time = time.time()
            self.waiting_for_coords = False
            
            if coords is not None:
                try:
                    window_center_x = float(coords[0])
                    window_center_y = float(coords[1])
                    
                    # Optional: Use all 4 points if provided
                    # if len(coords) == 8:
                    #     window_center_x = (sum(float(coords[i]) for i in range(0, 8, 2)) / 4.0)
                    #     window_center_y = (sum(float(coords[i]) for i in range(1, 8, 2)) / 4.0)
                    
                    image_shape = None
                    with self.image_lock:
                        if self.curr_image_raw is not None:
                            image_shape = self.curr_image_raw.shape
                    
                    if image_shape:
                        image_center_x = image_shape[1] / 2.0
                        image_center_y = image_shape[0] / 2.0

                        dx = image_center_x - window_center_x 
                        dy = image_center_y - window_center_y

                        self.get_logger().info(
                            f"Computed dx: {dx:.2f}, dy: {dy:.2f} from window center ({window_center_x:.2f}, {window_center_y:.2f}) "
                            f"and image center ({image_center_x:.2f}, {image_center_y:.2f})."
                        )
                        
                        # Update the radians based on dx and dy and the distance from the image center
                        self.x_rad -= float(dx) * math.pi / 1800
                        self.y_rad += float(dy) * math.pi / 1800
                        self.get_logger().info(f"Updated x_rad: {self.x_rad}, y_rad: {self.y_rad}")
                except ValueError as e:
                    self.get_logger().error(f"Error processing coordinates: {e}")
        except queue.Empty:
            # No coordinates available yet, continue with the last known values
            pass

        # Always publish the joint state with current values
        joint_state = JointState()
        joint_state.header.frame_id = "ugv_joint_state"
        joint_state.header.stamp = self.get_clock().now().to_msg()
        
        # Assuming the robot has two joints for x and y rotation
        joint_state.name = [
            'left_up_wheel_link_joint', 
            'left_down_wheel_link_joint', 
            'right_up_wheel_link_joint', 
            'right_down_wheel_link_joint', 
            'pt_base_link_to_pt_link1', 
            'pt_link1_to_pt_link2'
        ]
        joint_state.position = [0.0, 0.0, 0.0, 0.0, self.x_rad, self.y_rad]
        
        self.pub_cmdJoint.publish(joint_state)

def main(args=None):
    rclpy.init(args=args)

    pt_ctrl = LlmPtCtrl('llm_pt_ctrl')
    
    try:
        rclpy.spin(pt_ctrl)
    except KeyboardInterrupt:
        pass
    finally:
        pt_ctrl.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()