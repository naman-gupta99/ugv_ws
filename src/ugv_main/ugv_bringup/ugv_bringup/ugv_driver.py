#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import serial  
import json  
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float32, Float32MultiArray, Bool
import subprocess
import time
from cv_bridge import CvBridge
import cv2
import os

# Initialize serial communication with the UGV
ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)

class UgvDriver(Node):
    def __init__(self, name):
        super().__init__(name)

        # Subscribe to velocity commands (cmd_vel topic)
        self.cmd_vel_sub_ = self.create_subscription(Twist, "cmd_vel", self.cmd_vel_callback, 10)

        # Subscribe to joint states (ugv/joint_states topic)
        self.joint_states_sub = self.create_subscription(JointState, 'ugv/joint_states', self.joint_states_callback, 10)

        # Subscribe to LED control data (ugv/led_ctrl topic)
        self.led_ctrl_sub = self.create_subscription(Float32MultiArray, 'ugv/led_ctrl', self.led_ctrl_callback, 10)

        # Subscribe to voltage data (voltage topic)
        self.voltage_sub = self.create_subscription(Float32, 'voltage', self.voltage_callback, 10)

        # Subscribe to image_raw and image_rect (/image_raw topic and /image_rect topic)
        self.image_raw_sub = self.create_subscription(Image, 'image_raw', self.image_raw_callback, 10)
        self.image_rect_sub = self.create_subscription(Image, 'image_rect', self.image_rect_callback, 10)

        self.image_capture_sub = self.create_subscription(Bool, 'image_capture', self.image_capture_callback, 10)

        self.bridge = CvBridge()
        self.curr_image_raw = None
        self.curr_image_rect = None

    # Callback for processing velocity commands
    def cmd_vel_callback(self, msg):
        linear_velocity = msg.linear.x
        angular_velocity = msg.angular.z

        # Apply minimum threshold to angular velocity if linear velocity is zero
        if linear_velocity == 0:
            if 0 < angular_velocity < 0.2:
                angular_velocity = 0.2
            elif -0.2 < angular_velocity < 0:
                angular_velocity = -0.2

        # Send the velocity data to the UGV as a JSON string
        data = json.dumps({'T': '13', 'X': linear_velocity, 'Z': angular_velocity}) + "\n"
        ser.write(data.encode())

    # Callback for processing joint state updates
    def joint_states_callback(self, msg):
        if msg.header.frame_id != 'ugv_joint_state':
            return
        header = {
            'stamp': {
                'sec': msg.header.stamp.sec,
                'nanosec': msg.header.stamp.nanosec,
            },
            'frame_id': msg.header.frame_id,
        }

        # Extract joint positions and convert to degrees
        name = msg.name
        position = msg.position

        x_rad = position[name.index('pt_base_link_to_pt_link1')]
        y_rad = position[name.index('pt_link1_to_pt_link2')]

        # Convert radians to degrees for the UGV
        x_degree = x_rad * 180.0 / 3.14159
        y_degree = y_rad * 180.0 / 3.14159

        # Send the joint data as a JSON string to the UGV
        joint_data = json.dumps({
            'T': 133, 
            'X': x_degree, 
            'Y': y_degree, 
            "SPD": 0,
            "ACC": 0,
        }) + "\n"
                
        ser.write(joint_data.encode())

    # Callback for processing LED control commands
    def led_ctrl_callback(self, msg):
        IO4 = msg.data[0]
        IO5 = msg.data[1]
        
        # Send LED control data as a JSON string to the UGV
        led_ctrl_data = json.dumps({
            'T': 132, 
            "IO4": IO4,
            "IO5": IO5,
        }) + "\n"
                
        ser.write(led_ctrl_data.encode())

    # Callback for processing voltage data
    def voltage_callback(self, msg):
        voltage_value = msg.data

        # If voltage drops below a threshold, play a low battery warning sound
        if 0.1 < voltage_value < 9: 
            subprocess.run(['aplay', '-D', 'plughw:3,0', '/home/ws/ugv_ws/src/ugv_main/ugv_bringup/ugv_bringup/low_battery.wav'])
            time.sleep(5)

    # Callback for image_raw
    def image_raw_callback(self, msg):
        self.curr_image_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    
    # Callback for image_rect
    def image_rect_callback(self, msg):
        self.curr_image_rect = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    # Callback for image capture request
    def image_capture_callback(self, msg):
        if msg.data:
            self.get_logger().info("Capturing images...")
            # Create a folder named by the current date
            date_folder = time.strftime('%Y-%m-%d')
            folder_path = f'/home/ws/ugv_ws/captures/{date_folder}'
            os.makedirs(folder_path, exist_ok=True)

            # Save images with the current time as the filename
            timestamp = time.strftime('%H-%M-%S')
            if self.curr_image_raw is not None:
                cv2.imwrite(f'{folder_path}/{timestamp}_image_raw.jpg', self.curr_image_raw)
            if self.curr_image_rect is not None:
                cv2.imwrite(f'{folder_path}/{timestamp}_image_rect.jpg', self.curr_image_rect)
        

def main(args=None):
    rclpy.init(args=args)
    node = UgvDriver("ugv_driver")
    
    try:
        rclpy.spin(node)  # Keep the node running and handling callbacks
    except KeyboardInterrupt:
        pass  # Graceful shutdown on user interrupt
    finally:
        node.destroy_node()
        rclpy.shutdown()
        ser.close()  # Close the serial connection

if __name__ == '__main__':
    main()
