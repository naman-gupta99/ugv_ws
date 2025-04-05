#!/usr/bin/env python
# encoding: utf-8

import getpass
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import Int32, Bool
import pygame

def get_joystick_names():
    pygame.init()
    pygame.joystick.init()

    joystick_names = []
    joystick_count = pygame.joystick.get_count()
    
    if joystick_count == 0:
        print("no")
    else:
        for i in range(joystick_count):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            joystick_names.append(joystick.get_name())

    pygame.quit()
    return joystick_names
    
class JoyTeleop(Node):
	def __init__(self,name):
		super().__init__(name)
		self.Joy_active = True
		self.user_name = getpass.getuser()
		self.linear_Gear = 1
		self.angular_Gear = 1
		self.x_rad = 0
		self.y_rad = 0
		
		#create pub
		self.pub_cmdVel = self.create_publisher(Twist,'cmd_vel',  10)
		self.pub_cmdJoint = self.create_publisher(JointState, 'ugv/joint_states',  10)
		self.pub_JoyState = self.create_publisher(Bool,"JoyState",  10)
		self.pub_imgCapture = self.create_publisher(Bool,"image_capture",  10)
		
		#create sub
		self.sub_Joy = self.create_subscription(Joy,'joy', self.buttonCallback,10)
		
		#declare parameter and get the value
		self.declare_parameter('xspeed_limit',0.65)
		self.declare_parameter('yspeed_limit',0.5)
		self.declare_parameter('angular_speed_limit',1.0)
		self.xspeed_limit = self.get_parameter('xspeed_limit').get_parameter_value().double_value
		self.yspeed_limit = self.get_parameter('yspeed_limit').get_parameter_value().double_value
		self.angular_speed_limit = self.get_parameter('angular_speed_limit').get_parameter_value().double_value
		joysticks = get_joystick_names()
		self.joysticks = joysticks[0] if len(joysticks) != 0 else "no"
		self.switch_dict = {
			"Xbox 360 Controller": [9,10,3,4],
			"SHANWAN Android Gamepad": [13,14,2,None],
		}

	def buttonCallback(self,joy_data):
		print(joy_data)
		if not isinstance(joy_data, Joy): return
		if self.user_name == "root": self.user_jetson(joy_data)
		else: self.user_pc(joy_data)
    
	def user_jetson(self, joy_data):
		# self.get_logger().info(f"joy_data: {joy_data}")
			#linear Gear control
		if joy_data.buttons[self.switch_dict[self.joysticks][0]] == 1:
			if self.linear_Gear == 1.0: self.linear_Gear = 1.0 / 3
			elif self.linear_Gear == 1.0 / 3: self.linear_Gear = 2.0 / 3
			elif self.linear_Gear == 2.0 / 3: self.linear_Gear = 1
			# angular Gear control
		if joy_data.buttons[self.switch_dict[self.joysticks][1]] == 1:
			if self.angular_Gear == 1.0: self.angular_Gear = 1.0 / 4
			elif self.angular_Gear == 1.0 / 4: self.angular_Gear = 1.0 / 2
			elif self.angular_Gear == 1.0 / 2: self.angular_Gear = 3.0 / 4
			elif self.angular_Gear == 3.0 / 4: self.angular_Gear = 1.0
		if joy_data.buttons[self.switch_dict[self.joysticks][3]] == 1:
			self.get_logger().info("image capture")
			self.pub_imgCapture.publish(Bool(data=True))
		xlinear_speed = self.filter_data(joy_data.axes[1]) * self.xspeed_limit * self.linear_Gear
			#ylinear_speed = self.filter_data(joy_data.axes[2]) * self.yspeed_limit * self.linear_Gear
		ylinear_speed = self.filter_data(joy_data.axes[0]) * self.yspeed_limit * self.linear_Gear
		angular_speed = self.filter_data(joy_data.axes[self.switch_dict[self.joysticks][2]]) * self.angular_speed_limit * self.angular_Gear
		if xlinear_speed > self.xspeed_limit: xlinear_speed = self.xspeed_limit
		elif xlinear_speed < -self.xspeed_limit: xlinear_speed = -self.xspeed_limit
		if ylinear_speed > self.yspeed_limit: ylinear_speed = self.yspeed_limit
		elif ylinear_speed < -self.yspeed_limit: ylinear_speed = -self.yspeed_limit
		if angular_speed > self.angular_speed_limit: angular_speed = self.angular_speed_limit
		elif angular_speed < -self.angular_speed_limit: angular_speed = -self.angular_speed_limit
		twist = Twist()
		twist.linear.x = xlinear_speed
		twist.linear.y = ylinear_speed
		twist.angular.z = angular_speed
    
		self.x_rad += (joy_data.axes[6] * math.pi/180)  # 0.017453 = π/180
		self.y_rad += (joy_data.axes[7] * math.pi/180)
		
		self.x_rad = min(max(self.x_rad, -math.pi), math.pi)  # -π to π radians
		self.y_rad = min(max(self.y_rad, -math.pi/4), math.pi/2)  # -π/4 to π/2 radians
  
		jointState = JointState()
		# Set proper header information
		jointState.header.frame_id = "ugv_joint_state"
		jointState.header.stamp = self.get_clock().now().to_msg()
		jointState.name = [
			'left_up_wheel_link_joint', 
			'left_down_wheel_link_joint', 
			'right_up_wheel_link_joint', 
			'right_down_wheel_link_joint', 
			'pt_base_link_to_pt_link1', 
			'pt_link1_to_pt_link2'
		]
		jointState.position = [0.0, 0.0, 0.0, 0.0, self.x_rad, self.y_rad]
    
		self.get_logger().info("x_rad: %s, y_rad: %s" % (self.x_rad, self.y_rad))

		if self.Joy_active == True:
			print("joy control now")
			self.pub_cmdVel.publish(twist)
			self.pub_cmdJoint.publish(jointState)
        
	def user_pc(self, joy_data):
        # Gear control
		if joy_data.buttons[self.switch_dict[self.joysticks][0]] == 1:
			if self.linear_Gear == 1.0: self.linear_Gear = 1.0 / 3
			elif self.linear_Gear == 1.0 / 3: self.linear_Gear = 2.0 / 3
			elif self.linear_Gear == 2.0 / 3: self.linear_Gear = 1
		if joy_data.buttons[self.switch_dict[self.joysticks][1]] == 1:
			if self.angular_Gear == 1.0: self.angular_Gear = 1.0 / 4
			elif self.angular_Gear == 1.0 / 4: self.angular_Gear = 1.0 / 2
			elif self.angular_Gear == 1.0 / 2: self.angular_Gear = 3.0 / 4
			elif self.angular_Gear == 3.0 / 4: self.angular_Gear = 1.0
		xlinear_speed = self.filter_data(joy_data.axes[1]) * self.xspeed_limit * self.linear_Gear
		ylinear_speed = self.filter_data(joy_data.axes[0]) * self.yspeed_limit * self.linear_Gear
		angular_speed = self.filter_data(joy_data.axes[self.switch_dict[self.joysticks][2]]) * self.angular_speed_limit * self.angular_Gear
		if xlinear_speed > self.xspeed_limit: xlinear_speed = self.xspeed_limit
		elif xlinear_speed < -self.xspeed_limit: xlinear_speed = -self.xspeed_limit
		if ylinear_speed > self.yspeed_limit: ylinear_speed = self.yspeed_limit
		elif ylinear_speed < -self.yspeed_limit: ylinear_speed = -self.yspeed_limit
		if angular_speed > self.angular_speed_limit: angular_speed = self.angular_speed_limit
		elif angular_speed < -self.angular_speed_limit: angular_speed = -self.angular_speed_limit
		twist = Twist()
		twist.linear.x = xlinear_speed
		twist.linear.y = ylinear_speed
		twist.angular.z = angular_speed
		
		# Add joint state handling similar to user_jetson
		self.x_rad += (joy_data.axes[6] if len(joy_data.axes) > 6 else 0) * math.pi/180
		self.y_rad += (joy_data.axes[7] if len(joy_data.axes) > 7 else 0) * math.pi/180
		
		self.x_rad = min(max(self.x_rad, -math.pi), math.pi)  # -π to π radians
		self.y_rad = min(max(self.y_rad, -math.pi/4), math.pi/2)  # -π/4 to π/2 radians
  
		jointState = JointState()
		# Set proper header information
		jointState.header.frame_id = "ugv_joint_state"
		jointState.header.stamp = self.get_clock().now().to_msg()
		jointState.name = [
			'left_up_wheel_link_joint', 
			'left_down_wheel_link_joint', 
			'right_up_wheel_link_joint', 
			'right_down_wheel_link_joint', 
			'pt_base_link_to_pt_link1', 
			'pt_link1_to_pt_link2'
		]
		jointState.position = [0.0, 0.0, 0.0, 0.0, self.x_rad, self.y_rad]
		
		self.pub_cmdVel.publish(twist)
		if jointState.position[4] != 0 or jointState.position[5] != 0:
			self.pub_cmdJoint.publish(jointState)
        
	def filter_data(self, value):
		if abs(value) < 0.2: value = 0
		return value		
			
def main():
	rclpy.init()
	joy_ctrl = JoyTeleop('joy_ctrl')
	rclpy.spin(joy_ctrl)	
	
main()