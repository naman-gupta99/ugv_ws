import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from ugv_interface.action import Behavior
from geometry_msgs.msg import Twist, PoseStamped, Pose
from nav_msgs.msg import Odometry 
from sensor_msgs.msg import Image

import math
import json
import threading
import time

import numpy as np

a = "point_a" 
b = "point_b" 
c = "point_c" 
d = "point_d" 
e = "point_e" 
f = "point_f" 
g = "point_g" 

DEPTH_CLOSE_THRESHOLD_M = 0.30
DEPTH_IMAGE_TOPIC = '/oak/stereo/image_raw'
DRIVE_ON_HEADING_SPEED_M_S = 0.2
BACK_UP_SPEED_M_S = 0.1

class BehaviorController(Node):
    def __init__(self):
        super().__init__('behavior_ctrl')     
        self.callback_group = ReentrantCallbackGroup()
        self.pipeline_step_delay_s = 0.5
        self.bridge = CvBridge()
        self._depth_lock = threading.Lock()
        self._depth_close_obstacle_detected = False
        self._depth_min_distance_m = None
        # Create a subscription to the /odom topic to get the odometry data
        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10,
            callback_group=self.callback_group,
        )
        # Create a subscription to the /robot_pose topic to get the robot's current pose
        self.create_subscription(
            PoseStamped,
            '/robot_pose',
            self.robot_pose_callback,
            10,
            callback_group=self.callback_group,
        ) 
        # Create a subscription to the RGB-D depth image for motion safety.
        self.create_subscription(
            Image,
            DEPTH_IMAGE_TOPIC,
            self.depth_image_callback,
            10,
            callback_group=self.callback_group,
        )
        # Create an action server to handle the behavior action
        self.behavior_action_server = ActionServer(
            self,
            Behavior,
            'behavior',
            self.execute_callback,
            callback_group=self.callback_group,
        )
        # Create a publisher to the /cmd_vel topic to send velocity commands to the robot
        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        # Create a publisher to the /goal_pose topic to send goal poses to the robot
        self.goal_publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)
        # Initialize the distance and yaw variables
        self.distance = Pose().position
        self.yaw = 0.0
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.current_pose = None
        self.map_pose = None
        self.points = {}
        self.control_period_s = 0.05

    def robot_pose_callback(self, msg):
        # Store the current pose of the robot
        self.map_pose = msg.pose

    def depth_image_callback(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_m = np.asarray(depth, dtype=np.float32)

            if depth.dtype == np.uint16:
                depth_m *= 0.001

            valid_mask = np.isfinite(depth_m) & (depth_m > 0.0)
            close_mask = valid_mask & (depth_m < DEPTH_CLOSE_THRESHOLD_M)
            min_distance = float(np.min(depth_m[valid_mask])) if np.any(valid_mask) else None

            with self._depth_lock:
                self._depth_close_obstacle_detected = bool(np.any(close_mask))
                self._depth_min_distance_m = min_distance
        except Exception as exc:
            self.get_logger().warn(f'Failed to process depth image for safety check: {exc}')

    def _depth_hazard_message(self):
        with self._depth_lock:
            if not self._depth_close_obstacle_detected:
                return None
            min_distance = self._depth_min_distance_m

        if min_distance is None:
            return f'Depth obstacle detected closer than {DEPTH_CLOSE_THRESHOLD_M:.2f} m.'
        return f'Depth obstacle detected at {min_distance:.2f} m (< {DEPTH_CLOSE_THRESHOLD_M:.2f} m).'

    def _abort_motion_if_depth_hazard(self):
        hazard_message = self._depth_hazard_message()
        if hazard_message:
            self.get_logger().warn(hazard_message)
            self.stop()
            return True, hazard_message
        return False, 'ok'
                
    def execute_callback(self, goal_handle):
        # Log the start of the goal execution
        self.get_logger().info('Executing goal...')
        result = Behavior.Result()

        try:
            # Parse the command list from the goal request.
            json_list = json.loads(goal_handle.request.command)
            if not isinstance(json_list, list):
                raise ValueError('command must be a JSON list')
        except Exception as exc:
            goal_handle.abort()
            self._set_result(result, False, f'Invalid command payload: {exc}')
            return result

        for idx, json_data in enumerate(json_list):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self._set_result(result, False, 'Goal canceled')
                return result

            try:
                command_type = json_data['type']
                data_value = json_data.get('data', 0.0)
            except Exception as exc:
                goal_handle.abort()
                self._set_result(result, False, f'Invalid command entry: {exc}')
                return result

            ok, message = self.execute_behavior(goal_handle, command_type, data_value)
            if not ok:
                goal_handle.abort()
                self._set_result(result, False, message)
                return result

            # Add a small inter-step delay so sequential commands have a deterministic handoff.
            if idx < len(json_list) - 1:
                time.sleep(self.pipeline_step_delay_s)

        goal_handle.succeed()
        result = Behavior.Result()
        self._set_result(result, True, 'Behavior command sequence completed')
       
        return result

    def _set_result(self, result, success, message=''):
        result.result = bool(success)
        if hasattr(result, 'message'):
            result.message = message
        elif message and not success:
            self.get_logger().warn(message)

    def execute_behavior(self, goal_handle, command_type, data_value):
        try:
            if command_type == 'stop':
                self.stop()
                self._publish_feedback(
                    goal_handle,
                    command_type='stop',
                    target_value=0.0,
                    current_value=0.0,
                    progress=1.0,
                    is_settled=True,
                )
                return True, 'ok'

            if command_type == 'drive_on_heading':
                return self.drive_on_heading(goal_handle, float(data_value))
            if command_type == 'back_up':
                return self.back_up(goal_handle, float(data_value))
            if command_type == 'spin':
                return self.spin(goal_handle, float(data_value))
            if command_type == 'save_map_point':
                self.save_map_point(str(data_value))
                return True, 'ok'
            if command_type == 'pub_nav_point':
                self.pub_nav_point(str(data_value))
                return True, 'ok'

            return False, f'Unknown command type: {command_type}'
        except Exception as exc:
            self.get_logger().error(f'Error executing behavior command {command_type}: {exc}')
            return False, str(exc)
    
    def odom_callback(self, msg):
        # Get the orientation of the robot
        q1 = msg.pose.pose.orientation.x
        q2 = msg.pose.pose.orientation.y
        q3 = msg.pose.pose.orientation.z
        q0 = msg.pose.pose.orientation.w

        # Calculate the yaw of the robot
        siny_cosp = 2 * (q0 * q3 + q1 * q2)
        cosy_cosp = 1 - 2 * (q2 * q2 + q3 * q3)
        
        # Store the distance and yaw of the robot
        self.distance = msg.pose.pose.position
        self.yaw = math.atan2(siny_cosp, cosy_cosp)
        self.linear_velocity = msg.twist.twist.linear.x
        self.angular_velocity = msg.twist.twist.angular.z

    def _publish_feedback(self, goal_handle, command_type, target_value, current_value, progress, is_settled):
        feedback = Behavior.Feedback()
        feedback.feedback = True
        if hasattr(feedback, 'command_type'):
            feedback.command_type = command_type
        if hasattr(feedback, 'target_value'):
            feedback.target_value = float(target_value)
        if hasattr(feedback, 'current_value'):
            feedback.current_value = float(current_value)
        if hasattr(feedback, 'progress'):
            feedback.progress = max(0.0, min(1.0, float(progress)))
        if hasattr(feedback, 'linear_velocity'):
            feedback.linear_velocity = float(self.linear_velocity)
        if hasattr(feedback, 'angular_velocity'):
            feedback.angular_velocity = float(self.angular_velocity)
        if hasattr(feedback, 'is_settled'):
            feedback.is_settled = bool(is_settled)
        goal_handle.publish_feedback(feedback)

    @staticmethod
    def _wrap_angle(angle_rad):
        return (angle_rad + math.pi) % (2 * math.pi) - math.pi

    def _wait_for_settle(self, goal_handle, command_type, target_value, current_value, timeout_s=2.0):
        start_time = time.monotonic()
        settled_since = None
        while time.monotonic() - start_time < timeout_s:
            lin_ok = abs(self.linear_velocity) < 0.01
            ang_ok = abs(self.angular_velocity) < 0.01
            is_settled = lin_ok and ang_ok

            self._publish_feedback(
                goal_handle,
                command_type,
                target_value,
                current_value,
                1.0,
                is_settled,
            )

            if is_settled:
                if settled_since is None:
                    settled_since = time.monotonic()
                if time.monotonic() - settled_since >= 0.2:
                    return True
            else:
                settled_since = None
            time.sleep(self.control_period_s)
        return False
                               
    def drive_on_heading(self, goal_handle, distance):
        # Drive the robot on a heading
        self.get_logger().info('Drive on heading')
        twist_msg = Twist()
        twist_msg.linear.x = DRIVE_ON_HEADING_SPEED_M_S
        twist_msg.angular.z = 0.0
        target_distance = abs(distance)
        if target_distance < 1e-6:
            return True, 'No-op drive distance'

        hazard_active, hazard_message = self._abort_motion_if_depth_hazard()
        if hazard_active:
            return False, hazard_message
        
        # Store the start distance
        start_distance = self.distance
        start_time = time.monotonic()
           
        # Calculate the delta distance
        delta_distance = 0.0
        while abs(delta_distance) < target_distance:
            if goal_handle.is_cancel_requested:
                self.stop()
                return False, 'Goal canceled'
            if time.monotonic() - start_time > 60.0:
                self.stop()
                return False, 'Timeout in drive_on_heading'

            hazard_active, hazard_message = self._abort_motion_if_depth_hazard()
            if hazard_active:
                return False, hazard_message

            diff_x = self.distance.x - start_distance.x
            diff_y = self.distance.y - start_distance.y
            delta_distance = math.hypot(diff_x, diff_y)
            self.velocity_publisher.publish(twist_msg)
            progress = delta_distance / target_distance if target_distance > 0 else 1.0
            self._publish_feedback(
                goal_handle,
                command_type='drive_on_heading',
                target_value=target_distance,
                current_value=delta_distance,
                progress=progress,
                is_settled=False,
            )
            time.sleep(self.control_period_s)

        self.stop()
        self._wait_for_settle(goal_handle, 'drive_on_heading', target_distance, target_distance)
        return True, 'ok'

    def back_up(self, goal_handle, distance):
        # Back up the robot
        self.get_logger().info('Back up')
        twist_msg = Twist()
        twist_msg.linear.x = -BACK_UP_SPEED_M_S
        twist_msg.angular.z = 0.0
        target_distance = abs(distance)
        if target_distance < 1e-6:
            return True, 'No-op backup distance'

        hazard_active, hazard_message = self._abort_motion_if_depth_hazard()
        if hazard_active:
            return False, hazard_message
        
        # Store the start distance
        start_distance = self.distance
        start_time = time.monotonic()
  
        # Calculate the delta distance
        delta_distance = 0.0
        while abs(delta_distance) < target_distance:
            if goal_handle.is_cancel_requested:
                self.stop()
                return False, 'Goal canceled'
            if time.monotonic() - start_time > 60.0:
                self.stop()
                return False, 'Timeout in back_up'

            hazard_active, hazard_message = self._abort_motion_if_depth_hazard()
            if hazard_active:
                return False, hazard_message

            diff_x = self.distance.x - start_distance.x
            diff_y = self.distance.y - start_distance.y
            delta_distance = math.hypot(diff_x, diff_y)
            self.velocity_publisher.publish(twist_msg)
            progress = delta_distance / target_distance if target_distance > 0 else 1.0
            self._publish_feedback(
                goal_handle,
                command_type='back_up',
                target_value=target_distance,
                current_value=delta_distance,
                progress=progress,
                is_settled=False,
            )
            time.sleep(self.control_period_s)

        self.stop()
        self._wait_for_settle(goal_handle, 'back_up', target_distance, target_distance)
        return True, 'ok'
        
    def spin(self, goal_handle, angle):
        # Spin the robot
        self.get_logger().info('Spin')
        twist_msg = Twist()
        target_angle = abs(math.radians(angle))
        if target_angle < 1e-6:
            return True, 'No-op spin angle'

        # Spin control parameters
        max_speed = 0.3     # nominal angular speed (rad/s)
        min_speed = 0.07    # minimum slow-down speed near target (rad/s)
        slow_threshold = 0.4  # radians: start slowing when this close to target (~23°)

        # Determine spin direction (sign)
        direction = 1.0 if angle > 0 else -1.0

        twist_msg.linear.x = 0.0

        hazard_active, hazard_message = self._abort_motion_if_depth_hazard()
        if hazard_active:
            return False, hazard_message

        # Store the start yaw
        start_yaw = self.yaw
        start_time = time.monotonic()

        # Calculate the delta yaw and apply gradual slow-down to avoid overshoot
        delta_yaw = 0.0
        while abs(delta_yaw) < target_angle:
            if goal_handle.is_cancel_requested:
                self.stop()
                return False, 'Goal canceled'
            if time.monotonic() - start_time > 60.0:
                self.stop()
                return False, 'Timeout in spin'

            hazard_active, hazard_message = self._abort_motion_if_depth_hazard()
            if hazard_active:
                return False, hazard_message
            # remaining angle to rotate (radians)
            delta_yaw = self._wrap_angle(self.yaw - start_yaw)
            remaining = max(0.0, target_angle - abs(delta_yaw))

            # scale speed down as we approach the target to reduce inertia-driven overshoot
            if remaining <= slow_threshold:
                speed = max(min_speed, max_speed * (remaining / slow_threshold))
            else:
                speed = max_speed

            twist_msg.angular.z = direction * speed
            self.velocity_publisher.publish(twist_msg)

            progress = abs(delta_yaw) / target_angle if target_angle > 0 else 1.0
            self._publish_feedback(
                goal_handle,
                command_type='spin',
                target_value=target_angle,
                current_value=abs(delta_yaw),
                progress=progress,
                is_settled=False,
            )
            time.sleep(self.control_period_s)

        self.stop()
        self._wait_for_settle(goal_handle, 'spin', target_angle, target_angle)
        return True, 'ok'
        
    def stop(self):
        # Stop the robot
        self.get_logger().info('Stop')
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.velocity_publisher.publish(twist_msg)

    def save_map_point(self, point):
        # Save a map point
        if self.map_pose is not None:
            point_pose = Pose()
            point_pose = self.map_pose
            self.points[point] = point_pose
            self.get_logger().info(f'Added point "{point}": {point_pose}')
            self.save_points_to_file()  # Save to file whenever points are updated
        else:
            self.get_logger().warn('No current pose available to create map point.')

    def save_points_to_file(self):
        # Save the map points to a file
        with open('/home/ws/ugv_ws/map_points.txt', 'w') as file:
            for point_name, pose in self.points.items():
                file.write(f'{point_name}: Position(x={pose.position.x}, y={pose.position.y}, z={pose.position.z}), Orientation(x={pose.orientation.x}, y={pose.orientation.y}, z={pose.orientation.z}, w={pose.orientation.w})\n')
        self.get_logger().info('Saved points to map_points.txt')

    def pub_nav_point(self, point):
        # Publish a navigation point
        if point in self.points:
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = self.get_clock().now().to_msg()
            goal_pose.pose = self.points[point]
            self.goal_publisher.publish(goal_pose)
            self.get_logger().info(f'Sent goal to /goal_pose: {goal_pose.pose.position}')
        else:
            self.get_logger().warn(f'Point "{point}" not found in saved points.')
                    
def main(args=None):
    # Initialize the ROS 2 node
    rclpy.init(args=args)
    node = BehaviorController()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    # Spin the node
    executor.spin()
    executor.shutdown()
    node.destroy_node()
    # Shutdown the node
    rclpy.shutdown()

if __name__ == '__main__':
    main()
