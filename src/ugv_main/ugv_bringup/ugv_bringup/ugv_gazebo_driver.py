#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


class UgvGazeboDriver(Node):
    """Gazebo equivalent of ugv_driver.py for the pan-tilt camera.

    Subscribes to ugv/joint_states (same topic/format as the real driver),
    extracts the two pan-tilt joint angles, and publishes them to Gazebo
    via the libgazebo_ros_joint_pose_trajectory plugin (pt_joint_trajectory).

    A 10 Hz timer re-sends the last commanded position to actively hold the
    joints rather than relying on passive damping alone.
    """

    def __init__(self, name):
        super().__init__(name)

        self.joint_states_sub = self.create_subscription(
            JointState, 'ugv/joint_states', self.joint_states_callback, 10
        )

        self.traj_pub = self.create_publisher(
            JointTrajectory, 'pt_joint_trajectory', 10
        )

        self.target_pan = 0.0
        self.target_tilt = 0.0

        self.create_timer(1.0, self.hold_position)

    def joint_states_callback(self, msg):
        if msg.header.frame_id != 'ugv_joint_state':
            return

        try:
            self.target_pan = msg.position[msg.name.index('pt_base_link_to_pt_link1')]
            self.target_tilt = msg.position[msg.name.index('pt_link1_to_pt_link2')]
        except ValueError:
            self.get_logger().warn('pt joint names not found in JointState message')
            return

        self._publish_trajectory()

    def hold_position(self):
        self._publish_trajectory()

    def _publish_trajectory(self):
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.header.frame_id = 'world'
        traj.joint_names = ['pt_base_link_to_pt_link1', 'pt_link1_to_pt_link2']

        point = JointTrajectoryPoint()
        point.positions = [self.target_pan, self.target_tilt]
        point.time_from_start = Duration(sec=0, nanosec=0)  # execute immediately

        traj.points = [point]
        self.traj_pub.publish(traj)


def main(args=None):
    rclpy.init(args=args)
    node = UgvGazeboDriver('ugv_gazebo_driver')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
