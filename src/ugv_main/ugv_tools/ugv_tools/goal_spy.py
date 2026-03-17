import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path


class GoalSpy(Node):
    def __init__(self):
        super().__init__('goal_spy')
        self.create_subscription(Path, '/plan', self.cb, 10)
        self.get_logger().info('GoalSpy ready — set a Nav2 goal in RViz to capture it.')

    def cb(self, msg):
        if msg.poses:
            goal = msg.poses[-1].pose
            self.get_logger().info(
                f'Goal ->'
                f'\n  position:    x: {goal.position.x:.4f},  y: {goal.position.y:.4f},  z: {goal.position.z:.4f}'
                f'\n  orientation: x: {goal.orientation.x:.6f},  y: {goal.orientation.y:.6f},  z: {goal.orientation.z:.6f},  w: {goal.orientation.w:.6f}'
            )


def main():
    rclpy.init()
    node = GoalSpy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
