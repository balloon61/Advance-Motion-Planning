#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from motion_planning.configuration_space import *
from motion_planning.func_test import *
class TestNode(Node):
    def __init__(self):
        super().__init__("main")
        self.get_logger().info("Hello moveit")
        convex_decomp()


def main(args=None):
    rclpy.init(args=args)
    node = TestNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()