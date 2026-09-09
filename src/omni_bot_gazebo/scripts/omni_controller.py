#!/usr/bin/env python3

import math
import random
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker


class OmniController(Node):
    def __init__(self, target_x, target_y, max_linear_v, wheel_noise_std):
        super().__init__('omni_controller')

        self.target_x = target_x
        self.target_y = target_y
        self.get_logger().info(f"Target set to: x={self.target_x:.2f}, y={self.target_y:.2f}")

        # NEW: user-provided max linear speed, replaces hardcoded 0.5
        self.max_v = max_linear_v
        self.get_logger().info(f"Max linear velocity set to: {self.max_v:.2f} m/s")

        # NEW: standard deviation of gaussian noise injected per wheel (rad/s)
        self.wheel_noise_std = wheel_noise_std

        # Robot geometry (must match your xacro values)
        self.wheel_radius = 0.04
        self.wheel_offset = 0.16
        self.wheel_angles = [math.radians(90), math.radians(210), math.radians(330)]

        self.kp_linear = 1.0
        self.kp_angular = 1.0
        self.goal_tolerance = 0.05

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        self.cmd_pub = self.create_publisher(Twist, '/omni_bot/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/omni_bot/odom', self.odom_callback, 10)

        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)
        self.frame_id = 'odom'

        self.moving = False
        self.timer = self.create_timer(0.1, self.control_loop)
        self.marker_timer = self.create_timer(1.0, self.publish_target_marker)

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def compute_wheel_velocities(self, vx, vy, omega):
        wheel_vels = []
        for theta in self.wheel_angles:
            v = (-math.sin(theta) * vx + math.cos(theta) * vy
                 + self.wheel_offset * omega) / self.wheel_radius
            wheel_vels.append(v)
        return wheel_vels

    # NEW: injects independent gaussian noise into each wheel's velocity,
    # simulating real-world slip, encoder error, or motor inconsistency
    def add_wheel_noise(self, wheel_vels):
        noisy_vels = [
            v + random.gauss(0.0, self.wheel_noise_std) for v in wheel_vels
        ]
        return noisy_vels

    def publish_target_marker(self):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'target'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.target_x
        marker.pose.position.y = self.target_y
        marker.pose.position.z = 0.05
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.marker_pub.publish(marker)

    def publish_vector_marker(self):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'movement_vector'
        marker.id = 1
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        start = Point(x=self.current_x, y=self.current_y, z=0.1)
        end = Point(x=self.target_x, y=self.target_y, z=0.1)
        marker.points = [start, end]

        marker.scale.x = 0.03
        marker.scale.y = 0.06
        marker.scale.z = 0.1
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.marker_pub.publish(marker)

    def control_loop(self):
        if not self.moving:
            return

        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y
        distance = math.sqrt(dx ** 2 + dy ** 2)

        self.publish_vector_marker()

        if distance < self.goal_tolerance:
            self.cmd_pub.publish(Twist())
            self.get_logger().info("Target reached. Stopping.")
            self.moving = False
            return

        vx_world = self.kp_linear * dx
        vy_world = self.kp_linear * dy

        vx = math.cos(self.current_yaw) * vx_world + math.sin(self.current_yaw) * vy_world
        vy = -math.sin(self.current_yaw) * vx_world + math.cos(self.current_yaw) * vy_world

        # CHANGED: uses user-provided max_v instead of hardcoded 0.5
        vx = max(min(vx, self.max_v), -self.max_v)
        vy = max(min(vy, self.max_v), -self.max_v)
        omega = 0.0

        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.angular.z = omega
        self.cmd_pub.publish(twist)

        wheel_vels = self.compute_wheel_velocities(vx, vy, omega)

        # NEW: apply noise before printing (this simulates what noisy
        # wheel encoders would actually report, vs the ideal computed value)
        noisy_wheel_vels = self.add_wheel_noise(wheel_vels)

        self.get_logger().info(
            f"dist={distance:.2f} | "
            f"W1={noisy_wheel_vels[0]:.3f} rad/s, "
            f"W2={noisy_wheel_vels[1]:.3f} rad/s, "
            f"W3={noisy_wheel_vels[2]:.3f} rad/s "
            f"(ideal: {wheel_vels[0]:.3f}, {wheel_vels[1]:.3f}, {wheel_vels[2]:.3f})"
        )

    def start_moving(self):
        self.moving = True


def wait_for_start(controller_node):
    while True:
        cmd = input("Type 'start' to begin moving to target: ").strip().lower()
        if cmd == 'start':
            controller_node.start_moving()
            break


def main(args=None):
    rclpy.init(args=args)

    user_input = input("Enter target x y (space separated), or press Enter for random target: ").strip()
    if user_input:
        try:
            x_str, y_str = user_input.split()
            target_x, target_y = float(x_str), float(y_str)
        except ValueError:
            print("Invalid input format, generating random target instead.")
            target_x = random.uniform(-2.0, 2.0)
            target_y = random.uniform(-2.0, 2.0)
    else:
        target_x = random.uniform(-2.0, 2.0)
        target_y = random.uniform(-2.0, 2.0)

    # NEW: ask for max linear velocity
    v_input = input("Enter max linear velocity in m/s (press Enter for default 0.5): ").strip()
    if v_input:
        try:
            max_linear_v = float(v_input)
        except ValueError:
            print("Invalid input, using default 0.5 m/s.")
            max_linear_v = 0.5
    else:
        max_linear_v = 0.5

    # NEW: ask for wheel noise standard deviation
    noise_input = input("Enter wheel noise std dev in rad/s (press Enter for default 0.05): ").strip()
    if noise_input:
        try:
            wheel_noise_std = float(noise_input)
        except ValueError:
            print("Invalid input, using default 0.05 rad/s.")
            wheel_noise_std = 0.05
    else:
        wheel_noise_std = 0.05

    node = OmniController(target_x, target_y, max_linear_v, wheel_noise_std)

    input_thread = threading.Thread(target=wait_for_start, args=(node,), daemon=True)
    input_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()