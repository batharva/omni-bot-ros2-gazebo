#!/usr/bin/env python3

import math
import os
import random
import threading
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist, Point, Pose, TransformStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from gazebo_msgs.srv import SpawnEntity
from tf2_ros import TransformBroadcaster

import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation


HISTORY_LEN = 200


class OmniController(Node):
    def __init__(self):
        super().__init__('omni_controller')

        self.pole_height = 0.235
        self.pole_sdf_path = os.path.join(
            get_package_share_directory('omni_bot_gazebo'),
            'models', 'target_pole', 'model.sdf'
        )

        self.wheel_radius = 0.03
        self.wheel_offset = 0.105
        self.wheel_angles = [math.radians(90), math.radians(210), math.radians(330)]
        self.max_wheel_speed = 20.0

        self.J = self._build_jacobian()
        self._check_controllability()
        self.J_pinv = np.linalg.pinv(self.J)

        self.free_slide_dirs = [
            np.array([math.cos(t), math.sin(t)]) for t in self.wheel_angles
        ]

        self.kp_linear = 1.0
        self.goal_tolerance = 0.05

        self.target_x = 0.0
        self.target_y = 0.0
        self.max_v = 0.5

        # --- Point A of the straight A->B reference line (set when motion starts) ---
        self.start_x = 0.0
        self.start_y = 0.0
        self.line_yaw = 0.0   # heading of the straight line from A to B
        self.yaw_error = 0.0  # current_yaw vs line_yaw, kept for the GUI

        # Heading controller: steers the robot's facing vector to track the
        # A->B line direction, independent of the holonomic vx/vy position
        # control (which handles any sideways correction from noise/drift).
        self.kp_yaw = 1.5
        self.max_omega = 2.0

        # --- Per-wheel noise (each wheel now has its own independent std) ---
        self.wheel_noise_std = [0.05, 0.05, 0.05]

        self.moving = False
        self.target_set = False
        self.pole_spawned = False
        self.mission_complete = False

        self._actuation_limit_active = False

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # --- Wheel angle (theta) accumulators: integrated from velocity ---
        # theta_ideal = the "original track" (what the wheel should have rotated)
        # theta_noisy = what the wheel actually rotated given the injected noise
        self.wheel_theta_ideal = [0.0, 0.0, 0.0]
        self.wheel_theta_noisy = [0.0, 0.0, 0.0]

        self.control_dt = 0.1  # matches the control-loop timer period below

        # --- Robot-level combined noise drift ---
        # Isolates just the noise contribution per wheel (noisy - ideal),
        # maps it back to body-frame (vx, vy, omega) via the inverse
        # Jacobian, and integrates it. This is the "total" noise effect on
        # the robot as a whole (not sent to Gazebo — it's a monitoring
        # overlay only; the clean twist is still what actually drives the sim).
        self.robot_noise_drift_x = 0.0
        self.robot_noise_drift_y = 0.0
        self.robot_noise_drift_yaw = 0.0
        self.robot_noise_history = deque(maxlen=HISTORY_LEN)

        self.history = deque(maxlen=HISTORY_LEN)

        self.path_x = deque(maxlen=2000)
        self.path_y = deque(maxlen=2000)
        self._t0 = self.get_clock().now().nanoseconds / 1e9

        self.cmd_pub = self.create_publisher(Twist, '/omni_bot/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/omni_bot/odom', self.odom_callback, 10)

        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)
        self.frame_id = 'odom'

        # --- RViz2 visualization: TF frames, actual path, alignment markers ---
        # Frames are named *_debug_frame so they never collide with whatever
        # base_link/odom TF your robot's own URDF or odom plugin publishes.
        self.tf_broadcaster = TransformBroadcaster(self)
        self.rviz_marker_pub = self.create_publisher(MarkerArray, '/omni_bot/viz_markers', 10)
        self.path_pub = self.create_publisher(Path, '/omni_bot/actual_path', 10)
        self.actual_path_msg = Path()
        self.actual_path_msg.header.frame_id = self.frame_id
        self.rviz_timer = self.create_timer(0.1, self.publish_rviz_visuals)

        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')

        self.timer = self.create_timer(self.control_dt, self.control_loop)
        self.marker_timer = self.create_timer(1.0, self.publish_target_marker)

    def set_target(self, x, y):
        self.target_x = x
        self.target_y = y
        self.target_set = True
        self.mission_complete = False
        self.get_logger().info(f"Target set to: x={x:.2f}, y={y:.2f}")
        self.spawn_target_pole()

    def set_max_velocity(self, v):
        self.max_v = v
        self.get_logger().info(f"Max linear velocity set to: {v:.2f} m/s")

    def set_wheel_noise_std(self, stds):
        """stds: list/tuple of 3 floats, one noise std (rad/s) per wheel."""
        self.wheel_noise_std = list(stds)
        self.get_logger().info(
            f"Per-wheel noise std set to: "
            f"[{stds[0]:.3f}, {stds[1]:.3f}, {stds[2]:.3f}] rad/s"
        )

    def start_moving(self):
        if not self.target_set:
            self.get_logger().warn("Cannot start — no target set yet.")
            return
        self.mission_complete = False
        self.moving = True
        # Point A of the straight reference line is wherever the robot is
        # right now, at the moment it starts moving toward the target (B).
        self.start_x = self.current_x
        self.start_y = self.current_y
        self.line_yaw = math.atan2(self.target_y - self.start_y, self.target_x - self.start_x)
        self.get_logger().info(
            f"Started moving toward target. Line A->B established: "
            f"A=({self.start_x:.2f},{self.start_y:.2f}) heading={math.degrees(self.line_yaw):.1f}°"
        )

    def spawn_target_pole(self):
        self.get_logger().info("Waiting for /spawn_entity service...")
        if not self.spawn_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("/spawn_entity service not available — pole not spawned.")
            return

        try:
            with open(self.pole_sdf_path, 'r') as f:
                sdf = f.read()
        except FileNotFoundError:
            self.get_logger().error(f"Pole SDF not found at {self.pole_sdf_path}")
            return

        pose = Pose()
        pose.position.x = self.target_x
        pose.position.y = self.target_y
        pose.position.z = self.pole_height / 2.0
        pose.orientation.w = 1.0

        request = SpawnEntity.Request()
        request.name = 'target_pole'
        request.xml = sdf
        request.robot_namespace = ''
        request.initial_pose = pose
        request.reference_frame = 'world'

        future = self.spawn_client.call_async(request)
        future.add_done_callback(self._spawn_done_callback)

    def _spawn_done_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(
                    f"Pole spawned at target ({self.target_x:.2f}, {self.target_y:.2f})."
                )
                self.pole_spawned = True
            else:
                self.get_logger().error(f"Pole spawn failed: {response.status_message}")
        except Exception as e:
            self.get_logger().error(f"Pole spawn service call failed: {e}")

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def _build_jacobian(self):
        rows = []
        for theta in self.wheel_angles:
            rows.append([
                -math.sin(theta) / self.wheel_radius,
                math.cos(theta) / self.wheel_radius,
                self.wheel_offset / self.wheel_radius
            ])
        return np.array(rows)

    def _check_controllability(self):
        rank = np.linalg.matrix_rank(self.J)
        cond = np.linalg.cond(self.J)
        self.get_logger().info(
            f"Controllability check: Jacobian rank={rank}/3, condition number={cond:.2f}"
        )
        if rank < 3:
            self.get_logger().error(
                "CONTROLLABILITY LOST: Jacobian rank < 3 — wheel geometry cannot "
                "independently command all of vx, vy, omega. Check wheel_angles."
            )
        elif cond > 10.0:
            self.get_logger().warn(
                "Jacobian is poorly conditioned — some motion directions will "
                "require disproportionately high wheel speeds."
            )

    def compute_wheel_velocities(self, vx, vy, omega):
        body_vel = np.array([vx, vy, omega])
        return (self.J @ body_vel).tolist()

    def recover_body_velocity(self, wheel_vels):
        return self.J_pinv @ np.array(wheel_vels)

    def compute_free_slide_components(self, vx, vy):
        v_body = np.array([vx, vy])
        return [float(np.dot(v_body, d)) for d in self.free_slide_dirs]

    @staticmethod
    def _normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _yaw_to_quaternion(yaw):
        return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))

    def enforce_actuation_limits(self, vx, vy, omega):
        wheel_vels = self.compute_wheel_velocities(vx, vy, omega)
        max_required = max(abs(w) for w in wheel_vels)

        if max_required > self.max_wheel_speed:
            scale = self.max_wheel_speed / max_required
            vx *= scale
            vy *= scale
            omega *= scale
            if not self._actuation_limit_active:
                self.get_logger().warn(
                    f"Actuation limit exceeded (required {max_required:.2f} rad/s, "
                    f"max {self.max_wheel_speed:.2f} rad/s) — scaling commands down "
                    f"until this clears. Consider lowering Max Velocity in the GUI."
                )
                self._actuation_limit_active = True
        else:
            if self._actuation_limit_active:
                self.get_logger().info("Actuation limit no longer exceeded — full speed restored.")
                self._actuation_limit_active = False

        return vx, vy, omega

    def add_wheel_noise(self, wheel_vels):
        # Each wheel draws from its OWN std now, instead of one shared std.
        return [
            v + random.gauss(0.0, self.wheel_noise_std[i])
            for i, v in enumerate(wheel_vels)
        ]

    def publish_target_marker(self):
        if not self.target_set:
            return

        if self.mission_complete:
            r, g, b = 0.0, 1.0, 0.0
        else:
            r, g, b = 1.0, 0.0, 0.0

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
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0
        self.marker_pub.publish(marker)

        circle = Marker()
        circle.header.frame_id = self.frame_id
        circle.header.stamp = self.get_clock().now().to_msg()
        circle.ns = 'target'
        circle.id = 1
        circle.type = Marker.LINE_STRIP
        circle.action = Marker.ADD
        circle.pose.orientation.w = 1.0
        circle.scale.x = 0.02
        circle.color.r = r
        circle.color.g = g
        circle.color.b = b
        circle.color.a = 0.9

        num_pts = 36
        for i in range(num_pts + 1):
            angle = 2.0 * math.pi * i / num_pts
            p = Point()
            p.x = self.target_x + self.goal_tolerance * math.cos(angle)
            p.y = self.target_y + self.goal_tolerance * math.sin(angle)
            p.z = 0.02
            circle.points.append(p)

        self.marker_pub.publish(circle)

    def publish_rviz_visuals(self):
        now = self.get_clock().now().to_msg()

        # --- Bot's actual body frame (deviated by noise, drives from odom) ---
        qx, qy, qz, qw = self._yaw_to_quaternion(self.current_yaw)
        bot_tf = TransformStamped()
        bot_tf.header.stamp = now
        bot_tf.header.frame_id = self.frame_id
        bot_tf.child_frame_id = 'bot_debug_frame'
        bot_tf.transform.translation.x = self.current_x
        bot_tf.transform.translation.y = self.current_y
        bot_tf.transform.translation.z = 0.0
        bot_tf.transform.rotation.x = qx
        bot_tf.transform.rotation.y = qy
        bot_tf.transform.rotation.z = qz
        bot_tf.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(bot_tf)

        # --- Actual traveled path (shows the noise-induced deviation over time) ---
        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = self.current_x
        pose.pose.position.y = self.current_y
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        self.actual_path_msg.header.stamp = now
        self.actual_path_msg.poses.append(pose)
        if len(self.actual_path_msg.poses) > 2000:
            self.actual_path_msg.poses.pop(0)
        self.path_pub.publish(self.actual_path_msg)

        marker_array = MarkerArray()

        if self.target_set:
            # --- Target's body frame: aligned with the A->B line direction ---
            tqx, tqy, tqz, tqw = self._yaw_to_quaternion(self.line_yaw)
            target_tf = TransformStamped()
            target_tf.header.stamp = now
            target_tf.header.frame_id = self.frame_id
            target_tf.child_frame_id = 'target_debug_frame'
            target_tf.transform.translation.x = self.target_x
            target_tf.transform.translation.y = self.target_y
            target_tf.transform.translation.z = 0.0
            target_tf.transform.rotation.x = tqx
            target_tf.transform.rotation.y = tqy
            target_tf.transform.rotation.z = tqz
            target_tf.transform.rotation.w = tqw
            self.tf_broadcaster.sendTransform(target_tf)

            # --- Start point A frame (also aligned with the line) ---
            start_tf = TransformStamped()
            start_tf.header.stamp = now
            start_tf.header.frame_id = self.frame_id
            start_tf.child_frame_id = 'start_debug_frame'
            start_tf.transform.translation.x = self.start_x
            start_tf.transform.translation.y = self.start_y
            start_tf.transform.translation.z = 0.0
            start_tf.transform.rotation.x = tqx
            start_tf.transform.rotation.y = tqy
            start_tf.transform.rotation.z = tqz
            start_tf.transform.rotation.w = tqw
            self.tf_broadcaster.sendTransform(start_tf)

            # --- A -> B straight reference line ---
            line_marker = Marker()
            line_marker.header.frame_id = self.frame_id
            line_marker.header.stamp = now
            line_marker.ns = 'alignment'
            line_marker.id = 0
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.02
            line_marker.color.r = 1.0
            line_marker.color.g = 1.0
            line_marker.color.b = 0.0
            line_marker.color.a = 0.9
            line_marker.pose.orientation.w = 1.0
            p_a, p_b = Point(), Point()
            p_a.x, p_a.y = self.start_x, self.start_y
            p_b.x, p_b.y = self.target_x, self.target_y
            line_marker.points = [p_a, p_b]
            marker_array.markers.append(line_marker)

            # --- Perpendicular deviation connector: bot -> nearest point on the line ---
            ab = np.array([self.target_x - self.start_x, self.target_y - self.start_y])
            ab_len_sq = float(np.dot(ab, ab))
            if ab_len_sq > 1e-9:
                tparam = float(np.clip(
                    np.dot([self.current_x - self.start_x, self.current_y - self.start_y], ab) / ab_len_sq,
                    0.0, 1.0
                ))
                nearest = np.array([self.start_x, self.start_y]) + tparam * ab
            else:
                nearest = np.array([self.start_x, self.start_y])
            perp_dev = math.hypot(self.current_x - nearest[0], self.current_y - nearest[1])

            dev_marker = Marker()
            dev_marker.header.frame_id = self.frame_id
            dev_marker.header.stamp = now
            dev_marker.ns = 'alignment'
            dev_marker.id = 1
            dev_marker.type = Marker.LINE_STRIP
            dev_marker.action = Marker.ADD
            dev_marker.scale.x = 0.015
            dev_marker.color.r = 1.0
            dev_marker.color.g = 0.0
            dev_marker.color.b = 0.0
            dev_marker.color.a = 0.9
            dev_marker.pose.orientation.w = 1.0
            p_bot, p_near = Point(), Point()
            p_bot.x, p_bot.y = self.current_x, self.current_y
            p_near.x, p_near.y = float(nearest[0]), float(nearest[1])
            dev_marker.points = [p_bot, p_near]
            marker_array.markers.append(dev_marker)

            # --- Text readout: heading error + line deviation ---
            text_marker = Marker()
            text_marker.header.frame_id = self.frame_id
            text_marker.header.stamp = now
            text_marker.ns = 'alignment'
            text_marker.id = 2
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = self.current_x
            text_marker.pose.position.y = self.current_y
            text_marker.pose.position.z = 0.3
            text_marker.scale.z = 0.15
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = (
                f"heading err: {math.degrees(self.yaw_error):+.1f} deg\n"
                f"line dev: {perp_dev:.3f} m"
            )
            marker_array.markers.append(text_marker)

        self.rviz_marker_pub.publish(marker_array)

    def control_loop(self):
        self.path_x.append(self.current_x)
        self.path_y.append(self.current_y)

        if not self.moving:
            return

        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance < self.goal_tolerance:
            self.cmd_pub.publish(Twist())
            self.moving = False
            if not self.mission_complete:
                self.mission_complete = True
                self.get_logger().info(
                    f"MISSION COMPLETE — target reached within confidence "
                    f"circle (radius={self.goal_tolerance:.2f} m). Bot stopped."
                )
            return

        vx_world = self.kp_linear * dx
        vy_world = self.kp_linear * dy

        vx = math.cos(self.current_yaw) * vx_world + math.sin(self.current_yaw) * vy_world
        vy = -math.sin(self.current_yaw) * vx_world + math.cos(self.current_yaw) * vy_world

        vx = max(min(vx, self.max_v), -self.max_v)
        vy = max(min(vy, self.max_v), -self.max_v)

        # Heading control: always try to keep the robot's facing vector
        # aligned with the straight A->B line, regardless of how far the
        # noise has pushed the robot off that line. Position correction
        # (vx, vy above) and heading correction (omega here) are independent
        # thanks to the holonomic drive.
        self.yaw_error = self._normalize_angle(self.line_yaw - self.current_yaw)
        omega = max(min(self.kp_yaw * self.yaw_error, self.max_omega), -self.max_omega)

        vx, vy, omega = self.enforce_actuation_limits(vx, vy, omega)

        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.angular.z = omega
        self.cmd_pub.publish(twist)

        wheel_vels = self.compute_wheel_velocities(vx, vy, omega)
        noisy_vels = self.add_wheel_noise(wheel_vels)

        # Integrate velocity -> angle for each wheel (theta += omega * dt).
        # theta_ideal is the "original track" the wheel was commanded to follow;
        # theta_noisy is where the wheel actually ends up given the noise.
        # The growing gap between them IS the deviation.
        for i in range(3):
            self.wheel_theta_ideal[i] += wheel_vels[i] * self.control_dt
            self.wheel_theta_noisy[i] += noisy_vels[i] * self.control_dt

        # Combine the three wheels' noise into one robot-level effect: take
        # just the noise part of each wheel's velocity, map it back to
        # body-frame (vx, vy, omega) through the inverse Jacobian, and
        # integrate that into a drift trajectory.
        noise_only_wheel_vels = [noisy_vels[i] - wheel_vels[i] for i in range(3)]
        body_noise = self.recover_body_velocity(noise_only_wheel_vels)
        self.robot_noise_drift_x += float(body_noise[0]) * self.control_dt
        self.robot_noise_drift_y += float(body_noise[1]) * self.control_dt
        self.robot_noise_drift_yaw += float(body_noise[2]) * self.control_dt
        noise_magnitude = math.hypot(self.robot_noise_drift_x, self.robot_noise_drift_y)

        t = self.get_clock().now().nanoseconds / 1e9 - self._t0

        self.history.append((
            t,
            self.wheel_theta_ideal[0], self.wheel_theta_ideal[1], self.wheel_theta_ideal[2],
            self.wheel_theta_noisy[0], self.wheel_theta_noisy[1], self.wheel_theta_noisy[2],
        ))

        self.robot_noise_history.append((
            t,
            self.robot_noise_drift_x, self.robot_noise_drift_y,
            self.robot_noise_drift_yaw, noise_magnitude,
        ))


def ros_spin_thread(node):
    rclpy.spin(node)


class OmniGuiApp:
    def __init__(self, root, node: OmniController):
        self.node = node
        self.root = root
        root.title("Omni Bot Controller & Live Monitor")

        input_frame = ttk.Frame(root, padding=10)
        input_frame.grid(row=0, column=0, sticky="ew")

        ttk.Label(input_frame, text="Target X:").grid(row=0, column=0)
        self.target_x_entry = ttk.Entry(input_frame, width=8)
        self.target_x_entry.grid(row=0, column=1)

        ttk.Label(input_frame, text="Target Y:").grid(row=0, column=2)
        self.target_y_entry = ttk.Entry(input_frame, width=8)
        self.target_y_entry.grid(row=0, column=3)

        ttk.Button(input_frame, text="Random Target", command=self.random_target).grid(row=0, column=4, padx=5)
        ttk.Button(input_frame, text="Set Target", command=self.set_target).grid(row=0, column=5, padx=5)

        ttk.Label(input_frame, text="Max Velocity (m/s):").grid(row=1, column=0)
        self.velocity_entry = ttk.Entry(input_frame, width=8)
        self.velocity_entry.insert(0, "0.5")
        self.velocity_entry.grid(row=1, column=1)

        # --- Per-wheel noise entries (was a single shared entry before) ---
        ttk.Label(input_frame, text="Noise std W1/W2/W3 (rad/s):").grid(row=1, column=2)
        noise_frame = ttk.Frame(input_frame)
        noise_frame.grid(row=1, column=3, columnspan=2, sticky="w")

        self.noise_entry_1 = ttk.Entry(noise_frame, width=5)
        self.noise_entry_1.insert(0, "0.05")
        self.noise_entry_1.pack(side="left", padx=2)

        self.noise_entry_2 = ttk.Entry(noise_frame, width=5)
        self.noise_entry_2.insert(0, "0.05")
        self.noise_entry_2.pack(side="left", padx=2)

        self.noise_entry_3 = ttk.Entry(noise_frame, width=5)
        self.noise_entry_3.insert(0, "0.05")
        self.noise_entry_3.pack(side="left", padx=2)

        ttk.Button(input_frame, text="Apply", command=self.apply_settings).grid(row=1, column=5, padx=5)
        self.start_button = ttk.Button(input_frame, text="START", command=self.start)
        self.start_button.grid(row=2, column=5, padx=5)

        self.status_label = ttk.Label(input_frame, text="Status: waiting for target...")
        self.status_label.grid(row=3, column=0, columnspan=6, sticky="w", pady=(5, 0))

        self.fig = Figure(figsize=(14, 11))
        self.ax_2d = self.fig.add_subplot(3, 3, 1)
        self.ax_w1 = self.fig.add_subplot(3, 3, 2)
        self.ax_w2 = self.fig.add_subplot(3, 3, 3)
        self.ax_w3 = self.fig.add_subplot(3, 3, 4)
        self.ax_robot_noise = self.fig.add_subplot(3, 3, 5)
        self.ax_robot_noise_mag = self.fig.add_subplot(3, 3, 6)
        self.ax_alignment = self.fig.add_subplot(3, 3, 7)

        canvas_frame = ttk.Frame(root)
        canvas_frame.grid(row=1, column=0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack()

        self._last_mission_complete = False

        self.anim = FuncAnimation(self.fig, self.update_plots, interval=200, cache_frame_data=False)

    def random_target(self):
        x = round(random.uniform(-2.0, 2.0), 2)
        y = round(random.uniform(-2.0, 2.0), 2)
        self.target_x_entry.delete(0, tk.END)
        self.target_x_entry.insert(0, str(x))
        self.target_y_entry.delete(0, tk.END)
        self.target_y_entry.insert(0, str(y))

    def set_target(self):
        try:
            x = float(self.target_x_entry.get())
            y = float(self.target_y_entry.get())
            self.node.set_target(x, y)
            self._last_mission_complete = False
            self.status_label.config(text=f"Status: target set to ({x:.2f}, {y:.2f})")
        except ValueError:
            self.status_label.config(text="Status: invalid target values")

    def apply_settings(self):
        try:
            self.node.set_max_velocity(float(self.velocity_entry.get()))
        except ValueError:
            pass
        try:
            stds = [
                float(self.noise_entry_1.get()),
                float(self.noise_entry_2.get()),
                float(self.noise_entry_3.get()),
            ]
            self.node.set_wheel_noise_std(stds)
        except ValueError:
            pass
        self.status_label.config(text="Status: settings applied")

    def start(self):
        self.node.start_moving()
        self._last_mission_complete = False
        self.status_label.config(text="Status: moving toward target...")

    def update_plots(self, frame):
        self.ax_2d.clear()
        self.ax_2d.set_xlim(-3, 3)
        self.ax_2d.set_ylim(-3, 3)
        self.ax_2d.set_aspect('equal')
        self.ax_2d.grid(True)

        if self.node.mission_complete:
            self.ax_2d.set_title("2D Scenario — MISSION COMPLETE", color='green', fontweight='bold')
        else:
            self.ax_2d.set_title("2D Scenario")

        if self.node.path_x:
            self.ax_2d.plot(list(self.node.path_x), list(self.node.path_y), 'b-', linewidth=1, label='Path')

        self.ax_2d.plot(self.node.current_x, self.node.current_y, 'ko', markersize=8, label='Robot')
        arrow_len = 0.2
        self.ax_2d.arrow(
            self.node.current_x, self.node.current_y,
            arrow_len * math.cos(self.node.current_yaw),
            arrow_len * math.sin(self.node.current_yaw),
            head_width=0.08, color='black'
        )

        if self.node.target_set:
            target_color = 'g' if self.node.mission_complete else 'r'
            target_marker = 'o' if self.node.mission_complete else 'x'
            self.ax_2d.plot(
                self.node.target_x, self.node.target_y,
                target_marker, color=target_color, markersize=12, mew=3, label='Target'
            )
            conf_circle = Circle(
                (self.node.target_x, self.node.target_y),
                self.node.goal_tolerance,
                fill=False, linestyle='--', linewidth=1.5, edgecolor=target_color
            )
            self.ax_2d.add_patch(conf_circle)

        self.ax_2d.legend(loc='upper right', fontsize=7)

        if self.node.mission_complete and not self._last_mission_complete:
            self.status_label.config(text="Status: ✅ MISSION COMPLETE — target reached, bot fully stopped.")
            self.start_button.config(text="MISSION COMPLETE", state="disabled")
        elif not self.node.mission_complete and self._last_mission_complete:
            self.start_button.config(text="START", state="normal")
        self._last_mission_complete = self.node.mission_complete

        hist_snapshot = list(self.node.history)

        if hist_snapshot:
            t_vals, th1i, th2i, th3i, th1n, th2n, th3n = zip(*hist_snapshot)
        else:
            t_vals = th1i = th2i = th3i = th1n = th2n = th3n = []

        # Plot theta (wheel angle) instead of omega (wheel velocity).
        # "Ideal" = the original commanded track; "Noisy" = actual rotation
        # given the per-wheel noise. The gap between the two curves is the
        # deviation, and its current magnitude is shown in the title.
        wheel_data = [(th1i, th1n), (th2i, th2n), (th3i, th3n)]
        for i, (ax, (ideal, noisy)) in enumerate(zip([self.ax_w1, self.ax_w2, self.ax_w3], wheel_data)):
            ax.clear()
            dev = (noisy[-1] - ideal[-1]) if len(ideal) else 0.0
            ax.set_title(f"Wheel {i+1} θ (rad) — Δ={dev:+.3f} rad")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("theta (rad)")
            ax.grid(True)
            if t_vals:
                ax.plot(t_vals, ideal, 'b-', label='Ideal track')
                ax.plot(t_vals, noisy, 'r-', alpha=0.7, label='Actual (noisy)')
                ax.fill_between(t_vals, ideal, noisy, color='r', alpha=0.15)
                ax.legend(loc='upper left', fontsize=7)

        # --- Robot-level combined noise (all 3 wheels merged via inverse Jacobian) ---
        robot_hist = list(self.node.robot_noise_history)
        if robot_hist:
            rt, rdx, rdy, rdyaw, rmag = zip(*robot_hist)
        else:
            rt = rdx = rdy = rdyaw = rmag = []

        self.ax_robot_noise.clear()
        self.ax_robot_noise.set_title(
            f"Robot noise drift (body-frame) — yaw Δ={rdyaw[-1]:+.3f} rad" if rdyaw else
            "Robot noise drift (body-frame)"
        )
        self.ax_robot_noise.set_xlabel("drift x (m)")
        self.ax_robot_noise.set_ylabel("drift y (m)")
        self.ax_robot_noise.grid(True)
        self.ax_robot_noise.set_aspect('equal')
        if rdx:
            self.ax_robot_noise.plot(rdx, rdy, 'm-', linewidth=1, label='Drift path')
            self.ax_robot_noise.plot(rdx[-1], rdy[-1], 'mo', markersize=6)
            self.ax_robot_noise.plot(0, 0, 'k+', markersize=8, label='Start (no noise)')
            self.ax_robot_noise.legend(loc='upper right', fontsize=7)

        self.ax_robot_noise_mag.clear()
        self.ax_robot_noise_mag.set_title("Total robot noise magnitude (m)")
        self.ax_robot_noise_mag.set_xlabel("time (s)")
        self.ax_robot_noise_mag.set_ylabel("|drift| (m)")
        self.ax_robot_noise_mag.grid(True)
        if rt:
            self.ax_robot_noise_mag.plot(rt, rmag, 'm-', linewidth=1.2)

        # --- Body-frame alignment view ---
        # Straight reference line A->B, the bot's actual body frame (showing
        # how noise has knocked it off the line), and the target's body
        # frame (defined as facing along the line, i.e. where the bot's
        # heading vector is trying to align to).
        self.ax_alignment.clear()
        self.ax_alignment.set_aspect('equal')
        self.ax_alignment.grid(True)

        if self.node.target_set:
            ax_len, ay_len = 0.35, 0.35
            sx, sy = self.node.start_x, self.node.start_y
            tx, ty = self.node.target_x, self.node.target_y
            rx, ry = self.node.current_x, self.node.current_y
            yaw = self.node.current_yaw
            line_yaw = self.node.line_yaw

            # A -> B straight reference line
            self.ax_alignment.plot([sx, tx], [sy, ty], 'k--', linewidth=1, label='A→B line')
            self.ax_alignment.plot(sx, sy, 's', color='gray', markersize=8, label='A (start)')
            self.ax_alignment.plot(tx, ty, '^', color='green', markersize=9, label='B (target)')

            # Perpendicular deviation of the robot from the A->B line
            ab = np.array([tx - sx, ty - sy])
            ab_len_sq = float(np.dot(ab, ab))
            if ab_len_sq > 1e-9:
                tparam = float(np.clip(np.dot([rx - sx, ry - sy], ab) / ab_len_sq, 0.0, 1.0))
                nearest = np.array([sx, sy]) + tparam * ab
                self.ax_alignment.plot(
                    [rx, nearest[0]], [ry, nearest[1]], 'r:', linewidth=1, label='Deviation'
                )
            else:
                nearest = np.array([sx, sy])
            perp_dev = math.hypot(rx - nearest[0], ry - nearest[1])

            # Bot's actual body frame: x-axis (heading, red), y-axis (blue)
            self.ax_alignment.plot(rx, ry, 'ko', markersize=7)
            self.ax_alignment.arrow(rx, ry, ax_len * math.cos(yaw), ax_len * math.sin(yaw),
                                     head_width=0.06, color='red', label='Bot heading (x)')
            self.ax_alignment.arrow(rx, ry, ay_len * math.cos(yaw + math.pi / 2), ay_len * math.sin(yaw + math.pi / 2),
                                     head_width=0.06, color='blue', alpha=0.7)

            # Desired heading (line direction) drawn dashed at the bot's position
            self.ax_alignment.arrow(rx, ry, ax_len * math.cos(line_yaw), ax_len * math.sin(line_yaw),
                                     head_width=0.05, color='orange', linestyle='dashed', alpha=0.8,
                                     label='Desired (line) heading')

            # Target's body frame: aligned with the line direction — this is
            # the orientation the bot's heading vector is converging toward.
            self.ax_alignment.arrow(tx, ty, ax_len * math.cos(line_yaw), ax_len * math.sin(line_yaw),
                                     head_width=0.06, color='green', alpha=0.7)
            self.ax_alignment.arrow(tx, ty, ay_len * math.cos(line_yaw + math.pi / 2), ay_len * math.sin(line_yaw + math.pi / 2),
                                     head_width=0.06, color='cyan', alpha=0.7)

            yaw_err_deg = math.degrees(self.node.yaw_error)
            self.ax_alignment.set_title(
                f"Bot vs target frame — heading err={yaw_err_deg:+.1f}°, line dev={perp_dev:.3f} m"
            )
            self.ax_alignment.legend(loc='best', fontsize=6)
        else:
            self.ax_alignment.set_title("Bot vs target frame (no target set)")

        return []


def main(args=None):
    rclpy.init(args=args)
    node = OmniController()

    spin_thread = threading.Thread(target=ros_spin_thread, args=(node,), daemon=True)
    spin_thread.start()

    root = tk.Tk()
    app = OmniGuiApp(root, node)
    root.mainloop()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
