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
from geometry_msgs.msg import Twist, Point, Pose
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from gazebo_msgs.srv import SpawnEntity

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
        self.wheel_noise_std = 0.05
        self.moving = False
        self.target_set = False
        self.pole_spawned = False
        self.mission_complete = False

        self._actuation_limit_active = False

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        self.history = deque(maxlen=HISTORY_LEN)

        self.path_x = deque(maxlen=2000)
        self.path_y = deque(maxlen=2000)
        self._t0 = self.get_clock().now().nanoseconds / 1e9

        self.cmd_pub = self.create_publisher(Twist, '/omni_bot/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/omni_bot/odom', self.odom_callback, 10)

        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)
        self.frame_id = 'odom'

        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')

        self.timer = self.create_timer(0.1, self.control_loop)
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

    def set_noise_std(self, std):
        self.wheel_noise_std = std
        self.get_logger().info(f"Wheel noise std set to: {std:.3f} rad/s")

    def start_moving(self):
        if not self.target_set:
            self.get_logger().warn("Cannot start — no target set yet.")
            return
        self.mission_complete = False
        self.moving = True
        self.get_logger().info("Started moving toward target.")

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
        return [v + random.gauss(0.0, self.wheel_noise_std) for v in wheel_vels]

    def publish_target_marker(self):
        if not self.target_set:
            return

        # Target marker turns green once the mission is complete.
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

        # Confidence circle: a ring showing the goal_tolerance radius around
        # the target. The bot is considered to have arrived (and fully stops)
        # the moment it crosses into this ring — see control_loop().
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

    def control_loop(self):
        self.path_x.append(self.current_x)
        self.path_y.append(self.current_y)

        if not self.moving:
            return

        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance < self.goal_tolerance:
            # Bot has entered the target's confidence circle — command a
            # full stop (zero linear/angular velocity) and latch the
            # mission-complete state so it doesn't move again on its own.
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
        omega = 0.0

        vx, vy, omega = self.enforce_actuation_limits(vx, vy, omega)

        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.angular.z = omega
        self.cmd_pub.publish(twist)

        wheel_vels = self.compute_wheel_velocities(vx, vy, omega)
        noisy_vels = self.add_wheel_noise(wheel_vels)

        t = self.get_clock().now().nanoseconds / 1e9 - self._t0

        self.history.append((
            t,
            wheel_vels[0], wheel_vels[1], wheel_vels[2],
            noisy_vels[0], noisy_vels[1], noisy_vels[2]
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

        ttk.Label(input_frame, text="Wheel Noise Std (rad/s):").grid(row=1, column=2)
        self.noise_entry = ttk.Entry(input_frame, width=8)
        self.noise_entry.insert(0, "0.05")
        self.noise_entry.grid(row=1, column=3)

        ttk.Button(input_frame, text="Apply", command=self.apply_settings).grid(row=1, column=4, padx=5)
        self.start_button = ttk.Button(input_frame, text="START", command=self.start)
        self.start_button.grid(row=1, column=5, padx=5)

        self.status_label = ttk.Label(input_frame, text="Status: waiting for target...")
        self.status_label.grid(row=2, column=0, columnspan=6, sticky="w", pady=(5, 0))

        self.fig = Figure(figsize=(10, 7))
        self.ax_2d = self.fig.add_subplot(2, 2, 1)
        self.ax_w1 = self.fig.add_subplot(2, 2, 2)
        self.ax_w2 = self.fig.add_subplot(2, 2, 3)
        self.ax_w3 = self.fig.add_subplot(2, 2, 4)

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
            self.node.set_noise_std(float(self.noise_entry.get()))
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
            # Confidence circle around the target (radius = goal_tolerance).
            conf_circle = Circle(
                (self.node.target_x, self.node.target_y),
                self.node.goal_tolerance,
                fill=False, linestyle='--', linewidth=1.5, edgecolor=target_color
            )
            self.ax_2d.add_patch(conf_circle)

        self.ax_2d.legend(loc='upper right', fontsize=7)

        # Update the status label / start button once, right when the
        # mission-complete state changes, so the GUI clearly reflects that
        # the bot has fully stopped.
        if self.node.mission_complete and not self._last_mission_complete:
            self.status_label.config(text="Status: ✅ MISSION COMPLETE — target reached, bot fully stopped.")
            self.start_button.config(text="MISSION COMPLETE", state="disabled")
        elif not self.node.mission_complete and self._last_mission_complete:
            self.start_button.config(text="START", state="normal")
        self._last_mission_complete = self.node.mission_complete

        hist_snapshot = list(self.node.history)

        if hist_snapshot:
            t_vals, w1i, w2i, w3i, w1n, w2n, w3n = zip(*hist_snapshot)
        else:
            t_vals = w1i = w2i = w3i = w1n = w2n = w3n = []

        wheel_data = [(w1i, w1n), (w2i, w2n), (w3i, w3n)]
        for i, (ax, (ideal, noisy)) in enumerate(zip([self.ax_w1, self.ax_w2, self.ax_w3], wheel_data)):
            ax.clear()
            ax.set_title(f"Wheel {i+1} velocity (rad/s)")
            ax.set_xlabel("time (s)")
            ax.grid(True)
            if t_vals:
                ax.plot(t_vals, ideal, 'b-', label='Ideal')
                ax.plot(t_vals, noisy, 'r-', alpha=0.6, label='Noisy')
                ax.legend(loc='upper right', fontsize=7)

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