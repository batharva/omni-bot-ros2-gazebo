#!/usr/bin/env python3

import math
import random
import threading
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SpawnEntity

import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation


HISTORY_LEN = 200  # number of samples kept for the rolling wheel-velocity plots


class OmniControllerNode(Node):
    """
    ROS2 node: identical kinematics/constraints logic as before, but all
    inputs (target, max velocity, noise) are now set via thread-safe setter
    methods called from the GUI instead of terminal input().
    """

    def __init__(self):
        super().__init__('omni_gui_controller')

        # Robot geometry
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

        # State set from GUI — protected by GIL for simple float/bool writes,
        # which is sufficient here since we never do read-modify-write races
        self.target_x = 0.0
        self.target_y = 0.0
        self.max_v = 0.5
        self.wheel_noise_std = 0.05
        self.moving = False
        self.target_set = False

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # Rolling history buffers for plotting: time, ideal, noisy per wheel
        self.t_hist = deque(maxlen=HISTORY_LEN)
        self.ideal_hist = [deque(maxlen=HISTORY_LEN) for _ in range(3)]
        self.noisy_hist = [deque(maxlen=HISTORY_LEN) for _ in range(3)]
        self.path_x = deque(maxlen=2000)
        self.path_y = deque(maxlen=2000)
        self._t0 = self.get_clock().now().nanoseconds / 1e9

        self.cmd_pub = self.create_publisher(Twist, '/omni_bot/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/omni_bot/odom', self.odom_callback, 10)

        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')

        self.timer = self.create_timer(0.1, self.control_loop)

    # ---------- GUI-facing setters ----------

    def set_target(self, x, y):
        self.target_x = x
        self.target_y = y
        self.target_set = True
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
        self.moving = True
        self.get_logger().info("Started moving toward target.")

    # ---------- Kinematics (unchanged from before) ----------

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
        self.get_logger().info(f"Controllability: rank={rank}/3, condition={cond:.2f}")
        if rank < 3:
            self.get_logger().error("CONTROLLABILITY LOST: Jacobian rank < 3.")

    def compute_wheel_velocities(self, vx, vy, omega):
        return (self.J @ np.array([vx, vy, omega])).tolist()

    def recover_body_velocity(self, wheel_vels):
        return self.J_pinv @ np.array(wheel_vels)

    def enforce_actuation_limits(self, vx, vy, omega):
        wheel_vels = self.compute_wheel_velocities(vx, vy, omega)
        max_required = max(abs(w) for w in wheel_vels)
        if max_required > self.max_wheel_speed:
            scale = self.max_wheel_speed / max_required
            vx *= scale
            vy *= scale
            omega *= scale
        return vx, vy, omega

    def add_wheel_noise(self, wheel_vels):
        return [v + random.gauss(0.0, self.wheel_noise_std) for v in wheel_vels]

    # ---------- Gazebo pole spawn (unchanged) ----------

    def spawn_target_pole(self):
        if not self.spawn_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/spawn_entity not available — pole not spawned.")
            return

        pole_radius = 0.15
        pole_height = 0.235
        sdf = f"""
        <?xml version="1.0"?>
        <sdf version="1.6">
          <model name="target_pole_{int(self.target_x*1000)}_{int(self.target_y*1000)}">
            <static>true</static>
            <link name="link">
              <visual name="visual">
                <geometry><cylinder><radius>{pole_radius}</radius><length>{pole_height}</length></cylinder></geometry>
                <material><ambient>1 0 0 1</ambient><diffuse>1 0 0 1</diffuse></material>
              </visual>
              <collision name="collision">
                <geometry><cylinder><radius>{pole_radius}</radius><length>{pole_height}</length></cylinder></geometry>
              </collision>
            </link>
          </model>
        </sdf>
        """
        pose = Pose()
        pose.position.x = self.target_x
        pose.position.y = self.target_y
        pose.position.z = pole_height / 2.0
        pose.orientation.w = 1.0

        request = SpawnEntity.Request()
        request.name = f'target_pole_{int(self.target_x*1000)}_{int(self.target_y*1000)}'
        request.xml = sdf
        request.initial_pose = pose
        request.reference_frame = 'world'
        self.spawn_client.call_async(request)

    # ---------- Main control loop ----------

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

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
            self.get_logger().info("Target reached. Stopping.")
            self.moving = False
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
        self.t_hist.append(t)
        for i in range(3):
            self.ideal_hist[i].append(wheel_vels[i])
            self.noisy_hist[i].append(noisy_vels[i])


def ros_spin_thread(node):
    rclpy.spin(node)


class OmniGuiApp:
    def __init__(self, root, node: OmniControllerNode):
        self.node = node
        self.root = root
        root.title("Omni Bot Controller & Live Monitor")

        # ---------- Input panel ----------
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

        # ---------- Matplotlib figure ----------
        self.fig = Figure(figsize=(10, 7))
        self.ax_2d = self.fig.add_subplot(2, 2, 1)
        self.ax_w1 = self.fig.add_subplot(2, 2, 2)
        self.ax_w2 = self.fig.add_subplot(2, 2, 3)
        self.ax_w3 = self.fig.add_subplot(2, 2, 4)

        self.ax_2d.set_title("2D Scenario")
        self.ax_2d.set_xlim(-3, 3)
        self.ax_2d.set_ylim(-3, 3)
        self.ax_2d.set_aspect('equal')
        self.ax_2d.grid(True)

        for ax, name in zip([self.ax_w1, self.ax_w2, self.ax_w3], ["Wheel 1", "Wheel 2", "Wheel 3"]):
            ax.set_title(f"{name} velocity (rad/s)")
            ax.set_xlabel("time (s)")
            ax.grid(True)

        canvas_frame = ttk.Frame(root)
        canvas_frame.grid(row=1, column=0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack()

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
            self.status_label.config(text=f"Status: target set to ({x:.2f}, {y:.2f})")
        except ValueError:
            self.status_label.config(text="Status: invalid target values")

    def apply_settings(self):
        try:
            v = float(self.velocity_entry.get())
            self.node.set_max_velocity(v)
        except ValueError:
            pass
        try:
            n = float(self.noise_entry.get())
            self.node.set_noise_std(n)
        except ValueError:
            pass
        self.status_label.config(text="Status: settings applied")

    def start(self):
        self.node.start_moving()
        self.status_label.config(text="Status: moving toward target...")

    def update_plots(self, frame):
        # 2D scenario
        self.ax_2d.clear()
        self.ax_2d.set_title("2D Scenario")
        self.ax_2d.set_xlim(-3, 3)
        self.ax_2d.set_ylim(-3, 3)
        self.ax_2d.set_aspect('equal')
        self.ax_2d.grid(True)

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
            self.ax_2d.plot(self.node.target_x, self.node.target_y, 'rx', markersize=12, mew=3, label='Target')

        self.ax_2d.legend(loc='upper right', fontsize=7)

        # Wheel velocity plots
        t = list(self.node.t_hist)
        for i, ax in enumerate([self.ax_w1, self.ax_w2, self.ax_w3]):
            ax.clear()
            ax.set_title(f"Wheel {i+1} velocity (rad/s)")
            ax.set_xlabel("time (s)")
            ax.grid(True)
            if t:
                ax.plot(t, list(self.node.ideal_hist[i]), 'b-', label='Ideal')
                ax.plot(t, list(self.node.noisy_hist[i]), 'r-', alpha=0.6, label='Noisy')
                ax.legend(loc='upper right', fontsize=7)

        return []


def main(args=None):
    rclpy.init(args=args)
    node = OmniControllerNode()

    spin_thread = threading.Thread(target=ros_spin_thread, args=(node,), daemon=True)
    spin_thread.start()

    root = tk.Tk()
    app = OmniGuiApp(root, node)
    root.mainloop()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()