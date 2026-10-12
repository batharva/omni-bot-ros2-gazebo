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
from matplotlib.patches import Circle, Polygon
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


HISTORY_LEN = 200
MIN_VIEW_HALF = 1.5  # metres: the 2D view never zooms in tighter than +/- this

# --- Box-touch mission (all lengths in metres) ---------------------------------
# These must match the box in the Gazebo world file (box_mission.world).
# Fixed rectangular target box.
# The controller only needs to know the box COM, dimensions, and yaw.
# Dimensions: 50 x 20 x 10 cm = 0.50 x 0.20 x 0.10 m.
BOX_DIMS = (0.50, 0.20, 0.10)
BOX_DEFAULT_XY = (1.5, 0.8)     # box COM in world coordinates (m)
BOX_YAW = 0.0                   # box orientation in world (rad)

# The mission can target exactly one of these faces:
# '+X', '-X', '+Y', '-Y'
BOX_TARGET_SIDE = '-X'
STICK_TIP_OFFSET = 0.225        # robot centre -> stick tip along +x (0.135 base radius + 0.10 stick - 0.01 overlap)
BOX_CLEARANCE = 0.30            # while travelling, keep the robot CENTRE this far outside the box
DOCK_STANDOFF = 0.55            # robot-centre distance from the face where the final approach starts
DOCK_MAX_SPEED = 0.15           # m/s, speed cap during the final approach


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

        # --- Goal tolerance ------------------------------------------------
        # The mission counts as "complete" as soon as the bot's centre enters
        # a circle of this radius around the target -- it does NOT need to
        # land on the exact (x, y) coordinate. Stored internally in metres
        # (matching odom/world units); the GUI exposes it in millimetres.
        # Default here: 10 mm. Use set_goal_tolerance_mm() to change it
        # (e.g. 5.0 for a 5 mm circle, 10.0 for a 10 mm circle).
        self.goal_tolerance = 0.30  # metres (10 mm)

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

        # --- Box-touch mission state ---
        self.mission_mode = None        # 'point' (go to a coordinate) or 'box' (touch a box face)
        self.box_x, self.box_y = BOX_DEFAULT_XY
        self.box_yaw = BOX_YAW
        self.box_dims = BOX_DIMS
        self.box_side = '-X'
        self._box_face = None
        self.box_dock_yaw = 0.0         # heading that points the stick straight into the face
        self.box_waypoints = []         # planned path of the robot centre (world frame)
        self.box_wp_index = 0
        self.box_phase = 'idle'         # 'transit' -> 'dock' -> 'done'
        self.box_status = ''
        self.box_tip_gap = None         # stick tip to face distance along the face normal (m)
        self.touch_tol = 0.005          # metres
        self.kp_dock = 1.5

        self._actuation_limit_active = False

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        # Actual velocity in the GROUND frame, from odometry
        self.current_vx_world = 0.0
        self.current_vy_world = 0.0
        self.current_speed = 0.0  # |v| (m/s)
        self.current_omega = 0.0  # yaw rate about ground z (rad/s)
        self._prev_odom = None    # (stamp, x, y) for the finite difference

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

        # Velocity logs, all expressed in the GROUND (world) frame.
        # One tuple per sample so the GUI thread always reads a consistent row.
        # velocity_history:     (t, vx_cmd, vy_cmd, omega_cmd, speed_cmd,
        #                        vx_act, vy_act, speed_act, omega_act)
        # wheel_ground_history: (t, ideal, noisy) where ideal/noisy are
        #                        ((vx, vy) of wheel 1, wheel 2, wheel 3)
        self.velocity_history = deque(maxlen=HISTORY_LEN)
        # Per-wheel angular speed history (rad/s): (t, ideal_3, noisy_3)
        self.wheel_angular_history = deque(maxlen=HISTORY_LEN)

        self.path_x = deque(maxlen=2000)
        self.path_y = deque(maxlen=2000)
        self._t0 = self.get_clock().now().nanoseconds / 1e9

        self.cmd_pub = self.create_publisher(Twist, '/omni_bot/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/omni_bot/odom', self.odom_callback, 10)

        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)
        self.frame_id = 'odom'

        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')

        self.timer = self.create_timer(self.control_dt, self.control_loop)
        self.marker_timer = self.create_timer(1.0, self.publish_target_marker)

    def set_target(self, x, y):
        if self.mission_mode == 'box':
            # a new point target aborts a running box mission
            self.moving = False
            self.cmd_pub.publish(Twist())
        self.mission_mode = 'point'
        self.target_x = x
        self.target_y = y
        self.target_set = True
        self.mission_complete = False
        self.get_logger().info(f"Target set to: x={x:.2f}, y={y:.2f}")
        self.spawn_target_pole()

    def set_max_velocity(self, v):
        self.max_v = v
        self.get_logger().info(f"Max linear velocity set to: {v:.2f} m/s")

    def set_goal_tolerance_mm(self, tolerance_mm):
        """tolerance_mm: radius (in millimetres) of the circle around the
        target within which the bot's centre is considered to have arrived,
        e.g. 5.0 -> 5 mm circle, 10.0 -> 10 mm circle."""
        if tolerance_mm <= 0:
            self.get_logger().warn("Goal tolerance must be positive — ignoring.")
            return
        self.goal_tolerance = tolerance_mm / 1000.0
        self.get_logger().info(
            f"Goal tolerance set to: {tolerance_mm:.2f} mm "
            f"({self.goal_tolerance:.4f} m)"
        )

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
        self.mission_mode = 'point'
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

        # Actual ground-frame velocity: differentiate the odom POSITION (which is
        # always in the ground frame) using the message's own sim-time stamp.
        # This avoids guessing which frame the odom twist is expressed in.
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._prev_odom is None:
            self._prev_odom = (stamp, self.current_x, self.current_y)
        else:
            t_prev, x_prev, y_prev = self._prev_odom
            dt = stamp - t_prev
            if dt < 0.0:  # sim was reset
                self._prev_odom = (stamp, self.current_x, self.current_y)
            elif dt >= 0.05:  # >=50 ms baseline keeps the difference smooth
                self.current_vx_world = (self.current_x - x_prev) / dt
                self.current_vy_world = (self.current_y - y_prev) / dt
                self._prev_odom = (stamp, self.current_x, self.current_y)
        self.current_speed = math.hypot(self.current_vx_world, self.current_vy_world)
        self.current_omega = msg.twist.twist.angular.z  # yaw rate: same in any frame

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

    @staticmethod
    def body_to_world(vx, vy, yaw):
        """Rotate a body-frame vector into the ground (world) frame."""
        c, s = math.cos(yaw), math.sin(yaw)
        return c * vx - s * vy, s * vx + c * vy

    def wheel_ground_velocities(self, vx, vy, omega, yaw):
        """Velocity of each wheel CENTRE relative to the ground, in ground-frame
        axes:  v_wheel = v_robot_world + omega x r_wheel_world,
        with r_wheel_world = wheel_offset * (cos(a), sin(a)), a = wheel_angle + yaw.
        (vx, vy, omega) is a body-frame twist. Returns [(wx, wy)] for wheels 1..3.
        Projected on each wheel's rolling direction this equals the wheel's
        driven speed (wheel_radius * wheel angular velocity) from the Jacobian."""
        vx_w, vy_w = self.body_to_world(float(vx), float(vy), yaw)
        out = []
        for th in self.wheel_angles:
            a = th + yaw
            out.append((
                vx_w - float(omega) * self.wheel_offset * math.sin(a),
                vy_w + float(omega) * self.wheel_offset * math.cos(a),
            ))
        return out

    def compute_free_slide_components(self, vx, vy):
        v_body = np.array([vx, vy])
        return [float(np.dot(v_body, d)) for d in self.free_slide_dirs]

    @staticmethod
    def _normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

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
        if not self.target_set or self.mission_mode == 'box':
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

    # ------------------------------------------------------------------
    # Box-touch mission: drive so the STICK TIP touches one chosen face of a
    # box whose centre of mass and dimensions are known.
    #   1. plan a path for the robot centre AROUND the box (box inflated by BOX_CLEARANCE)
    #   2. arrive at a pre-dock point in front of the chosen face and turn the
    #      stick to point straight into the face
    #   3. creep forward along the face normal until the tip is within touch_tol
    # ------------------------------------------------------------------
    _FACE_NORMALS = {'+X': (1.0, 0.0), '-X': (-1.0, 0.0), '+Y': (0.0, 1.0), '-Y': (0.0, -1.0)}

    def set_touch_tolerance_mm(self, tol_mm):
        if tol_mm <= 0:
            self.get_logger().warn("Touch tolerance must be positive — ignoring.")
            return
        self.touch_tol = tol_mm / 1000.0

    def box_to_world(self, px, py):
        c, s = math.cos(self.box_yaw), math.sin(self.box_yaw)
        return self.box_x + c * px - s * py, self.box_y + s * px + c * py

    def world_to_box(self, wx, wy):
        c, s = math.cos(self.box_yaw), math.sin(self.box_yaw)
        dx, dy = wx - self.box_x, wy - self.box_y
        return c * dx + s * dy, -s * dx + c * dy

    def box_corners(self):
        hl, hw = self.box_dims[0] / 2.0, self.box_dims[1] / 2.0
        return [self.box_to_world(sx * hl, sy * hw)
                for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]

    def box_face_geometry(self, side):
        """Face centre, outward unit normal n, unit tangent t (all world frame) and
        the face's half-width. `side` is given in the BOX frame: +X/-X are the
        faces normal to the box's length axis, +Y/-Y the faces normal to its width axis."""
        nlx, nly = self._FACE_NORMALS[side]
        hl, hw = self.box_dims[0] / 2.0, self.box_dims[1] / 2.0
        half_n = hl if nlx != 0.0 else hw   # centre -> face distance
        half_t = hw if nlx != 0.0 else hl   # half of the face's width
        c, s = math.cos(self.box_yaw), math.sin(self.box_yaw)
        nx, ny = c * nlx - s * nly, s * nlx + c * nly
        return {
            'centre': (self.box_x + nx * half_n, self.box_y + ny * half_n),
            'n': (nx, ny),
            't': (-ny, nx),
            'half_width': half_t,
        }

    def box_tolerance_zone_corners(self):
        """Return a thin rectangular tolerance zone around the selected box face.
        The zone is centered on the face and extends touch_tol outward/inward
        along the face normal, while covering the usable face width.
        """
        if self._box_face is None:
            return []
        face = self._box_face
        fx, fy = face['centre']
        nx, ny = face['n']
        tx, ty = face['t']
        # Slight margin keeps the visual zone aligned with the actual touch
        # condition, which requires the tip to stay on the usable face.
        half_width = max(0.0, face['half_width'] - 0.01)
        depth = self.touch_tol
        return [
            (fx + tx * half_width + nx * depth, fy + ty * half_width + ny * depth),
            (fx - tx * half_width + nx * depth, fy - ty * half_width + ny * depth),
            (fx - tx * half_width - nx * depth, fy - ty * half_width - ny * depth),
            (fx + tx * half_width - nx * depth, fy + ty * half_width - ny * depth),
        ]

    def stick_tip(self, x=None, y=None, yaw=None):
        x = self.current_x if x is None else x
        y = self.current_y if y is None else y
        yaw = self.current_yaw if yaw is None else yaw
        return x + STICK_TIP_OFFSET * math.cos(yaw), y + STICK_TIP_OFFSET * math.sin(yaw)

    @staticmethod
    def _segment_blocked(p, q, ax, ay):
        """True if segment p->q (box frame) crosses the OPEN interior of the
        rectangle |x|<ax, |y|<ay. Liang-Barsky clip, then test the clipped midpoint
        (so sliding along an edge or touching a corner counts as free)."""
        dx, dy = q[0] - p[0], q[1] - p[1]
        t0, t1 = 0.0, 1.0
        for pk, qk in ((-dx, p[0] + ax), (dx, ax - p[0]), (-dy, p[1] + ay), (dy, ay - p[1])):
            if abs(pk) < 1e-12:
                if qk < 0.0:
                    return False        # parallel to this edge and outside it
            else:
                r = qk / pk
                if pk < 0.0:
                    if r > t1:
                        return False
                    t0 = max(t0, r)
                else:
                    if r < t0:
                        return False
                    t1 = min(t1, r)
        if t1 - t0 < 1e-9:
            return False
        tm = 0.5 * (t0 + t1)
        mx, my = p[0] + tm * dx, p[1] + tm * dy
        return abs(mx) < ax - 1e-6 and abs(my) < ay - 1e-6

    def _plan_box_path(self, start_w, goal_w):
        """Shortest path for the robot CENTRE from start to goal that stays outside
        the box inflated by BOX_CLEARANCE (visibility graph over the 4 inflated
        corners + Dijkstra). Returns world-frame waypoints, excluding the start."""
        ax = self.box_dims[0] / 2.0 + BOX_CLEARANCE
        ay = self.box_dims[1] / 2.0 + BOX_CLEARANCE
        S = self.world_to_box(*start_w)
        G = self.world_to_box(*goal_w)

        first = []
        if abs(S[0]) < ax and abs(S[1]) < ay:
            # Started inside the clearance zone: step out through the nearest side first
            exits = [(ax - S[0], (ax + 0.02, S[1])), (S[0] + ax, (-ax - 0.02, S[1])),
                     (ay - S[1], (S[0], ay + 0.02)), (S[1] + ay, (S[0], -ay - 0.02))]
            S = min(exits, key=lambda e: e[0])[1]
            first = [S]

        eps = 1e-3
        corners = [(sx * (ax + eps), sy * (ay + eps)) for sx in (1, -1) for sy in (1, -1)]
        nodes = [S] + corners + [G]
        n = len(nodes)
        dist = [math.inf] * n
        prev = [None] * n
        done = [False] * n
        dist[0] = 0.0
        for _ in range(n):
            u = min((i for i in range(n) if not done[i]), key=lambda i: dist[i])
            done[u] = True
            for v in range(n):
                if done[v] or self._segment_blocked(nodes[u], nodes[v], ax, ay):
                    continue
                d = dist[u] + math.dist(nodes[u], nodes[v])
                if d < dist[v]:
                    dist[v] = d
                    prev[v] = u
        path, k = [], n - 1
        while k is not None:
            path.append(nodes[k])
            k = prev[k]
        path.reverse()
        if path[0] != S:            # goal unreachable (should not happen) -> go straight
            path = [S, G]
        return [self.box_to_world(*pt) for pt in first + path[1:]]

    def start_fixed_box_mission(self):
        """Start the mission for the fixed box defined by the world constants."""
        self.start_box_mission(
            BOX_DEFAULT_XY[0],
            BOX_DEFAULT_XY[1],
            BOX_TARGET_SIDE,
        )

    def start_box_mission(self, bx, by, side):
        if side not in self._FACE_NORMALS:
            self.get_logger().warn(f"Unknown box side '{side}' — use one of +X, -X, +Y, -Y.")
            return
        self.box_x, self.box_y = bx, by
        self.box_side = side
        face = self.box_face_geometry(side)
        nx, ny = face['n']
        fx, fy = face['centre']
        self.box_dock_yaw = math.atan2(-ny, -nx)        # stick points along -n (into the face)
        dock_centre = (fx + nx * STICK_TIP_OFFSET, fy + ny * STICK_TIP_OFFSET)
        pre_dock = (fx + nx * DOCK_STANDOFF, fy + ny * DOCK_STANDOFF)
        self.box_waypoints = self._plan_box_path((self.current_x, self.current_y), pre_dock)
        self.box_wp_index = 0
        self.box_tip_gap = None
        self.box_status = 'starting'

        # Reuse the "bot vs target frame" window: A = start, B = docked robot centre,
        # desired heading = docking heading.
        self.start_x, self.start_y = self.current_x, self.current_y
        self.target_x, self.target_y = dock_centre
        self.line_yaw = self.box_dock_yaw
        self.target_set = True

        # publish state last so the control loop never sees a half-configured mission
        self._box_face = face
        self.box_phase = 'transit'
        self.mission_complete = False
        self.mission_mode = 'box'
        self.moving = True
        self.get_logger().info(
            f"Box mission started: touch the {side} face at ({fx:.2f}, {fy:.2f}) with the stick tip; "
            f"{len(self.box_waypoints)} waypoint(s) planned, final heading "
            f"{math.degrees(self.box_dock_yaw):.1f}°."
        )

    def _box_mission_command(self):
        """One control tick of the box mission. Returns the body-frame command
        (vx, vy, omega), or None once the stick tip has touched the face."""
        face = self._box_face
        nx, ny = face['n']
        tx, ty = face['t']
        fx, fy = face['centre']

        # Heading: keep the stick pointing straight into the chosen face
        self.yaw_error = self._normalize_angle(self.box_dock_yaw - self.current_yaw)
        omega = max(min(self.kp_yaw * self.yaw_error, self.max_omega), -self.max_omega)

        # Stick tip relative to the face: gap along the outward normal
        # (>0 outside the box, 0 = touching) and offset along the face
        tip_x, tip_y = self.stick_tip()
        rel_x, rel_y = tip_x - fx, tip_y - fy
        gap = rel_x * nx + rel_y * ny
        lateral = rel_x * tx + rel_y * ty
        self.box_tip_gap = gap

        vx_w = vy_w = 0.0

        if self.box_phase == 'transit':
            last = len(self.box_waypoints) - 1
            # skip intermediate waypoints we are already next to
            while (self.box_wp_index < last and
                   math.hypot(self.box_waypoints[self.box_wp_index][0] - self.current_x,
                              self.box_waypoints[self.box_wp_index][1] - self.current_y) < 0.08):
                self.box_wp_index += 1
            wx, wy = self.box_waypoints[self.box_wp_index]
            ex, ey = wx - self.current_x, wy - self.current_y
            dist = math.hypot(ex, ey)
            if self.box_wp_index < last:
                speed = self.max_v
                self.box_status = f"transit — waypoint {self.box_wp_index + 1}/{last + 1}"
            else:
                # pre-dock point: settle here and finish turning before docking
                speed = min(self.max_v, self.kp_linear * dist)
                self.box_status = (f"aligning at pre-dock point — heading err "
                                   f"{math.degrees(self.yaw_error):+.1f}°")
                if dist < 0.03 and abs(self.yaw_error) < math.radians(2.0):
                    self.box_phase = 'dock'
                    self.get_logger().info("Pre-dock pose reached — docking.")
            if dist > 1e-6:
                vx_w, vy_w = speed * ex / dist, speed * ey / dist

        if self.box_phase == 'dock':
            # Touch is accepted only when the stick tip is inside the
            # tolerance band on the selected face, within the usable face width,
            # and the robot is correctly aligned.
            if (0.0 <= gap <= self.touch_tol
                    and abs(lateral) <= face['half_width'] - 0.01
                    and abs(self.yaw_error) < math.radians(5.0)):
                self.cmd_pub.publish(Twist())
                self.moving = False
                self.mission_complete = True
                self.box_phase = 'done'
                self.box_status = 'touched'
                self.get_logger().info(
                    f"BOX MISSION COMPLETE — stick tip touched the {self.box_side} face "
                    f"(gap {gap * 1000:.1f} mm, {lateral * 1000:+.0f} mm from face centre). Bot stopped."
                )
                return None
            ex, ey = fx - tip_x, fy - tip_y           # tip -> face centre
            dist = math.hypot(ex, ey)
            speed = min(self.max_v, DOCK_MAX_SPEED, self.kp_dock * dist)
            self.box_status = f"docking — tip gap {gap * 1000:.0f} mm"
            if dist > 1e-6:
                vx_w, vy_w = speed * ex / dist, speed * ey / dist

        # ground-frame velocity -> body frame (same convention as the point mission)
        c, s = math.cos(self.current_yaw), math.sin(self.current_yaw)
        return c * vx_w + s * vy_w, -s * vx_w + c * vy_w, omega

    def control_loop(self):
        self.path_x.append(self.current_x)
        self.path_y.append(self.current_y)

        if not self.moving:
            return

        if self.mission_mode == 'box':
            cmd = self._box_mission_command()
            if cmd is None:
                return  # stick tip touched the face this tick
            vx, vy, omega = cmd
        else:
            dx = self.target_x - self.current_x
            dy = self.target_y - self.current_y
            distance = math.sqrt(dx ** 2 + dy ** 2)

            # Mission is complete as soon as the bot's centre enters the
            # goal-tolerance circle around the target — it does not need to
            # hit the exact (x, y) coordinate.
            if distance < self.goal_tolerance:
                self.cmd_pub.publish(Twist())
                self.moving = False
                if not self.mission_complete:
                    self.mission_complete = True
                    self.get_logger().info(
                        f"MISSION COMPLETE — target reached within confidence "
                        f"circle (radius={self.goal_tolerance * 1000:.1f} mm). Bot stopped."
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

        # Commanded robot velocity rotated into the ground frame
        yaw = self.current_yaw
        vx_w, vy_w = self.body_to_world(vx, vy, yaw)
        self.velocity_history.append((
            t, vx_w, vy_w, omega, math.hypot(vx, vy),
            self.current_vx_world, self.current_vy_world,
            self.current_speed, self.current_omega,
        ))

        # Per-wheel ANGULAR SPEED:
        # compute_wheel_velocities() already returns each wheel's rolling
        # angular speed in rad/s because the Jacobian divides by wheel_radius.
        # Therefore there is no need to display wheel Vx/Vy velocities.
        self.wheel_angular_history.append((
            t,
            list(wheel_vels),      # ideal wheel angular speeds [rad/s]
            list(noisy_vels),      # noisy/actual wheel angular speeds [rad/s]
        ))


def ros_spin_thread(node):
    rclpy.spin(node)


class OmniGuiApp:
    """
    Main window holds only the controls. Each graph group now lives in its
    own Toplevel window:
      - win_2d        : 2D scenario (path, robot, target)
      - win_theta      : all 3 wheel theta plots (ideal vs noisy)
      - win_noise      : total robot noise magnitude
      - win_alignment  : bot vs target body frame
      - win_velocity   : robot velocity + per-wheel angular speeds
    A single Tk `after` loop (no FuncAnimation, since animations only drive
    one figure/canvas) refreshes all five windows together every 200 ms.
    """

    def __init__(self, root, node: OmniController):
        self.node = node
        self.root = root
        root.title("Omni Bot Controller")

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

        # --- Per-wheel noise entries ---
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

        # --- Goal tolerance entry ---
        ttk.Label(input_frame, text="Goal Tolerance (mm):").grid(row=2, column=0)
        self.tolerance_entry = ttk.Entry(input_frame, width=8)
        self.tolerance_entry.insert(0, f"{self.node.goal_tolerance * 1000:.1f}")
        self.tolerance_entry.grid(row=2, column=1)

        ttk.Button(input_frame, text="Apply", command=self.apply_settings).grid(row=1, column=5, padx=5)
        self.start_button = ttk.Button(input_frame, text="START", command=self.start)
        self.start_button.grid(row=2, column=5, padx=5)

        self.status_label = ttk.Label(input_frame, text="Status: waiting for target...")
        self.status_label.grid(row=3, column=0, columnspan=6, sticky="w", pady=(5, 0))

        # --- Box-touch mission controls ---
        ttk.Separator(input_frame, orient="horizontal").grid(
            row=4, column=0, columnspan=6, sticky="ew", pady=6)

        ttk.Label(input_frame, text="Box COM X (m):").grid(row=5, column=0)
        self.box_x_entry = ttk.Entry(input_frame, width=8)
        self.box_x_entry.insert(0, str(BOX_DEFAULT_XY[0]))
        self.box_x_entry.grid(row=5, column=1)

        ttk.Label(input_frame, text="Box COM Y (m):").grid(row=5, column=2)
        self.box_y_entry = ttk.Entry(input_frame, width=8)
        self.box_y_entry.insert(0, str(BOX_DEFAULT_XY[1]))
        self.box_y_entry.grid(row=5, column=3)

        ttk.Label(input_frame, text="Touch side:").grid(row=6, column=0)
        self.box_side_combo = ttk.Combobox(
            input_frame, values=["+X", "-X", "+Y", "-Y"], width=6, state="readonly")
        self.box_side_combo.set(BOX_TARGET_SIDE)
        self.box_side_combo.grid(row=6, column=1)

        ttk.Label(input_frame, text="Touch tol (mm):").grid(row=6, column=2)
        self.touch_tol_entry = ttk.Entry(input_frame, width=8)
        self.touch_tol_entry.insert(0, "5")
        self.touch_tol_entry.grid(row=6, column=3)

        ttk.Button(input_frame, text="START BOX MISSION",
                   command=self.start_box_mission).grid(row=6, column=5, padx=5)

        self._last_mission_complete = False

        # ------------------------------------------------------------------
        # Window 1: 2D scenario (path, robot, target)
        # ------------------------------------------------------------------
        self.win_2d = tk.Toplevel(root)
        self.win_2d.title("1) 2D Scenario")
        self.fig_2d = Figure(figsize=(5.5, 5.5))
        self.ax_2d = self.fig_2d.add_subplot(1, 1, 1)
        self.canvas_2d = FigureCanvasTkAgg(self.fig_2d, master=self.win_2d)
        self.canvas_2d.get_tk_widget().pack(fill="both", expand=True)

        # ------------------------------------------------------------------
        # Window 2: all 3 wheel theta plots
        # ------------------------------------------------------------------
        self.win_theta = tk.Toplevel(root)
        self.win_theta.title("2) Wheel Theta — Ideal vs Noisy")
        self.fig_theta = Figure(figsize=(13, 4.2))
        self.ax_w1 = self.fig_theta.add_subplot(1, 3, 1)
        self.ax_w2 = self.fig_theta.add_subplot(1, 3, 2)
        self.ax_w3 = self.fig_theta.add_subplot(1, 3, 3)
        self.canvas_theta = FigureCanvasTkAgg(self.fig_theta, master=self.win_theta)
        self.canvas_theta.get_tk_widget().pack(fill="both", expand=True)

        # ------------------------------------------------------------------
        # Window 3: total robot noise magnitude
        # ------------------------------------------------------------------
        self.win_noise = tk.Toplevel(root)
        self.win_noise.title("3) Total Robot Noise Magnitude")
        self.fig_noise = Figure(figsize=(5.5, 4.2))
        self.ax_robot_noise_mag = self.fig_noise.add_subplot(1, 1, 1)
        self.canvas_noise = FigureCanvasTkAgg(self.fig_noise, master=self.win_noise)
        self.canvas_noise.get_tk_widget().pack(fill="both", expand=True)

        # ------------------------------------------------------------------
        # Window 4: bot vs target body frame
        # ------------------------------------------------------------------
        self.win_alignment = tk.Toplevel(root)
        self.win_alignment.title("4) Bot vs Target Frame")
        self.fig_alignment = Figure(figsize=(5.5, 5.5))
        self.ax_alignment = self.fig_alignment.add_subplot(1, 1, 1)
        self.canvas_alignment = FigureCanvasTkAgg(self.fig_alignment, master=self.win_alignment)
        self.canvas_alignment.get_tk_widget().pack(fill="both", expand=True)

        # ------------------------------------------------------------------
        # Window 5: robot velocity + per-wheel angular speed.
        # Left: robot linear + angular velocity.
        # Right: angular speed of wheels 1..3.
        # ------------------------------------------------------------------
        self.win_velocity = tk.Toplevel(root)
        self.win_velocity.title("5) Velocity — Robot + Wheel Angular Speeds")
        self.fig_velocity = Figure(figsize=(11, 7.5), constrained_layout=True)
        gs = self.fig_velocity.add_gridspec(3, 2)
        self.ax_lin_vel = self.fig_velocity.add_subplot(gs[0:2, 0])
        self.ax_ang_vel = self.fig_velocity.add_subplot(gs[2, 0])
        self.ax_wv = [self.fig_velocity.add_subplot(gs[i, 1]) for i in range(3)]
        self.canvas_velocity = FigureCanvasTkAgg(self.fig_velocity, master=self.win_velocity)
        self.canvas_velocity.get_tk_widget().pack(fill="both", expand=True)

        # Tile the plot windows around the main control window instead of
        # letting them all pop up stacked on top of each other.
        root.update_idletasks()
        base_x = root.winfo_x() + root.winfo_width() + 20
        base_y = max(root.winfo_y(), 0)
        self.win_2d.geometry(f"+{base_x}+{base_y}")
        self.win_theta.geometry(f"+{base_x}+{base_y + 480}")
        self.win_noise.geometry(f"+{base_x + 560}+{base_y}")
        self.win_alignment.geometry(f"+{base_x + 560}+{base_y + 480}")
        # Velocity window goes under the control panel (free space on the left)
        self.win_velocity.geometry(f"+{root.winfo_x()}+{root.winfo_y() + root.winfo_height() + 40}")

        # Closing any plot window just hides it (keeps refreshing it in the
        # background) rather than tearing down the whole app.
        for win in (self.win_2d, self.win_theta, self.win_noise, self.win_alignment,
                    self.win_velocity):
            win.protocol("WM_DELETE_WINDOW", win.withdraw)

        # Single periodic refresh loop drives all 4 windows together.
        self.root.after(200, self.update_plots)

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
        try:
            self.node.set_goal_tolerance_mm(float(self.tolerance_entry.get()))
        except ValueError:
            pass
        self.status_label.config(text="Status: settings applied")

    def start(self):
        self.node.start_moving()
        self._last_mission_complete = False
        self.status_label.config(text="Status: moving toward target...")

    def start_box_mission(self):
        try:
            bx = float(self.box_x_entry.get())
            by = float(self.box_y_entry.get())
            tol_mm = float(self.touch_tol_entry.get())
        except ValueError:
            self.status_label.config(text="Status: invalid box mission values")
            return
        self.apply_settings()  # pick up max velocity / noise entries
        self.node.set_touch_tolerance_mm(tol_mm)
        side = self.box_side_combo.get()
        self.node.start_box_mission(bx, by, side)
        self._last_mission_complete = False
        self.status_label.config(text=f"Status: box mission started — aiming for the {side} face...")

    def update_plots(self):
        # ---------------- Window 1: 2D scenario ----------------
        px, py = list(self.node.path_x), list(self.node.path_y)
        n = min(len(px), len(py))  # ROS thread may be between the x and y append
        px, py = px[:n], py[:n]

        # Auto-fit the view to whatever the user has set up: robot, its path,
        # and the target together with its tolerance circle.
        xs = [self.node.current_x] + px
        ys = [self.node.current_y] + py
        if self.node.target_set and self.node.mission_mode != 'box':
            tol = self.node.goal_tolerance
            xs += [self.node.target_x - tol, self.node.target_x + tol]
            ys += [self.node.target_y - tol, self.node.target_y + tol]
        if self.node.mission_mode == 'box':
            for qx, qy in self.node.box_corners() + list(self.node.box_waypoints):
                xs.append(qx)
                ys.append(qy)
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        # 20% margin, rounded up to the next 0.5 m so the view doesn't twitch every frame
        half = max(MIN_VIEW_HALF, math.ceil(span * 0.6 * 2) / 2)
        cx = 0.5 * (max(xs) + min(xs))
        cy = 0.5 * (max(ys) + min(ys))

        self.ax_2d.clear()
        self.ax_2d.set_xlim(cx - half, cx + half)
        self.ax_2d.set_ylim(cy - half, cy + half)
        self.ax_2d.set_aspect('equal')
        self.ax_2d.grid(True)

        if self.node.mission_complete:
            self.ax_2d.set_title("2D Scenario — MISSION COMPLETE", color='green', fontweight='bold')
        else:
            self.ax_2d.set_title("2D Scenario")

        if px:
            self.ax_2d.plot(px, py, 'b-', linewidth=1, label='Path')

        self.ax_2d.plot(self.node.current_x, self.node.current_y, 'ko', markersize=8, label='Robot')
        arrow_len = 2 * half / 30  # heading arrow scales with the zoom level
        self.ax_2d.arrow(
            self.node.current_x, self.node.current_y,
            arrow_len * math.cos(self.node.current_yaw),
            arrow_len * math.sin(self.node.current_yaw),
            head_width=0.4 * arrow_len, color='black'
        )

        if self.node.target_set and self.node.mission_mode != 'box':
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

        if self.node.mission_mode == 'box':
            fc = self.node._box_face
            self.ax_2d.add_patch(Polygon(
                self.node.box_corners(), closed=True, facecolor='steelblue',
                edgecolor='navy', alpha=0.5, label='Box'))
            # the face the stick must touch, highlighted
            hw = fc['half_width']
            (fx, fy), (tx_, ty_) = fc['centre'], fc['t']
            self.ax_2d.plot([fx + tx_ * hw, fx - tx_ * hw], [fy + ty_ * hw, fy - ty_ * hw],
                            '-', color='orange', linewidth=4, label=f'Target face ({self.node.box_side})')
            # Box touch tolerance zone — analogous to the circular tolerance
            # zone used for the point/pole mission.
            tol_corners = self.node.box_tolerance_zone_corners()
            if tol_corners:
                self.ax_2d.add_patch(Polygon(
                    tol_corners, closed=True,
                    facecolor='green' if self.node.mission_complete else 'limegreen',
                    edgecolor='green', alpha=0.25,
                    linestyle='--', linewidth=1.5,
                    label=f'Touch tolerance ±{self.node.touch_tol * 1000:.1f} mm'))
            wps = [(self.node.start_x, self.node.start_y)] + list(self.node.box_waypoints)
            self.ax_2d.plot([p[0] for p in wps], [p[1] for p in wps], 'c--', linewidth=1, label='Planned path')
            tip_x, tip_y = self.node.stick_tip()
            self.ax_2d.plot([self.node.current_x, tip_x], [self.node.current_y, tip_y],
                            '-', color='gold', linewidth=3, label='Stick')
            self.ax_2d.plot(tip_x, tip_y, 'r.', markersize=8)

        self.ax_2d.legend(loc='upper right', fontsize=7)

        if self.node.mission_complete and not self._last_mission_complete:
            if self.node.mission_mode == 'box':
                gap_mm = (self.node.box_tip_gap or 0.0) * 1000.0
                self.status_label.config(
                    text=f"Status: ✅ BOX MISSION COMPLETE — stick tip touched the "
                         f"{self.node.box_side} face (gap {gap_mm:.1f} mm), bot stopped."
                )
            else:
                self.status_label.config(
                    text=f"Status: ✅ MISSION COMPLETE — target reached within "
                         f"{self.node.goal_tolerance * 1000:.1f} mm, bot fully stopped."
                )
                self.start_button.config(text="MISSION COMPLETE", state="disabled")
        elif not self.node.mission_complete and self._last_mission_complete:
            self.start_button.config(text="START", state="normal")
        if self.node.mission_mode == 'box' and self.node.moving:
            self.status_label.config(text=f"Status: box mission — {self.node.box_status}")
        self._last_mission_complete = self.node.mission_complete

        # ---------------- Window 2: all 3 wheel thetas ----------------
        hist_snapshot = list(self.node.history)

        if hist_snapshot:
            t_vals, th1i, th2i, th3i, th1n, th2n, th3n = zip(*hist_snapshot)
        else:
            t_vals = th1i = th2i = th3i = th1n = th2n = th3n = []

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

        # ---------------- Window 3: total robot noise magnitude ----------------
        robot_hist = list(self.node.robot_noise_history)
        if robot_hist:
            rt, _, _, _, rmag = zip(*robot_hist)
        else:
            rt = rmag = []

        self.ax_robot_noise_mag.clear()
        self.ax_robot_noise_mag.set_title("Total robot noise magnitude (m)")
        self.ax_robot_noise_mag.set_xlabel("time (s)")
        self.ax_robot_noise_mag.set_ylabel("|drift| (m)")
        self.ax_robot_noise_mag.grid(True)
        if rt:
            self.ax_robot_noise_mag.plot(rt, rmag, 'm-', linewidth=1.2)

        # ---------------- Window 4: bot vs target body frame ----------------
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
            if self.node.mission_mode == 'box' and self.node.box_tip_gap is not None:
                self.ax_alignment.set_title(
                    f"Bot vs dock frame — heading err={yaw_err_deg:+.1f}°, "
                    f"tip gap={self.node.box_tip_gap * 1000:.0f} mm"
                )
            else:
                self.ax_alignment.set_title(
                    f"Bot vs target frame — heading err={yaw_err_deg:+.1f}°, line dev={perp_dev:.3f} m"
                )
            self.ax_alignment.legend(loc='best', fontsize=6)
        else:
            self.ax_alignment.set_title("Bot vs target frame (no target set)")

        # ---------------- Window 5: velocities in the GROUND (world) frame ----------------
        vel_hist = list(self.node.velocity_history)
        if vel_hist:
            vt, vxc, vyc, omc, spd_c, vxa, vya, spd_a, oma = zip(*vel_hist)
        else:
            vt = vxc = vyc = omc = spd_c = vxa = vya = spd_a = oma = []

        self.ax_lin_vel.clear()
        self.ax_lin_vel.set_title("Robot linear velocity — ground frame (m/s)")
        self.ax_lin_vel.set_ylabel("m/s")
        self.ax_lin_vel.grid(True)
        if vt:
            self.ax_lin_vel.plot(vt, vxc, 'g-', label='Vx cmd')
            self.ax_lin_vel.plot(vt, vxa, 'g--', alpha=0.8, label='Vx actual')
            self.ax_lin_vel.plot(vt, vyc, 'm-', label='Vy cmd')
            self.ax_lin_vel.plot(vt, vya, 'm--', alpha=0.8, label='Vy actual')
            self.ax_lin_vel.plot(vt, spd_c, 'b:', label='|V| cmd')
            self.ax_lin_vel.plot(vt, spd_a, 'r:', label='|V| actual')
            self.ax_lin_vel.legend(loc='upper right', fontsize=7, ncol=3)

        self.ax_ang_vel.clear()
        self.ax_ang_vel.set_title("Yaw rate about ground z (rad/s)")
        self.ax_ang_vel.set_xlabel("time (s)")
        self.ax_ang_vel.set_ylabel("rad/s")
        self.ax_ang_vel.grid(True)
        if vt:
            self.ax_ang_vel.plot(vt, omc, 'b--', label='omega cmd')
            self.ax_ang_vel.plot(vt, oma, 'r-', alpha=0.7, label='omega actual (odom)')
            self.ax_ang_vel.legend(loc='upper right', fontsize=7)

        # Per-wheel ANGULAR SPEED — no Vx/Vy components for individual wheels.
        # wheel_vels/noisy_vels are already in rad/s:
        #     wheel_omega = rolling_velocity / wheel_radius
        wheel_hist = list(self.node.wheel_angular_history)
        wt = [row[0] for row in wheel_hist]

        for i, ax in enumerate(self.ax_wv):
            ax.clear()
            ax.set_title(f"Wheel {i+1} angular speed")
            ax.set_ylabel("rad/s")
            if i == len(self.ax_wv) - 1:
                ax.set_xlabel("time (s)")
            ax.grid(True)

            if wt:
                omega_ideal = [row[1][i] for row in wheel_hist]
                omega_noisy = [row[2][i] for row in wheel_hist]

                ax.plot(wt, omega_ideal, 'g-', label='Ideal ω')
                ax.plot(
                    wt, omega_noisy, 'r--',
                    alpha=0.75, linewidth=0.9, label='Noisy ω'
                )
                ax.legend(loc='upper right', fontsize=7)

        # ---------------- Redraw all 5 windows, then reschedule ----------------
        self.canvas_2d.draw_idle()
        self.canvas_theta.draw_idle()
        self.canvas_noise.draw_idle()
        self.canvas_alignment.draw_idle()
        self.canvas_velocity.draw_idle()

        self.root.after(200, self.update_plots)


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