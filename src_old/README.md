# omni-bot-ros2-gazebo

Simulation of a 3-wheel omnidirectional robot in Gazebo using ROS2 Humble, with target-based navigation, wheel velocity noise injection, and live RViz2 visualization.

## Features

- 3-wheel omni-directional robot (wheels at 120°/90°-210°-330° layout)
- Target-based navigation (user-specified or randomly generated target)
- Real-time wheel velocity computation and logging
- Gaussian noise injection on wheel velocities (simulates encoder/slip noise)
- Live RViz2 visualization: target marker (red sphere) + movement vector (green arrow)

## Prerequisites

- Ubuntu 22.04
- ROS2 Humble
- Gazebo Classic 11

## Installation

### 1. Install ROS2 Humble

```bash
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

sudo apt install -y software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install -y curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt upgrade -y
sudo apt install -y ros-humble-desktop-full

echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 2. Install Gazebo Classic 11 and ROS2 integration packages

```bash
sudo apt install -y \
  gazebo \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-gazebo-ros2-control \
  ros-humble-xacro \
  ros-humble-joint-state-publisher \
  ros-humble-joint-state-publisher-gui \
  ros-humble-robot-state-publisher \
  ros-humble-teleop-twist-keyboard \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-rviz2
```

Verify Gazebo installed correctly:
```bash
gazebo --version
```

### 3. Clone and build this repo

```bash
mkdir -p ~/omni_ws/src
cd ~/omni_ws/src
git clone https://github.com/batharva/omni-bot-ros2-gazebo.git
cd ~/omni_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

Add the workspace to your `.bashrc` so it's sourced automatically in every new terminal:
```bash
echo "source ~/omni_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Usage

### 1. Launch Gazebo, spawn the robot, and open RViz2

```bash
ros2 launch omni_bot_gazebo spawn_omni.launch.py
```

This will:
- Start Gazebo Classic with the omni bot spawned
- Start `robot_state_publisher`
- Open RViz2 with a pre-configured view (Fixed Frame: `odom`, odometry arrow, and marker display)

### 2. Run the navigation controller (in a new terminal)

```bash
source ~/omni_ws/install/setup.bash
ros2 run omni_bot_gazebo omni_controller.py
```

You'll be prompted for:
1. **Target position** — enter `x y` (space-separated), or press Enter for a random target
2. **Max linear velocity** (m/s) — press Enter for default `0.5`
3. **Wheel noise standard deviation** (rad/s) — press Enter for default `0.05`

Then type `start` to begin navigation. The bot will drive toward the target while:
- Printing live wheel velocities (ideal vs. noisy) for all 3 wheels
- Publishing a **red sphere** marker at the target
- Publishing a **green arrow** marker showing the current movement vector toward the target, visible live in RViz2

## Package structure

```
omni_ws/
└── src/
    ├── omni_bot_description/     # URDF/xacro robot model
    │   └── urdf/omni_bot.urdf.xacro
    └── omni_bot_gazebo/          # Launch files, controller, RViz config
        ├── launch/spawn_omni.launch.py
        ├── scripts/omni_controller.py
        ├── rviz/omni_bot.rviz
        └── worlds/
```

## Notes

- The robot uses the `libgazebo_ros_planar_move` Gazebo plugin for holonomic motion — this bypasses physically simulating omni-wheel roller contacts (which are notoriously unstable in ODE) and instead directly maps `cmd_vel` (vx, vy, ω) to base motion. Wheel velocities are computed analytically from this and are for visualization/logging purposes, not physical actuation.
- Odometry frame is `odom`; the robot base frame is `base_link`.

## License

MIT 
