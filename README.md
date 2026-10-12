# Omni Bot — ROS 2 & Gazebo Simulation

A ROS 2-based omnidirectional robot simulation in Gazebo featuring configurable robot motion, multi-robot coordination, and a box-interaction task with real-time visualization and optional performance graphs.

<p align="center">
  <a href="https://youtu.be/h3x6rgIEqbQ">
    <img src="https://img.youtube.com/vi/h3x6rgIEqbQ/maxresdefault.jpg" alt="Omni Bot ROS 2 Gazebo Demo" width="800">
  </a>
</p>

<p align="center">
  <strong><a href="https://youtu.be/h3x6rgIEqbQ">▶ Watch the Latest Demo Video</a></strong>
  &nbsp; | &nbsp;
  <strong><a href="https://github.com/batharva/omni-bot-ros2-gazebo">GitHub Repository</a></strong>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Simulation Scenario](#simulation-scenario)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Build the Workspace](#build-the-workspace)
- [Run the Simulation](#run-the-simulation)
- [Configuration Options](#configuration-options)
- [Visualization and Analysis](#visualization-and-analysis)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [Author](#author)

## Overview

The **Omni Bot ROS 2 Gazebo Simulation** project explores omnidirectional mobile robot behavior using ROS 2 and Gazebo.

The project includes a configurable simulation in which one or two independently controlled omnidirectional robots approach a box from assigned sides. The robots are configured to bring their respective stick tips into contact with designated regions on the box faces within a specified tolerance.

The simulation combines robot modeling, motion control, configurable task parameters, and visualization to provide a framework for studying multi-robot interactions.

## Features

- **Omnidirectional robot simulation:** Simulate mobile robot motion in Gazebo.
- **Multi-robot support:** Configure one or two independently controlled robots.
- **Configurable target sides:** Assign robots to approach different faces of the box, such as `-X` and `+X`.
- **Interactive configuration:** Configure the number of robots, target sides, tolerance, velocity, and box center of mass through the launch interface.
- **Box-interaction task:** Evaluate whether each robot's stick tip reaches its assigned box-face tolerance region.
- **Real-time visualization:** Launch 2D scenario and robot-body-frame visualizations.
- **Optional performance graphs:** Enable or disable graph generation through a launch argument.
- **ROS 2 integration:** Use ROS 2 launch files and nodes to organize and execute the simulation.

## Simulation Scenario

The main simulation uses a box and one or two robots approaching their assigned target faces.

### Two-Robot Configuration

| Parameter                  | Configuration                                                            |
| -------------------------- | ------------------------------------------------------------------------ |
| Number of robots           | 2                                                                        |
| Box center of mass (X, Y)  | `(1.5, 0.8) m`                                                           |
| Box dimensions (L × W × H) | `0.50 × 0.20 × 0.10 m`                                                   |
| Example Bot 1 target       | `-X` face                                                                |
| Example Bot 2 target       | `+X` face                                                                |
| Success criterion          | Each stick tip reaches its assigned face within the configured tolerance |

The target sides and task parameters can be changed through the launch configuration.

**Mission objective:** Each robot must reach its assigned target region on the box. The success condition is based on stick-tip contact with the designated box face, not contact between the two robots.

## Technology Stack

| Technology                    | Purpose                                                                  |
| ----------------------------- | ------------------------------------------------------------------------ |
| ROS 2                         | Robot communication, launch orchestration, and control                   |
| Gazebo                        | Physics-based robot simulation                                           |
| Python                        | Simulation utilities, control logic, and visualization, where applicable |
| RViz / 2D visualization tools | Robot and scenario visualization, where configured                       |
| Git & GitHub                  | Version control and project hosting                                      |

## Prerequisites

Before running the project, make sure your system has:

- Ubuntu with a compatible ROS 2 installation.
- Gazebo and the ROS–Gazebo integration required by the project.
- `colcon` build tools.
- Git.
- The ROS 2 dependencies specified by the repository's package manifests.

> **Note:** Use the ROS 2 distribution and Gazebo version supported by the packages in this repository. Install any additional dependencies declared in `package.xml` before building.

## Installation

Clone the repository into your ROS 2 workspace:

```bash
mkdir -p ~/omni_bot_ws/src
cd ~/omni_bot_ws/src

git clone https://github.com/batharva/omni-bot-ros2-gazebo.git
```

Navigate to the workspace:

```bash
cd ~/omni_bot_ws
```

Source your ROS 2 installation. The following example uses ROS 2 Humble:

```bash
source /opt/ros/humble/setup.bash
```

If you are using a different ROS 2 distribution, replace `humble` with your installed distribution.

## Build the Workspace

Install dependencies declared by the workspace packages:

```bash
cd ~/omni_bot_ws

rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

Build the packages:

```bash
colcon build --symlink-install
```

Source the workspace:

```bash
source install/setup.bash
```

Run these commands in each new terminal that needs access to the workspace.

## Run the Simulation

### 1. Launch the Two-Robot Box Scenario

Run the main launch file:

```bash
ros2 launch omni_bot_gazebo multi_bot_box.launch.py num_bots:=2 graphs:=false
```

This launches the box-interaction scenario with two robots and disables optional graphs.

### 2. Enable Performance Graphs

To launch the scenario with graphs enabled:

```bash
ros2 launch omni_bot_gazebo multi_bot_box.launch.py num_bots:=2 graphs:=true
```

### 3. Run a Single-Robot Scenario

Set the number of robots to one:

```bash
ros2 launch omni_bot_gazebo multi_bot_box.launch.py num_bots:=1 graphs:=false
```

The launch interface and supported parameters determine the target side, velocity, tolerance, and box configuration.

## Configuration Options

The main launch file is:

```text
omni_bot_gazebo/multi_bot_box.launch.py
```

| Parameter          | Description                                       |
| ------------------ | ------------------------------------------------- |
| `num_bots`         | Select the number of robots to simulate           |
| `graphs`           | Enable or disable optional graphs                 |
| Target side        | Assign the box face that each robot must approach |
| Tolerance          | Configure the acceptable target-contact region    |
| Velocity           | Configure the robot's commanded velocity          |
| Box center of mass | Set the box position used by the scenario         |

Example launch command:

```bash
ros2 launch omni_bot_gazebo multi_bot_box.launch.py num_bots:=2 graphs:=false
```

Use the launch interface to adjust the supported task parameters.

## Visualization and Analysis

The simulation provides visualization tools to help inspect robot motion and the box-interaction task.

- **Scenario visualization:** Observe the box, robots, and target configuration in 2D.
- **Body-frame visualization:** Inspect robot-relative motion and orientation.
- **Performance graphs:** Enable optional graphs for additional analysis.
- **Task verification:** Check whether each robot's stick tip reaches its assigned box-face tolerance region.

These visualizations help make the simulation easier to debug and support further development of robot-control strategies.

## Project Structure

The repository is organized around ROS 2 packages. The following is a conceptual overview; consult the actual repository for the complete file tree.

```text
omni-bot-ros2-gazebo/
├── src/
│   ├── omni_bot_description/
│   │   └── Robot models and descriptions
│   └── omni_bot_gazebo/
│       ├── launch/
│       │   └── multi_bot_box.launch.py
│       └── Simulation and task-related files
├── README.md
└── Other configuration and build files
```

## Future Improvements

Potential extensions to the project include:

- More robust multi-robot coordination.
- Improved trajectory planning and motion control.
- Additional simulation scenarios and obstacle configurations.
- Automated success-rate measurement and performance comparison.
- Expanded visualization and experiment logging.

## Author

**Atharva Bhosale**

Mechanical Engineering | Robotics | ROS 2 | Gazebo

- **GitHub:** [@batharva](https://github.com/batharva)
- **Project Repository:** [omni-bot-ros2-gazebo](https://github.com/batharva/omni-bot-ros2-gazebo)
- **Latest Demo:** [Watch on YouTube](https://youtu.be/h3x6rgIEqbQ)

---

If you find this project useful, consider starring the repository on GitHub.
