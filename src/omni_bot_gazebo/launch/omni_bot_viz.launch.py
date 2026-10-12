#!/usr/bin/env python3
"""
Full launch file for the omni-bot: spawns the robot in Gazebo, starts
robot_state_publisher, launches the omni-bot controller node, and opens
RViz2 pre-loaded with the alignment/deviation visualization config.

Run with:
    ros2 launch <your_package_name> omni_bot_full.launch.py

If you haven't wired the controller script up as a proper ROS2 entry point
yet (i.e. `ros2 run <pkg> omni_controller` doesn't work), swap the
controller_node Node() below for the ExecuteProcess() alternative that's
commented out — it just runs the .py file directly with python3.
"""

import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # --- Package names — adjust if any of these differ in your repo ---
    description_package = 'omni_bot_description'   # holds the URDF/xacro
    controller_package = 'omni_bot_gazebo'          # holds controller + rviz config
    controller_executable = 'omni_controller.py'    # matches `ros2 run omni_bot_gazebo omni_controller.py`
    rviz_config_filename = 'omni_bot.rviz'          # lives in <controller_package>/rviz/

    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_description = get_package_share_directory(description_package)

    # --- Launch Gazebo itself ---
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        )
    )

    # --- Process the robot's xacro/URDF ---
    xacro_file = os.path.join(pkg_description, 'urdf', 'omni_bot.urdf.xacro')
    robot_description_config = xacro.process_file(xacro_file)
    robot_desc = robot_description_config.toxml()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_desc}],
    )

    # --- Spawn the robot into the running Gazebo world ---
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'omni_bot'],
        output='screen',
    )

    # --- Your omni-bot controller node ---
    controller_node = Node(
        package=controller_package,
        executable=controller_executable,
        name='omni_controller',
        output='screen',
    )

    # --- Alternative if you don't have a ROS2 entry point set up yet: ---
    # controller_node = ExecuteProcess(
    #     cmd=['python3', '/absolute/path/to/omni_controller_theta.py'],
    #     output='screen',
    # )

    # --- RViz2, pre-loaded with your alignment/deviation config ---
    try:
        rviz_config_path = os.path.join(
            get_package_share_directory(controller_package), 'rviz', rviz_config_filename
        )
    except Exception:
        # Fallback: assume it's sitting next to this launch file.
        rviz_config_path = os.path.join(os.path.dirname(__file__), rviz_config_filename)

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen',
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
        controller_node,
        rviz_node,
    ])