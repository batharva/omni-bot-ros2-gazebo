import os
import xacro

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    # ---------------------------------------------------------
    # Package paths
    # ---------------------------------------------------------
    gazebo_ros_pkg = get_package_share_directory("gazebo_ros")
    description_pkg = get_package_share_directory("omni_bot_description")
    gazebo_pkg = get_package_share_directory("omni_bot_gazebo")

    # ---------------------------------------------------------
    # World
    # ---------------------------------------------------------
    world_file = os.path.join(
        gazebo_pkg,
        "worlds",
        "box_touch.world"
    )

    # Fail early if the world path is wrong.
    if not os.path.isfile(world_file):
        raise FileNotFoundError(
            f"Box world not found: {world_file}"
        )

    # ---------------------------------------------------------
    # Start Gazebo
    #
    # Important:
    # Do NOT start Gazebo paused and do NOT use a second world.
    # gazebo.launch.py receives exactly one world file here.
    # ---------------------------------------------------------
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                gazebo_ros_pkg,
                "launch",
                "gazebo.launch.py"
            )
        ),
        launch_arguments={
            "world": world_file,
            "verbose": "true",
        }.items(),
    )

    # ---------------------------------------------------------
    # Process robot XACRO
    # ---------------------------------------------------------
    xacro_file = os.path.join(
        description_pkg,
        "urdf",
        "omni_bot.urdf.xacro"
    )

    if not os.path.isfile(xacro_file):
        raise FileNotFoundError(
            f"XACRO file not found: {xacro_file}"
        )

    robot_description = xacro.process_file(
        xacro_file
    ).toxml()

    # ---------------------------------------------------------
    # Robot State Publisher
    # ---------------------------------------------------------
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description
            }
        ],
    )

    # ---------------------------------------------------------
    # Spawn robot AFTER Gazebo starts
    #
    # spawn_entity waits for the Gazebo spawn service.
    # ---------------------------------------------------------
    spawn_robot = Node(
    package="gazebo_ros",
    executable="spawn_entity.py",
    name="spawn_omni_bot",
    output="screen",
    arguments=[
        "-topic",
        "robot_description",
        "-entity",
        "omni_bot",

        # ROBOT START POSITION
        "-x", "0.0",
        "-y", "0.0",
        "-z", "0.05",

        # ROBOT START ORIENTATION
        "-Y", "0.0",
    ],
)

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_robot,
    ])
