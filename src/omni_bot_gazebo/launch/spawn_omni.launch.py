import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_description = get_package_share_directory('omni_bot_description')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        )
    )

    xacro_file = os.path.join(pkg_description, 'urdf', 'omni_bot.urdf.xacro')
    robot_description_config = xacro.process_file(xacro_file)
    robot_desc = robot_description_config.toxml()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_desc}]
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'omni_bot'],
        output='screen'
    )

    # rviz_node = Node(
    #     package='rviz2',
    #     executable='rviz2',
    #     arguments=['-d', os.path.join(
    #         get_package_share_directory('omni_bot_gazebo'), 'rviz', 'omni_bot.rviz')],
    #     output='screen'
    # )

    # FIXED: rviz_node was missing from this list, so it never actually launched
    return LaunchDescription([gazebo, robot_state_publisher, spawn_entity])