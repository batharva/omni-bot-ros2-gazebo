cmake_minimum_required(VERSION 3.8)
project(omni_bot_gazebo)

find_package(ament_cmake REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(gazebo_plugins REQUIRED)

install(DIRECTORY
  launch
  worlds
  rviz
  models
  DESTINATION share/${PROJECT_NAME}/
)

install(PROGRAMS
  scripts/omni_controller.py
  DESTINATION lib/${PROJECT_NAME}
)


ament_package()