from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    package_dir = get_package_share_directory("custom_dog_description")
    urdf_path = os.path.join(package_dir, "urdf", "custom_dog.urdf")
    use_rviz = LaunchConfiguration("use_rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_rviz", default_value="true"),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[{"robot_description": Command(["xacro ", urdf_path])}],
            ),
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                condition=IfCondition(use_rviz),
            ),
        ]
    )
