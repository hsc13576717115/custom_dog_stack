from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    description_share = get_package_share_directory("custom_dog_description")
    display_launch = os.path.join(description_share, "launch", "display.launch.py")
    return LaunchDescription([IncludeLaunchDescription(PythonLaunchDescriptionSource(display_launch))])
