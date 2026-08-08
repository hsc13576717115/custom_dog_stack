#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
    echo "ROS 2 Humble is not installed in this environment." >&2
    exit 1
fi

# ROS 2 Humble's generated setup script reads AMENT_TRACE_SETUP_FILES before
# initializing it, which is incompatible with Bash nounset.
set +u
source /opt/ros/humble/setup.bash
set -u
cd "${project_root}"
colcon \
    --log-base log/ros2 \
    build \
    --base-paths ros2/src \
    --build-base build/ros2 \
    --install-base install/ros2 \
    --symlink-install
