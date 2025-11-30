#!/bin/bash
# Quick launch script for S2 Perception System

set -e

# Check if ROS2 is sourced
if [ -z "$ROS_DISTRO" ]; then
    echo "Error: ROS2 is not sourced. Please source your ROS2 installation."
    exit 1
fi

# Check if workspace is built and sourced
if [ ! -d "install" ]; then
    echo "Error: Workspace not built. Please run:"
    echo "  colcon build --symlink-install"
    exit 1
fi

# Source the workspace
source install/setup.bash

# Launch with arguments
echo "Launching S2 Perception System..."
ros2 launch s2_perception s2_perception.launch.py "$@"