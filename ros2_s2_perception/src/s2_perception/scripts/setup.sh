#!/bin/bash
# Setup script for S2 Perception System

set -e

echo "=========================================="
echo "S2 Perception System Setup"
echo "=========================================="

# Check if ROS2 is installed
if [ -z "$ROS_DISTRO" ]; then
    echo "Error: ROS2 is not sourced. Please source your ROS2 installation:"
    echo "  source /opt/ros/<distro>/setup.bash"
    exit 1
fi

echo "ROS2 Distribution: $ROS_DISTRO"

# Install system dependencies
echo ""
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install --user \
    numpy \
    opencv-python \
    psutil \
    transforms3d

# Install ROS2 package dependencies
echo ""
echo "Installing ROS2 package dependencies..."
sudo apt-get install -y \
    ros-${ROS_DISTRO}-realsense2-camera \
    ros-${ROS_DISTRO}-realsense2-camera-msgs \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-pcl-ros \
    ros-${ROS_DISTRO}-pcl-conversions \
    ros-${ROS_DISTRO}-vision-msgs \
    ros-${ROS_DISTRO}-diagnostic-updater \
    ros-${ROS_DISTRO}-rviz2

# Optional: Install Isaac ROS Visual SLAM (if available)
echo ""
echo "Note: Isaac ROS Visual SLAM requires NVIDIA GPU and additional setup."
echo "Visit: https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Build the workspace:"
echo "   cd ros2_s2_perception"
echo "   colcon build --symlink-install"
echo ""
echo "2. Source the workspace:"
echo "   source install/setup.bash"
echo ""
echo "3. Launch the system:"
echo "   ros2 launch s2_perception s2_perception.launch.py"