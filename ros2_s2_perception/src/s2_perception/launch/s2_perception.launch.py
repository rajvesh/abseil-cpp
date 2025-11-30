#!/usr/bin/env python3
"""
S2 Perception System Launch File

Launches all perception subsystems:
- RealSense sensor acquisition
- Visual SLAM
- Obstacle detection
- Sensor fusion (EKF)
- Health monitoring
- RViz visualization
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate launch description for S2 perception system."""
    
    # Declare launch arguments
    use_rviz = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz for visualization'
    )
    
    enable_vlm = DeclareLaunchArgument(
        'enable_vlm',
        default_value='true',
        description='Enable Vision Language Model for obstacle classification'
    )
    
    enable_tracking = DeclareLaunchArgument(
        'enable_tracking',
        default_value='true',
        description='Enable dynamic obstacle tracking'
    )
    
    rgb_fps = DeclareLaunchArgument(
        'rgb_fps',
        default_value='30',
        description='RGB camera frame rate'
    )
    
    depth_fps = DeclareLaunchArgument(
        'depth_fps',
        default_value='30',
        description='Depth camera frame rate'
    )
    
    # Get package share directory
    pkg_share = FindPackageShare('s2_perception').find('s2_perception')
    
    # RViz configuration file
    rviz_config_file = PathJoinSubstitution([
        FindPackageShare('s2_perception'),
        'rviz',
        's2_perception.rviz'
    ])
    
    # Node: RealSense Sensor Acquisition
    realsense_node = Node(
        package='s2_perception',
        executable='realsense_node.py',
        name='realsense_acquisition',
        output='screen',
        parameters=[{
            'rgb_fps': LaunchConfiguration('rgb_fps'),
            'depth_fps': LaunchConfiguration('depth_fps'),
            'imu_rate': 200,
            'enable_pointcloud': True,
            'depth_range_min': 0.1,
            'depth_range_max': 10.0,
            'rgb_width': 640,
            'rgb_height': 480,
            'depth_width': 640,
            'depth_height': 480,
        }]
    )
    
    # Node: Visual SLAM
    slam_node = Node(
        package='s2_perception',
        executable='slam_node.py',
        name='visual_slam',
        output='screen',
        parameters=[{
            'target_fps': 30,
            'max_landmarks': 1000,
            'keyframe_distance': 0.5,
            'loop_closure_enabled': True,
            'pose_accuracy_threshold': 0.05,  # 5 cm
            'map_update_rate': 10,
        }]
    )
    
    # Node: Obstacle Detection
    obstacle_detector_node = Node(
        package='s2_perception',
        executable='obstacle_detector_node.py',
        name='obstacle_detector',
        output='screen',
        parameters=[{
            'target_fps': 15,
            'detection_threshold': 0.5,
            'max_detection_range': 10.0,
            'enable_tracking': LaunchConfiguration('enable_tracking'),
            'enable_vlm': LaunchConfiguration('enable_vlm'),
            'semantic_classes': ['human', 'tool', 'bin', 'floor', 'wall', 'obstacle'],
        }]
    )
    
    # Node: Sensor Fusion (EKF)
    ekf_fusion_node = Node(
        package='s2_perception',
        executable='ekf_fusion_node.py',
        name='ekf_fusion',
        output='screen',
        parameters=[{
            'output_rate': 100,  # >100 Hz requirement
            'imu_rate': 200,
            'process_noise_pos': 0.01,
            'process_noise_vel': 0.1,
            'process_noise_ori': 0.01,
            'measurement_noise_slam': 0.05,
        }]
    )
    
    # Node: Health Monitoring
    health_monitor_node = Node(
        package='s2_perception',
        executable='health_monitor_node.py',
        name='health_monitor',
        output='screen',
        parameters=[{
            'heartbeat_rate': 10,  # Hz (SAFE-03)
            'metrics_rate': 1,  # Hz (DIAG-01)
            'cpu_threshold': 80.0,
            'memory_threshold': 80.0,
            'temp_threshold': 80.0,
        }]
    )
    
    # Node: RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        output='screen'
    )
    
    # Create launch description
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(use_rviz)
    ld.add_action(enable_vlm)
    ld.add_action(enable_tracking)
    ld.add_action(rgb_fps)
    ld.add_action(depth_fps)
    
    # Add nodes
    ld.add_action(realsense_node)
    ld.add_action(slam_node)
    ld.add_action(obstacle_detector_node)
    ld.add_action(ekf_fusion_node)
    ld.add_action(health_monitor_node)
    ld.add_action(rviz_node)
    
    return ld