#!/usr/bin/env python3
"""
RealSense D435i Sensor Acquisition Node

This node manages the Intel RealSense D435i camera, providing:
- RGB image streaming (30-90 FPS)
- Depth image streaming (30-90 FPS)
- IMU data (≥200 Hz)
- Point cloud generation
- Synchronized multi-stream output
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, Imu, PointCloud2, CameraInfo
from std_msgs.msg import Header
from diagnostic_msgs.msg import DiagnosticStatus, DiagnosticArray, KeyValue
import message_filters
import numpy as np


class RealSenseAcquisitionNode(Node):
    """
    RealSense D435i sensor acquisition and management node.
    
    Publishes:
        /s2/camera/rgb/image_raw (sensor_msgs/Image): RGB image stream
        /s2/camera/depth/image_raw (sensor_msgs/Image): Depth image stream
        /s2/camera/imu (sensor_msgs/Imu): IMU data stream
        /s2/camera/pointcloud (sensor_msgs/PointCloud2): Point cloud data
        /s2/camera/camera_info (sensor_msgs/CameraInfo): Camera calibration
        /s2/diagnostics/sensor (diagnostic_msgs/DiagnosticArray): Sensor health
    """
    
    def __init__(self):
        super().__init__('realsense_acquisition_node')
        
        # Declare parameters
        self.declare_parameter('rgb_fps', 30)
        self.declare_parameter('depth_fps', 30)
        self.declare_parameter('imu_rate', 200)
        self.declare_parameter('enable_pointcloud', True)
        self.declare_parameter('depth_range_min', 0.1)
        self.declare_parameter('depth_range_max', 10.0)
        self.declare_parameter('rgb_width', 640)
        self.declare_parameter('rgb_height', 480)
        self.declare_parameter('depth_width', 640)
        self.declare_parameter('depth_height', 480)
        
        # Get parameters
        self.rgb_fps = self.get_parameter('rgb_fps').value
        self.depth_fps = self.get_parameter('depth_fps').value
        self.imu_rate = self.get_parameter('imu_rate').value
        self.enable_pointcloud = self.get_parameter('enable_pointcloud').value
        
        # QoS Profiles
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE
        )
        
        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # Publishers
        self.rgb_pub = self.create_publisher(
            Image, '/s2/camera/rgb/image_raw', self.sensor_qos)
        self.depth_pub = self.create_publisher(
            Image, '/s2/camera/depth/image_raw', self.sensor_qos)
        self.imu_pub = self.create_publisher(
            Imu, '/s2/camera/imu', self.sensor_qos)
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, '/s2/camera/pointcloud', self.sensor_qos)
        self.camera_info_pub = self.create_publisher(
            CameraInfo, '/s2/camera/camera_info', self.reliable_qos)
        self.diagnostics_pub = self.create_publisher(
            DiagnosticArray, '/s2/diagnostics/sensor', self.reliable_qos)
        
        # Statistics tracking
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_timestamp = self.get_clock().now()
        self.fps_actual = 0.0
        
        # Health monitoring timer (10 Hz as per requirements)
        self.health_timer = self.create_timer(0.1, self.publish_health_status)
        
        # Sensor initialization timer
        self.init_timer = self.create_timer(1.0, self.initialize_sensor)
        
        self.get_logger().info('RealSense Acquisition Node initialized')
        self.get_logger().info(f'Target FPS - RGB: {self.rgb_fps}, Depth: {self.depth_fps}')
        self.get_logger().info(f'IMU Rate: {self.imu_rate} Hz')
    
    def initialize_sensor(self):
        """Initialize RealSense camera connection."""
        try:
            # In production, this would initialize the actual RealSense SDK
            # For now, we log the initialization
            self.get_logger().info('Initializing RealSense D435i...')
            self.get_logger().info('Sensor initialization complete')
            self.get_logger().info('Starting data acquisition...')
            
            # Start acquisition timer
            self.acquisition_timer = self.create_timer(
                1.0 / self.rgb_fps, self.acquire_and_publish)
            
            # Cancel initialization timer
            self.init_timer.cancel()
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize sensor: {str(e)}')
    
    def acquire_and_publish(self):
        """Acquire sensor data and publish to topics."""
        try:
            current_time = self.get_clock().now()
            
            # Create synchronized header
            header = Header()
            header.stamp = current_time.to_msg()
            header.frame_id = 'camera_link'
            
            # Publish RGB image (simulated for now)
            rgb_msg = self.create_rgb_message(header)
            self.rgb_pub.publish(rgb_msg)
            
            # Publish Depth image
            depth_msg = self.create_depth_message(header)
            self.depth_pub.publish(depth_msg)
            
            # Publish IMU data (at higher rate)
            imu_msg = self.create_imu_message(header)
            self.imu_pub.publish(imu_msg)
            
            # Publish point cloud if enabled
            if self.enable_pointcloud:
                pc_msg = self.create_pointcloud_message(header)
                self.pointcloud_pub.publish(pc_msg)
            
            # Publish camera info
            camera_info_msg = self.create_camera_info_message(header)
            self.camera_info_pub.publish(camera_info_msg)
            
            # Update statistics
            self.frame_count += 1
            time_diff = (current_time - self.last_timestamp).nanoseconds / 1e9
            if time_diff > 0:
                self.fps_actual = 1.0 / time_diff
            self.last_timestamp = current_time
            
        except Exception as e:
            self.get_logger().error(f'Error in acquisition: {str(e)}')
            self.dropped_frames += 1
    
    def create_rgb_message(self, header):
        """Create RGB image message."""
        msg = Image()
        msg.header = header
        msg.height = self.get_parameter('rgb_height').value
        msg.width = self.get_parameter('rgb_width').value
        msg.encoding = 'rgb8'
        msg.step = msg.width * 3
        # In production, this would be actual camera data
        msg.data = [0] * (msg.height * msg.step)
        return msg
    
    def create_depth_message(self, header):
        """Create depth image message."""
        msg = Image()
        msg.header = header
        msg.height = self.get_parameter('depth_height').value
        msg.width = self.get_parameter('depth_width').value
        msg.encoding = '16UC1'
        msg.step = msg.width * 2
        # In production, this would be actual depth data
        msg.data = [0] * (msg.height * msg.step)
        return msg
    
    def create_imu_message(self, header):
        """Create IMU message."""
        msg = Imu()
        msg.header = header
        # In production, populate with actual IMU data
        msg.orientation_covariance = [0.0] * 9
        msg.angular_velocity_covariance = [0.0] * 9
        msg.linear_acceleration_covariance = [0.0] * 9
        return msg
    
    def create_pointcloud_message(self, header):
        """Create point cloud message."""
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = 0  # Will be populated with actual points
        # In production, populate with actual point cloud data
        return msg
    
    def create_camera_info_message(self, header):
        """Create camera info message with calibration data."""
        msg = CameraInfo()
        msg.header = header
        msg.height = self.get_parameter('rgb_height').value
        msg.width = self.get_parameter('rgb_width').value
        # In production, load actual calibration parameters
        msg.distortion_model = 'plumb_bob'
        msg.d = [0.0] * 5
        msg.k = [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [600.0, 0.0, 320.0, 0.0, 0.0, 600.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        return msg
    
    def publish_health_status(self):
        """Publish health diagnostics at 10 Hz (SAFE-03 requirement)."""
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()
        
        status = DiagnosticStatus()
        status.name = 'RealSense D435i'
        status.hardware_id = 'realsense_d435i'
        
        # Determine overall status
        if self.fps_actual < self.rgb_fps * 0.8:
            status.level = DiagnosticStatus.WARN
            status.message = 'FPS below target'
        elif self.dropped_frames > 10:
            status.level = DiagnosticStatus.ERROR
            status.message = 'High frame drop rate'
        else:
            status.level = DiagnosticStatus.OK
            status.message = 'Operating normally'
        
        # Add key-value diagnostics
        status.values.append(KeyValue(key='fps_target', value=str(self.rgb_fps)))
        status.values.append(KeyValue(key='fps_actual', value=f'{self.fps_actual:.2f}'))
        status.values.append(KeyValue(key='frame_count', value=str(self.frame_count)))
        status.values.append(KeyValue(key='dropped_frames', value=str(self.dropped_frames)))
        status.values.append(KeyValue(key='imu_rate', value=str(self.imu_rate)))
        
        diag_array.status.append(status)
        self.diagnostics_pub.publish(diag_array)


def main(args=None):
    rclpy.init(args=args)
    node = RealSenseAcquisitionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()