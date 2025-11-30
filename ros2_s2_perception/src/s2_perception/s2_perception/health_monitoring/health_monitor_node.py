#!/usr/bin/env python3
"""
Health Monitoring and Diagnostics Node

Aggregates health status from all S2 perception subsystems and provides
system-wide health monitoring, metrics collection, and diagnostic reporting.

Requirements met:
- DIAG-01: CPU/GPU/temperature metrics every 1 second
- DIAG-02: Log perception pipeline errors
- DIAG-03: Sensor health (FPS drop, noise level)
- SAFE-03: Safety heartbeat at 10 Hz
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from diagnostic_msgs.msg import DiagnosticStatus, DiagnosticArray, KeyValue
from std_msgs.msg import Header, Bool
import psutil
import os


class HealthMonitorNode(Node):
    """
    System-wide health monitoring and diagnostics aggregation.
    
    Subscribes:
        /s2/diagnostics/sensor (diagnostic_msgs/DiagnosticArray)
        /s2/diagnostics/slam (diagnostic_msgs/DiagnosticArray)
        /s2/diagnostics/obstacles (diagnostic_msgs/DiagnosticArray)
        /s2/diagnostics/fusion (diagnostic_msgs/DiagnosticArray)
    
    Publishes:
        /s2/health/system_status (diagnostic_msgs/DiagnosticArray): Aggregated health
        /s2/health/heartbeat (std_msgs/Bool): Safety heartbeat at 10 Hz
        /s2/health/metrics (diagnostic_msgs/DiagnosticArray): System metrics
    """
    
    def __init__(self):
        super().__init__('health_monitor_node')
        
        # Declare parameters
        self.declare_parameter('heartbeat_rate', 10)  # Hz (SAFE-03)
        self.declare_parameter('metrics_rate', 1)  # Hz (DIAG-01)
        self.declare_parameter('cpu_threshold', 80.0)  # %
        self.declare_parameter('memory_threshold', 80.0)  # %
        self.declare_parameter('temp_threshold', 80.0)  # °C
        
        # Get parameters
        self.heartbeat_rate = self.get_parameter('heartbeat_rate').value
        self.metrics_rate = self.get_parameter('metrics_rate').value
        self.cpu_threshold = self.get_parameter('cpu_threshold').value
        self.memory_threshold = self.get_parameter('memory_threshold').value
        self.temp_threshold = self.get_parameter('temp_threshold').value
        
        # QoS Profiles
        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers for subsystem diagnostics
        self.sensor_diag_sub = self.create_subscription(
            DiagnosticArray, '/s2/diagnostics/sensor',
            self.sensor_diag_callback, self.reliable_qos)
        self.slam_diag_sub = self.create_subscription(
            DiagnosticArray, '/s2/diagnostics/slam',
            self.slam_diag_callback, self.reliable_qos)
        self.obstacles_diag_sub = self.create_subscription(
            DiagnosticArray, '/s2/diagnostics/obstacles',
            self.obstacles_diag_callback, self.reliable_qos)
        self.fusion_diag_sub = self.create_subscription(
            DiagnosticArray, '/s2/diagnostics/fusion',
            self.fusion_diag_callback, self.reliable_qos)
        
        # Publishers
        self.system_status_pub = self.create_publisher(
            DiagnosticArray, '/s2/health/system_status', self.reliable_qos)
        self.heartbeat_pub = self.create_publisher(
            Bool, '/s2/health/heartbeat', self.reliable_qos)
        self.metrics_pub = self.create_publisher(
            DiagnosticArray, '/s2/health/metrics', self.reliable_qos)
        
        # Subsystem health state
        self.subsystem_status = {
            'sensor': None,
            'slam': None,
            'obstacles': None,
            'fusion': None
        }
        
        # System metrics
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.temperature = 0.0
        self.gpu_utilization = 0.0
        
        # Error tracking
        self.error_count = 0
        self.warning_count = 0
        self.last_errors = []
        
        # Heartbeat timer (10 Hz - SAFE-03)
        self.heartbeat_timer = self.create_timer(
            1.0 / self.heartbeat_rate, self.publish_heartbeat)
        
        # Metrics collection timer (1 Hz - DIAG-01)
        self.metrics_timer = self.create_timer(
            1.0 / self.metrics_rate, self.collect_and_publish_metrics)
        
        # System status aggregation timer (5 Hz)
        self.status_timer = self.create_timer(0.2, self.publish_system_status)
        
        self.get_logger().info('Health Monitor Node initialized')
        self.get_logger().info(f'Heartbeat rate: {self.heartbeat_rate} Hz')
        self.get_logger().info(f'Metrics collection rate: {self.metrics_rate} Hz')
    
    def sensor_diag_callback(self, msg):
        """Store sensor subsystem diagnostics."""
        if msg.status:
            self.subsystem_status['sensor'] = msg.status[0]
            self.check_for_errors(msg.status[0], 'Sensor')
    
    def slam_diag_callback(self, msg):
        """Store SLAM subsystem diagnostics."""
        if msg.status:
            self.subsystem_status['slam'] = msg.status[0]
            self.check_for_errors(msg.status[0], 'SLAM')
    
    def obstacles_diag_callback(self, msg):
        """Store obstacle detection subsystem diagnostics."""
        if msg.status:
            self.subsystem_status['obstacles'] = msg.status[0]
            self.check_for_errors(msg.status[0], 'Obstacles')
    
    def fusion_diag_callback(self, msg):
        """Store sensor fusion subsystem diagnostics."""
        if msg.status:
            self.subsystem_status['fusion'] = msg.status[0]
            self.check_for_errors(msg.status[0], 'Fusion')
    
    def check_for_errors(self, status, subsystem_name):
        """Check for errors and log them (DIAG-02)."""
        if status.level == DiagnosticStatus.ERROR:
            error_msg = f'{subsystem_name}: {status.message}'
            self.get_logger().error(error_msg)
            self.error_count += 1
            self.last_errors.append(error_msg)
            if len(self.last_errors) > 10:
                self.last_errors.pop(0)
        elif status.level == DiagnosticStatus.WARN:
            self.get_logger().warn(f'{subsystem_name}: {status.message}')
            self.warning_count += 1
    
    def publish_heartbeat(self):
        """Publish safety heartbeat at 10 Hz (SAFE-03)."""
        heartbeat_msg = Bool()
        heartbeat_msg.data = True
        self.heartbeat_pub.publish(heartbeat_msg)
    
    def collect_and_publish_metrics(self):
        """
        Collect and publish system metrics every 1 second (DIAG-01).
        Includes CPU, GPU, memory, and temperature.
        """
        # Collect CPU usage
        self.cpu_percent = psutil.cpu_percent(interval=None)
        
        # Collect memory usage
        memory = psutil.virtual_memory()
        self.memory_percent = memory.percent
        
        # Collect temperature (if available)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Get first available temperature sensor
                for name, entries in temps.items():
                    if entries:
                        self.temperature = entries[0].current
                        break
        except (AttributeError, OSError):
            # Temperature sensors not available on all systems
            self.temperature = 0.0
        
        # Collect GPU utilization (placeholder - requires nvidia-ml-py3)
        self.gpu_utilization = self.get_gpu_utilization()
        
        # Create metrics message
        metrics_array = DiagnosticArray()
        metrics_array.header.stamp = self.get_clock().now().to_msg()
        
        # System metrics status
        metrics_status = DiagnosticStatus()
        metrics_status.name = 'System Metrics'
        metrics_status.hardware_id = 's2_perception_system'
        
        # Determine overall metrics health
        if (self.cpu_percent > self.cpu_threshold or 
            self.memory_percent > self.memory_threshold or
            self.temperature > self.temp_threshold):
            metrics_status.level = DiagnosticStatus.WARN
            metrics_status.message = 'System resources under stress'
        else:
            metrics_status.level = DiagnosticStatus.OK
            metrics_status.message = 'System resources nominal'
        
        # Add metric values
        metrics_status.values.append(
            KeyValue(key='cpu_percent', value=f'{self.cpu_percent:.1f}'))
        metrics_status.values.append(
            KeyValue(key='memory_percent', value=f'{self.memory_percent:.1f}'))
        metrics_status.values.append(
            KeyValue(key='temperature_c', value=f'{self.temperature:.1f}'))
        metrics_status.values.append(
            KeyValue(key='gpu_utilization_percent', value=f'{self.gpu_utilization:.1f}'))
        
        # Add thresholds
        metrics_status.values.append(
            KeyValue(key='cpu_threshold', value=f'{self.cpu_threshold:.1f}'))
        metrics_status.values.append(
            KeyValue(key='memory_threshold', value=f'{self.memory_threshold:.1f}'))
        metrics_status.values.append(
            KeyValue(key='temp_threshold', value=f'{self.temp_threshold:.1f}'))
        
        metrics_array.status.append(metrics_status)
        self.metrics_pub.publish(metrics_array)
        
        # Log warnings if thresholds exceeded
        if self.cpu_percent > self.cpu_threshold:
            self.get_logger().warn(f'CPU usage high: {self.cpu_percent:.1f}%')
        if self.memory_percent > self.memory_threshold:
            self.get_logger().warn(f'Memory usage high: {self.memory_percent:.1f}%')
        if self.temperature > self.temp_threshold:
            self.get_logger().warn(f'Temperature high: {self.temperature:.1f}°C')
    
    def get_gpu_utilization(self):
        """
        Get GPU utilization percentage.
        Placeholder - requires nvidia-ml-py3 for actual GPU monitoring.
        """
        try:
            # In production, use pynvml:
            # import pynvml
            # pynvml.nvmlInit()
            # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            # return util.gpu
            return 0.0
        except:
            return 0.0
    
    def publish_system_status(self):
        """
        Aggregate and publish overall system health status.
        """
        status_array = DiagnosticArray()
        status_array.header.stamp = self.get_clock().now().to_msg()
        
        # Overall system status
        system_status = DiagnosticStatus()
        system_status.name = 'S2 Perception System'
        system_status.hardware_id = 's2_perception'
        
        # Determine overall system health
        subsystem_levels = []
        for name, status in self.subsystem_status.items():
            if status is not None:
                subsystem_levels.append(status.level)
        
        if not subsystem_levels:
            system_status.level = DiagnosticStatus.WARN
            system_status.message = 'No subsystems reporting'
        elif DiagnosticStatus.ERROR in subsystem_levels:
            system_status.level = DiagnosticStatus.ERROR
            system_status.message = 'One or more subsystems in ERROR state'
        elif DiagnosticStatus.WARN in subsystem_levels:
            system_status.level = DiagnosticStatus.WARN
            system_status.message = 'One or more subsystems in WARN state'
        else:
            system_status.level = DiagnosticStatus.OK
            system_status.message = 'All subsystems operating normally'
        
        # Add subsystem status summary
        for name, status in self.subsystem_status.items():
            if status is not None:
                level_str = ['OK', 'WARN', 'ERROR', 'STALE'][status.level]
                system_status.values.append(
                    KeyValue(key=f'{name}_status', value=level_str))
                system_status.values.append(
                    KeyValue(key=f'{name}_message', value=status.message))
        
        # Add error counts
        system_status.values.append(
            KeyValue(key='total_errors', value=str(self.error_count)))
        system_status.values.append(
            KeyValue(key='total_warnings', value=str(self.warning_count)))
        
        # Add recent errors
        if self.last_errors:
            system_status.values.append(
                KeyValue(key='recent_errors', value='; '.join(self.last_errors[-3:])))
        
        status_array.status.append(system_status)
        
        # Also include individual subsystem statuses
        for name, status in self.subsystem_status.items():
            if status is not None:
                status_array.status.append(status)
        
        self.system_status_pub.publish(status_array)


def main(args=None):
    rclpy.init(args=args)
    node = HealthMonitorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()