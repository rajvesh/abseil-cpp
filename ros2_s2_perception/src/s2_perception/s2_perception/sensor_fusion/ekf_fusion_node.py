#!/usr/bin/env python3
"""
Extended Kalman Filter (EKF) Sensor Fusion Node

Fuses Visual SLAM pose estimates with IMU data for robust state estimation.
Implements a 16-state EKF as per requirements.

State vector: [position(3), velocity(3), orientation(4), gyro_bias(3), accel_bias(3)]
Output rate: >100 Hz
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from diagnostic_msgs.msg import DiagnosticStatus, DiagnosticArray, KeyValue
import numpy as np


class EKFFusionNode(Node):
    """
    EKF-based sensor fusion for SLAM + IMU.
    
    Subscribes:
        /s2/slam/odometry (nav_msgs/Odometry): SLAM pose estimates
        /s2/camera/imu (sensor_msgs/Imu): IMU measurements
    
    Publishes:
        /s2/fusion/odometry (nav_msgs/Odometry): Fused state at >100 Hz
        /s2/fusion/pose (geometry_msgs/PoseWithCovarianceStamped): Fused pose
        /s2/diagnostics/fusion (diagnostic_msgs/DiagnosticArray): Fusion health
    """
    
    def __init__(self):
        super().__init__('ekf_fusion_node')
        
        # Declare parameters
        self.declare_parameter('output_rate', 100)  # Hz
        self.declare_parameter('imu_rate', 200)  # Hz
        self.declare_parameter('process_noise_pos', 0.01)
        self.declare_parameter('process_noise_vel', 0.1)
        self.declare_parameter('process_noise_ori', 0.01)
        self.declare_parameter('measurement_noise_slam', 0.05)
        
        # Get parameters
        self.output_rate = self.get_parameter('output_rate').value
        self.imu_rate = self.get_parameter('imu_rate').value
        
        # QoS Profiles
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers
        self.slam_sub = self.create_subscription(
            Odometry, '/s2/slam/odometry', 
            self.slam_callback, self.reliable_qos)
        self.imu_sub = self.create_subscription(
            Imu, '/s2/camera/imu', 
            self.imu_callback, self.sensor_qos)
        
        # Publishers
        self.odom_pub = self.create_publisher(
            Odometry, '/s2/fusion/odometry', self.reliable_qos)
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/s2/fusion/pose', self.reliable_qos)
        self.diagnostics_pub = self.create_publisher(
            DiagnosticArray, '/s2/diagnostics/fusion', self.reliable_qos)
        
        # EKF State (16-dimensional)
        # [px, py, pz, vx, vy, vz, qw, qx, qy, qz, bg_x, bg_y, bg_z, ba_x, ba_y, ba_z]
        self.state = np.zeros(16)
        self.state[6] = 1.0  # Initialize quaternion w component
        
        # State covariance matrix (16x16)
        self.P = np.eye(16) * 0.1
        
        # Process noise covariance
        self.Q = np.eye(16) * 0.01
        
        # Measurement noise covariance (SLAM)
        self.R_slam = np.eye(7) * 0.05  # position(3) + orientation(4)
        
        # IMU measurements buffer
        self.last_imu_time = None
        self.imu_initialized = False
        self.slam_initialized = False
        
        # Statistics
        self.update_count = 0
        self.slam_updates = 0
        self.imu_updates = 0
        self.slam_dropouts = 0
        self.last_slam_time = None
        
        # Output timer (>100 Hz)
        self.output_timer = self.create_timer(
            1.0 / self.output_rate, self.publish_state)
        
        # Health monitoring timer (10 Hz)
        self.health_timer = self.create_timer(0.1, self.publish_health_status)
        
        self.get_logger().info('EKF Fusion Node initialized')
        self.get_logger().info(f'Output rate: {self.output_rate} Hz')
        self.get_logger().info(f'Expected IMU rate: {self.imu_rate} Hz')
    
    def imu_callback(self, msg):
        """
        IMU prediction step.
        High-frequency updates (≥200 Hz) for state propagation.
        """
        current_time = self.get_clock().now()
        
        if not self.imu_initialized:
            self.last_imu_time = current_time
            self.imu_initialized = True
            return
        
        # Calculate dt
        dt = (current_time - self.last_imu_time).nanoseconds / 1e9
        self.last_imu_time = current_time
        
        if dt <= 0 or dt > 0.1:  # Sanity check
            return
        
        # Extract IMU measurements
        accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        
        gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        
        # EKF Prediction Step
        self.predict(accel, gyro, dt)
        
        self.imu_updates += 1
    
    def slam_callback(self, msg):
        """
        SLAM measurement update.
        Corrects drift using visual odometry.
        """
        current_time = self.get_clock().now()
        
        if not self.slam_initialized:
            # Initialize state from first SLAM measurement
            self.state[0] = msg.pose.pose.position.x
            self.state[1] = msg.pose.pose.position.y
            self.state[2] = msg.pose.pose.position.z
            self.state[6] = msg.pose.pose.orientation.w
            self.state[7] = msg.pose.pose.orientation.x
            self.state[8] = msg.pose.pose.orientation.y
            self.state[9] = msg.pose.pose.orientation.z
            self.slam_initialized = True
            self.last_slam_time = current_time
            self.get_logger().info('EKF initialized from SLAM')
            return
        
        # Check for SLAM dropouts
        if self.last_slam_time is not None:
            time_since_last = (current_time - self.last_slam_time).nanoseconds / 1e9
            if time_since_last > 0.5:  # 500ms threshold
                self.slam_dropouts += 1
                self.get_logger().warn(
                    f'SLAM dropout detected: {time_since_last:.2f}s since last update')
        
        self.last_slam_time = current_time
        
        # EKF Update Step
        measurement = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z
        ])
        
        self.update(measurement)
        
        self.slam_updates += 1
    
    def predict(self, accel, gyro, dt):
        """
        EKF prediction step using IMU.
        Propagates state and covariance forward in time.
        """
        # Extract current state
        pos = self.state[0:3]
        vel = self.state[3:6]
        quat = self.state[6:10]
        gyro_bias = self.state[10:13]
        accel_bias = self.state[13:16]
        
        # Remove biases
        gyro_corrected = gyro - gyro_bias
        accel_corrected = accel - accel_bias
        
        # Rotate acceleration to world frame (simplified - should use quaternion)
        accel_world = accel_corrected  # Placeholder
        
        # State prediction (simplified kinematic model)
        # Position: p = p + v*dt + 0.5*a*dt^2
        new_pos = pos + vel * dt + 0.5 * accel_world * dt**2
        
        # Velocity: v = v + a*dt
        new_vel = vel + accel_world * dt
        
        # Orientation: integrate angular velocity (simplified)
        # In production: use proper quaternion integration
        new_quat = quat  # Placeholder
        
        # Biases remain constant (random walk model)
        new_gyro_bias = gyro_bias
        new_accel_bias = accel_bias
        
        # Update state
        self.state[0:3] = new_pos
        self.state[3:6] = new_vel
        self.state[6:10] = new_quat
        self.state[10:13] = new_gyro_bias
        self.state[13:16] = new_accel_bias
        
        # Normalize quaternion
        quat_norm = np.linalg.norm(self.state[6:10])
        if quat_norm > 0:
            self.state[6:10] /= quat_norm
        
        # Covariance prediction: P = F*P*F' + Q
        # F is the state transition Jacobian (simplified here)
        F = np.eye(16)
        F[0:3, 3:6] = np.eye(3) * dt  # position-velocity coupling
        
        self.P = F @ self.P @ F.T + self.Q * dt
    
    def update(self, measurement):
        """
        EKF update step using SLAM measurement.
        Corrects state estimate based on visual odometry.
        """
        # Measurement model: z = H*x + v
        # We measure position and orientation directly
        H = np.zeros((7, 16))
        H[0:3, 0:3] = np.eye(3)  # Position
        H[3:7, 6:10] = np.eye(4)  # Orientation
        
        # Innovation: y = z - H*x
        predicted_measurement = H @ self.state
        innovation = measurement - predicted_measurement
        
        # Innovation covariance: S = H*P*H' + R
        S = H @ self.P @ H.T + self.R_slam
        
        # Kalman gain: K = P*H' * inv(S)
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.get_logger().warn('Singular innovation covariance matrix')
            return
        
        # State update: x = x + K*y
        self.state = self.state + K @ innovation
        
        # Normalize quaternion
        quat_norm = np.linalg.norm(self.state[6:10])
        if quat_norm > 0:
            self.state[6:10] /= quat_norm
        
        # Covariance update: P = (I - K*H)*P
        I_KH = np.eye(16) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_slam @ K.T  # Joseph form
    
    def publish_state(self):
        """Publish fused state at >100 Hz."""
        if not self.slam_initialized:
            return
        
        current_time = self.get_clock().now()
        
        # Publish as Odometry
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = 'map'
        odom.child_frame_id = 'base_link'
        
        # Position
        odom.pose.pose.position.x = self.state[0]
        odom.pose.pose.position.y = self.state[1]
        odom.pose.pose.position.z = self.state[2]
        
        # Orientation
        odom.pose.pose.orientation.w = self.state[6]
        odom.pose.pose.orientation.x = self.state[7]
        odom.pose.pose.orientation.y = self.state[8]
        odom.pose.pose.orientation.z = self.state[9]
        
        # Velocity
        odom.twist.twist.linear.x = self.state[3]
        odom.twist.twist.linear.y = self.state[4]
        odom.twist.twist.linear.z = self.state[5]
        
        # Covariance (extract relevant parts)
        pose_cov = np.zeros(36)
        pose_cov[0:3] = np.diag(self.P[0:3, 0:3])  # Position variance
        pose_cov[21:24] = np.diag(self.P[6:9, 6:9])  # Orientation variance
        odom.pose.covariance = pose_cov.tolist()
        
        self.odom_pub.publish(odom)
        
        # Also publish as PoseWithCovariance
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header = odom.header
        pose_msg.pose.pose = odom.pose.pose
        pose_msg.pose.covariance = odom.pose.covariance
        self.pose_pub.publish(pose_msg)
        
        self.update_count += 1
    
    def publish_health_status(self):
        """Publish fusion health diagnostics at 10 Hz."""
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()
        
        status = DiagnosticStatus()
        status.name = 'Sensor Fusion (EKF)'
        status.hardware_id = 'ekf_fusion'
        
        # Determine status level
        if not self.slam_initialized:
            status.level = DiagnosticStatus.WARN
            status.message = 'Waiting for SLAM initialization'
        elif self.slam_dropouts > 5:
            status.level = DiagnosticStatus.ERROR
            status.message = 'Excessive SLAM dropouts'
        elif self.last_slam_time is not None:
            time_since_slam = (self.get_clock().now() - self.last_slam_time).nanoseconds / 1e9
            if time_since_slam > 1.0:
                status.level = DiagnosticStatus.WARN
                status.message = 'SLAM updates delayed'
            else:
                status.level = DiagnosticStatus.OK
                status.message = 'Fusion operating normally'
        else:
            status.level = DiagnosticStatus.WARN
            status.message = 'No SLAM updates received'
        
        # Add diagnostics
        status.values.append(KeyValue(key='output_rate', value=str(self.output_rate)))
        status.values.append(KeyValue(key='update_count', value=str(self.update_count)))
        status.values.append(KeyValue(key='slam_updates', value=str(self.slam_updates)))
        status.values.append(KeyValue(key='imu_updates', value=str(self.imu_updates)))
        status.values.append(KeyValue(key='slam_dropouts', value=str(self.slam_dropouts)))
        
        # State uncertainties
        pos_uncertainty = np.sqrt(np.trace(self.P[0:3, 0:3]))
        vel_uncertainty = np.sqrt(np.trace(self.P[3:6, 3:6]))
        status.values.append(KeyValue(
            key='position_uncertainty_m', value=f'{pos_uncertainty:.4f}'))
        status.values.append(KeyValue(
            key='velocity_uncertainty_m_s', value=f'{vel_uncertainty:.4f}'))
        
        diag_array.status.append(status)
        self.diagnostics_pub.publish(diag_array)


def main(args=None):
    rclpy.init(args=args)
    node = EKFFusionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()