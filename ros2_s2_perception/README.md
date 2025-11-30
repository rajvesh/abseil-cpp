# S2 Sensory and Perception Management System

A comprehensive ROS2-based perception system for mobile robotics, providing Visual SLAM, obstacle detection, sensor fusion, and health monitoring capabilities.

## Overview

The S2 Perception System is designed for autonomous mobile robots operating in dynamic indoor environments. It integrates multiple perception modalities to provide robust localization, mapping, and obstacle avoidance capabilities.

### Key Features

- **Visual SLAM**: RGB-D + IMU based SLAM with loop closure (±5cm accuracy)
- **Obstacle Detection**: VLM-based classification with 3D tracking and risk scoring
- **Sensor Fusion**: Extended Kalman Filter (EKF) fusing SLAM and IMU at >100 Hz
- **Health Monitoring**: Real-time diagnostics with 10 Hz safety heartbeat
- **RViz Visualization**: Comprehensive visualization of all perception outputs

## System Architecture

### Subsystems

1. **Sensor Acquisition** - RealSense D435i camera management
   - RGB streaming @ 30-90 FPS
   - Depth streaming @ 30-90 FPS
   - IMU data @ ≥200 Hz
   - Point cloud generation

2. **Visual SLAM** - Pose estimation and mapping
   - 6-DOF pose estimation
   - Sparse 3D map (>1000 landmarks)
   - Loop closure detection
   - <50ms processing latency

3. **Obstacle Detection** - VLM-based perception
   - Dense point clouds @ ≥15 FPS
   - Semantic segmentation (human, tool, bin, floor, wall)
   - Dynamic obstacle tracking
   - Risk scoring
   - <200ms processing latency

4. **Sensor Fusion** - EKF-based state estimation
   - 16-state EKF (position, velocity, orientation, biases)
   - >100 Hz output rate
   - Handles sensor dropouts
   - Covariance estimation

5. **Health Monitoring** - System diagnostics
   - CPU/GPU/temperature metrics @ 1 Hz
   - Safety heartbeat @ 10 Hz
   - Error logging and reporting
   - Subsystem health aggregation

## Requirements

### Hardware
- Intel RealSense D435i camera
- Computing platform with:
  - 8+ GB RAM
  - 4+ CPU cores
  - Optional: NVIDIA GPU for VLM acceleration

### Software
- Ubuntu 22.04 (recommended)
- ROS2 Humble or Iron
- Python 3.10+
- OpenCV
- NumPy

## Installation

### 1. Clone Repository

```bash
cd /path/to/workspace
git clone <repository-url>
cd abseil-cpp/ros2_s2_perception
```

### 2. Run Setup Script

```bash
cd src/s2_perception/scripts
chmod +x setup.sh
./setup.sh
```

This will install all required dependencies.

### 3. Build Workspace

```bash
cd ros2_s2_perception
colcon build --symlink-install
```

### 4. Source Workspace

```bash
source install/setup.bash
```

## Usage

### Quick Start

Use the provided launch script:

```bash
cd ros2_s2_perception
chmod +x src/s2_perception/scripts/run.sh
./src/s2_perception/scripts/run.sh
```

Or launch directly:

```bash
ros2 launch s2_perception s2_perception.launch.py
```

### Launch Parameters

```bash
# Launch without RViz
ros2 launch s2_perception s2_perception.launch.py use_rviz:=false

# Disable VLM (faster, less accurate)
ros2 launch s2_perception s2_perception.launch.py enable_vlm:=false

# Adjust camera frame rates
ros2 launch s2_perception s2_perception.launch.py rgb_fps:=60 depth_fps:=60

# Disable obstacle tracking
ros2 launch s2_perception s2_perception.launch.py enable_tracking:=false
```

## Topics

### Published Topics

#### Sensor Data
- `/s2/camera/rgb/image_raw` (sensor_msgs/Image) - RGB images
- `/s2/camera/depth/image_raw` (sensor_msgs/Image) - Depth images
- `/s2/camera/imu` (sensor_msgs/Imu) - IMU data
- `/s2/camera/pointcloud` (sensor_msgs/PointCloud2) - Point cloud
- `/s2/camera/camera_info` (sensor_msgs/CameraInfo) - Camera calibration

#### SLAM Outputs
- `/s2/slam/odometry` (nav_msgs/Odometry) - 6-DOF pose @ ≥30 Hz
- `/s2/slam/map` (sensor_msgs/PointCloud2) - Sparse 3D map
- `/s2/slam/path` (nav_msgs/Path) - Trajectory history
- `/s2/slam/keyframes` (sensor_msgs/PointCloud2) - Keyframe positions

#### Obstacle Detection
- `/s2/obstacles/detections` (vision_msgs/Detection3DArray) - 3D detections
- `/s2/obstacles/pointcloud` (sensor_msgs/PointCloud2) - Dense point cloud
- `/s2/obstacles/markers` (visualization_msgs/MarkerArray) - Visualization markers
- `/s2/obstacles/semantic_map` (sensor_msgs/Image) - Semantic segmentation

#### Sensor Fusion
- `/s2/fusion/odometry` (nav_msgs/Odometry) - Fused state @ >100 Hz
- `/s2/fusion/pose` (geometry_msgs/PoseWithCovarianceStamped) - Fused pose

#### Health & Diagnostics
- `/s2/health/system_status` (diagnostic_msgs/DiagnosticArray) - Aggregated health
- `/s2/health/heartbeat` (std_msgs/Bool) - Safety heartbeat @ 10 Hz
- `/s2/health/metrics` (diagnostic_msgs/DiagnosticArray) - System metrics
- `/s2/diagnostics/sensor` (diagnostic_msgs/DiagnosticArray) - Sensor health
- `/s2/diagnostics/slam` (diagnostic_msgs/DiagnosticArray) - SLAM health
- `/s2/diagnostics/obstacles` (diagnostic_msgs/DiagnosticArray) - Obstacle detection health
- `/s2/diagnostics/fusion` (diagnostic_msgs/DiagnosticArray) - Fusion health

## Performance Specifications

| Metric | Target | Description |
|--------|--------|-------------|
| SLAM Pose Accuracy | ≤5 cm | Indoor positioning error |
| SLAM Latency | <50 ms | End-to-end processing time |
| Obstacle Detection FPS | ≥15 FPS | Dense point cloud generation |
| Obstacle Detection Latency | <200 ms | VLM inference + tracking |
| Fusion Output Rate | >100 Hz | Fused state estimation |
| Safety Heartbeat | 10 Hz | System alive signal |
| Metrics Collection | 1 Hz | CPU/GPU/temperature |
| Detection Recall | >95% | Obstacle detection accuracy |

## Configuration

### Sensor Parameters

Edit `launch/s2_perception.launch.py` or pass as launch arguments:

```python
parameters=[{
    'rgb_fps': 30,              # RGB frame rate
    'depth_fps': 30,            # Depth frame rate
    'imu_rate': 200,            # IMU sample rate (Hz)
    'enable_pointcloud': True,  # Generate point clouds
}]
```

### SLAM Parameters

```python
parameters=[{
    'target_fps': 30,                   # SLAM processing rate
    'max_landmarks': 1000,              # Maximum map landmarks
    'keyframe_distance': 0.5,           # Keyframe spacing (m)
    'loop_closure_enabled': True,       # Enable loop closure
    'pose_accuracy_threshold': 0.05,    # 5 cm accuracy target
}]
```

### Obstacle Detection Parameters

```python
parameters=[{
    'target_fps': 15,                   # Detection rate
    'detection_threshold': 0.5,         # Confidence threshold
    'enable_tracking': True,            # Track dynamic objects
    'enable_vlm': True,                 # Use VLM for classification
}]
```

## Development

### Project Structure

```
ros2_s2_perception/
├── src/
│   └── s2_perception/
│       ├── s2_perception/              # Python package
│       │   ├── sensor_acquisition/     # RealSense driver
│       │   ├── visual_slam/            # SLAM implementation
│       │   ├── obstacle_detection/     # Obstacle detection & VLM
│       │   ├── sensor_fusion/          # EKF fusion
│       │   └── health_monitoring/      # Diagnostics
│       ├── launch/                     # Launch files
│       ├── config/                     # Configuration files
│       ├── rviz/                       # RViz configurations
│       ├── scripts/                    # Utility scripts
│       ├── CMakeLists.txt
│       └── package.xml
└── README.md
```

### Adding New Nodes

1. Create node file in appropriate module
2. Add executable to `CMakeLists.txt`
3. Update launch file if needed
4. Document in README

### Testing

```bash
# Run all tests
colcon test

# Run specific package tests
colcon test --packages-select s2_perception

# View test results
colcon test-result --all
```

## Troubleshooting

### Camera Not Detected

```bash
# Check RealSense connection
rs-enumerate-devices

# Verify permissions
sudo usermod -a -G video $USER
# Log out and back in
```

### High CPU Usage

- Reduce camera frame rates
- Disable VLM: `enable_vlm:=false`
- Reduce SLAM target FPS

### SLAM Drift

- Ensure good lighting conditions
- Check for sufficient visual features
- Verify IMU calibration
- Enable loop closure

### Low Detection Recall

- Increase `detection_threshold`
- Verify depth data quality
- Check lighting conditions
- Enable VLM for better classification

## Integration with S3 (Mobility Controller)

The S2 system provides the following outputs for S3:

- **Pose**: 6-DOF position and orientation @ ≥30 Hz
- **Velocity**: Linear and angular velocity estimates
- **Obstacle Map**: 3D detections with risk scores
- **Free Space**: Navigable area estimation
- **Semantic Flags**: Object classifications (human, obstacle, etc.)

Subscribe to `/s2/fusion/odometry` and `/s2/obstacles/detections` in your S3 controller.

## Requirements Compliance

This implementation meets all specified requirements:

- ✅ SLAM-01 to SLAM-05: Visual SLAM with RGB-D + IMU
- ✅ PC-01 to PC-05: Perception and point cloud processing
- ✅ CTRL-01 to CTRL-04: Outputs to S3 mobility controller
- ✅ COM-01 to COM-04: ROS2 communication with QoS
- ✅ SAFE-01 to SAFE-05: Safety and fault handling
- ✅ DIAG-01 to DIAG-04: Health monitoring and diagnostics
- ✅ Performance targets: Latency, FPS, accuracy

## License

Apache 2.0

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## Support

For issues and questions:
- GitHub Issues: <repository-url>/issues
- Documentation: <repository-url>/wiki

## Authors

S2 Perception Team

## Acknowledgments

- Intel RealSense SDK
- ROS2 Community
- NVIDIA Isaac ROS