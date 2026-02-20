# Sonar 3D-15 ROS2 Driver

ROS 2 driver and bag converter for the [Water Linked Sonar 3D-15](https://www.waterlinked.com/sonar-3d-15) multibeam imaging sonar.

## Overview

This package decodes RIP1-framed protobuf packets from the Sonar 3D-15 and publishes standard ROS 2 messages:

| Topic | Message Type | Description |
|-------|-------------|-------------|
| `/sonar_3d/point_cloud` | `sensor_msgs/msg/PointCloud2` | 3D point cloud from RangeImage data |
| `/sonar_3d/range_image` | `sensor_msgs/msg/Image` | Greyscale bitmap image (mono8) |

The package provides two executables:

- **`multicast_listener`** — live ROS 2 node that subscribes to raw sonar data and publishes decoded messages in real time
- **`sonar_bag_converter`** — offline tool that reads raw sonar data from recording files or ROS 2 bags and writes new bags with decoded topics

## Prerequisites

- ROS 2 Humble / Jazzy (tested on Jazzy)
- Python 3.10+
- `protobuf` (`pip install protobuf`)
- `numpy`, `opencv-python`, `cv_bridge`

### Generating the Protobuf File

The driver requires `sonar_3d_15_protocol_pb2.py`, generated from the Water Linked `.proto` definition. If you need to regenerate it:

```bash
pip install protobuf grpcio-tools
protoc --python_out=src/sonar3d/sonar3d/ sonar-3d-15-protocol.proto
```

## Usage

### Live Listener

Subscribes to `/sonar_3d/raw_data_multibyte` (`std_msgs/msg/UInt8MultiArray`) and publishes decoded point clouds and images.

```bash
# Launch with default settings
ros2 launch sonar3d sonar3d.launch.py

# Run directly
ros2 run sonar3d multicast_listener

# Use ROS clock instead of sensor timestamp
ros2 run sonar3d multicast_listener --ros-args -p use_sensor_stamp:=false
```

### Bag Converter

Three conversion modes for offline processing:

```bash
# Sonar recording file to ROS 2 bag
ros2 run sonar3d sonar_bag_converter --mode file_to_bag --file <sonar-recording-file>

# ROS 2 bag (raw multibyte) to ROS 2 bag (decoded), using bag timestamps
ros2 run sonar3d sonar_bag_converter --mode bag_to_bag --rosbag <input-bag-dir>

# ROS 2 bag (raw multibyte) to ROS 2 bag (decoded), using sensor timestamps
ros2 run sonar3d sonar_bag_converter --mode multibyte_bag_to_bag --rosbag <input-bag-dir>
```

You can also specify a custom raw data topic:

```bash
ros2 run sonar3d sonar_bag_converter --mode bag_to_bag \
    --rosbag /path/to/bag.db3 \
    --raw-topic /tube1/sonar_raw
```

## PointCloud2 Fields

Each point contains 6 float32 fields (24 bytes per point):

| Field | Offset | Description |
|-------|--------|-------------|
| `x` | 0 | X coordinate (meters) |
| `y` | 4 | Y coordinate (meters) |
| `z` | 8 | Z coordinate (meters) |
| `yaw` | 12 | Yaw angle (radians) |
| `pitch` | 16 | Pitch angle (radians) |
| `distance` | 20 | Range distance (meters) |

## Package Structure

```
Sonar-3D-15-ROS-driver/
├── README.md
└── src/sonar3d/
    ├── launch/sonar3d.launch.py
    ├── package.xml
    ├── requirements.txt
    ├── setup.py
    ├── setup.cfg
    └── sonar3d/
        ├── __init__.py
        ├── multicast_listener.py      # Live ROS 2 node
        ├── sonar_bag_converter.py      # Offline bag converter
        └── sonar_3d_15_protocol_pb2.py # Protobuf definitions
```

## License

MIT License