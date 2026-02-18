#!/usr/bin/env python3
"""
ROS2 Bag-to-Bag Converter for Water Linked Sonar 3D-15.

Reads raw sonar data from a ROS2 bag (UInt8MultiArray on /sonar_3d/raw_data_multibyte)
or from a sonar recording file, decodes the RIP1/Protobuf packets, and writes a new
ROS2 bag with:
  - /sonar_3d/point_cloud  (sensor_msgs/msg/PointCloud2)
  - /sonar_3d/range_image  (sensor_msgs/msg/Image)

REQUIREMENTS:
  - ROS2 (Humble or later)
  - rosbag2_py
  - protobuf (pip install protobuf)
  - sonar_3d_15_protocol_pb2.py (generated from the Water Linked .proto file)
  - numpy, cv_bridge

USAGE:
  # Convert a sonar recording file to a ROS2 bag:
  ros2 run sonar3d sonar_bag_converter --mode file_to_bag --file <sonar_recording_file>

  # Convert raw data in a ROS2 bag to decoded point cloud / image bag:
  ros2 run sonar3d sonar_bag_converter --mode bag_to_bag --rosbag <input_bag_dir>

  # Convert raw sonar multibyte bag to decoded bag:
  ros2 run sonar3d sonar_bag_converter --mode multibyte_bag_to_bag --rosbag <input_bag_dir>
"""

import os
import sys
import struct
import zlib
import math
import argparse
from enum import Enum
from datetime import datetime, timezone

import numpy as np

# ROS2 imports
import rclpy
from rclpy.serialization import serialize_message, deserialize_message
from std_msgs.msg import UInt8MultiArray
from sensor_msgs.msg import Image, PointCloud2, PointField
from builtin_interfaces.msg import Time
import rosbag2_py
from rosidl_runtime_py.utilities import get_message

from cv_bridge import CvBridge

from sonar_3d_15_protocol_pb2 import (
    Packet,
    BitmapImageGreyscale8,
    RangeImage,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SONAR3D_FRAME = "sonar_3d"
SONAR_RAW_DATA_TOPIC = f"/{SONAR3D_FRAME}/raw_data_multibyte"
SONAR_RANGE_IMAGE_TOPIC = f"/{SONAR3D_FRAME}/range_image"
SONAR_POINT_CLOUD_TOPIC = f"/{SONAR3D_FRAME}/point_cloud"

YEAR_CHECK = 2023  # year check
TIME_FORMAT = "%Y-%m-%d-%H%M%S"  # firmware >= 1.5.0
RAW_DATA_FILE_PREFIX = "sonar-recording-"  # firmware >= 1.5.0


class Mode(Enum):
    FILE_TO_BAG = "file_to_bag"
    BAG_TO_BAG = "bag_to_bag"
    MULTIBYTE_BAG_TO_BAG = "multibyte_bag_to_bag"


# ──────────────────────────────────────────────────────────────────────────────
# RIP1 / Protobuf helpers 
# ──────────────────────────────────────────────────────────────────────────────
def parse_rip1_packet(data: bytes) -> bytes | None:
    """Parse the RIP1 framing and return the protobuf payload, or None."""
    if len(data) < 13:
        return None
    if data[:4] != b"RIP1":
        return None
    total_length = struct.unpack("<I", data[4:8])[0]
    if len(data) < total_length:
        return None
    payload = data[8 : total_length - 4]
    crc_received = struct.unpack("<I", data[total_length - 4 : total_length])[0]
    crc_calculated = zlib.crc32(data[:total_length - 4]) & 0xFFFFFFFF
    if crc_calculated != crc_received:
        return None
    return payload

def decode_protobuf_packet(payload: bytes):
    """
    Decode a Protobuf Packet.
    Returns (type_name, message_object) or None.
    """
    packet = Packet()
    try:
        packet.ParseFromString(payload)
    except Exception:
        return None

    any_msg = packet.msg
    if not any_msg.IsInitialized():
        return None

    bmp = BitmapImageGreyscale8()
    if any_msg.Unpack(bmp):
        return ("BitmapImageGreyscale8", bmp)

    rng = RangeImage()
    if any_msg.Unpack(rng):
        return ("RangeImage", rng)

    return ("Unknown", any_msg)


# ──────────────────────────────────────────────────────────────────────────────
# Conversion helpers
# ──────────────────────────────────────────────────────────────────────────────
def _sec_to_ros2_time(epoch_sec: float) -> Time:
    """Convert epoch seconds to builtin_interfaces/Time."""
    t = Time()
    t.sec = int(epoch_sec)
    t.nanosec = int((epoch_sec - int(epoch_sec)) * 1e9)
    return t


def range_image_to_pointcloud2(ri, stamp: Time, frame_id: str) -> PointCloud2:
    """
    Convert a RangeImage protobuf message to sensor_msgs/msg/PointCloud2.

    Fields: x, y, z, yaw, pitch, distance   (all FLOAT32)
    """
    max_px = ri.width - 1
    max_py = ri.height - 1
    fov_h = math.radians(ri.fov_horizontal)
    fov_v = math.radians(ri.fov_vertical)

    fields = [
        PointField(name="x",        offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y",        offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z",        offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name="yaw",      offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name="pitch",    offset=16, datatype=PointField.FLOAT32, count=1),
        PointField(name="distance", offset=20, datatype=PointField.FLOAT32, count=1),
    ]
    point_step = 24  # 6 × 4 bytes

    points_data = bytearray()
    valid_count = 0

    for px in range(ri.width):
        for py in range(ri.height):
            pv = ri.image_pixel_data[py * ri.width + px]
            if pv == 0:
                continue

            yaw_rad = (px / max_px) * fov_h - fov_h / 2
            pitch_rad = (py / max_py) * fov_v - fov_v / 2
            dist = pv * ri.image_pixel_scale

            x = dist * math.cos(pitch_rad) * math.cos(yaw_rad)
            y = dist * math.cos(pitch_rad) * math.sin(yaw_rad)
            z = -dist * math.sin(pitch_rad)

            points_data += struct.pack("<ffffff", x, y, z, yaw_rad, pitch_rad, dist)
            valid_count += 1

    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = valid_count
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.row_step = point_step * valid_count
    msg.data = bytes(points_data)
    msg.is_dense = True
    return msg


def bitmap_to_image(bmp, stamp: Time, frame_id: str) -> Image:
    """
    Convert a BitmapImageGreyscale8 protobuf message to sensor_msgs/msg/Image.
    """
    img_np = np.zeros((bmp.height, bmp.width), dtype=np.uint8)
    for y in range(bmp.height - 1, 0, -1): 
        for x in range(bmp.width):
            img_np[y, x] = bmp.image_pixel_data[y * bmp.width + x]

    bridge = CvBridge()
    ros_img = bridge.cv2_to_imgmsg(img_np, encoding="mono8")
    ros_img.header.stamp = stamp
    ros_img.header.frame_id = frame_id
    return ros_img


# ──────────────────────────────────────────────────────────────────────────────
# Packet handler     - returns (msg_type_int, ros_msg) or None
#   msg_type_int:  1 = Image (range_image topic)
#                  2 = PointCloud2 (point_cloud topic)
# ──────────────────────────────────────────────────────────────────────────────
def handle_packet(data: bytes, use_sensor_stamp: bool = True, override_stamp: Time | None = None):
    """
    Decode one RIP1 packet and return (type_int, ros2_msg) or None.
    """
    payload = parse_rip1_packet(data)
    if payload is None:
        return None

    result = decode_protobuf_packet(payload)
    if result is None:
        return None

    msg_type, msg_obj = result

    if msg_type == "BitmapImageGreyscale8":
        seq_id = msg_obj.header.sequence_id
        dt = msg_obj.header.timestamp.ToDatetime()
        print(f"  BitmapImageGreyscale8  seq={seq_id}  {msg_obj.width}x{msg_obj.height}  ts={dt.isoformat()}")
        if dt.year < YEAR_CHECK:
            return None

        if override_stamp is not None:
            stamp = override_stamp
        elif use_sensor_stamp:
            stamp = _sec_to_ros2_time(dt.timestamp())
        else:
            # Use current wall-clock
            now = datetime.now(timezone.utc).timestamp()
            stamp = _sec_to_ros2_time(now)

        ros_img = bitmap_to_image(msg_obj, stamp, SONAR3D_FRAME)
        return (1, ros_img)

    elif msg_type == "RangeImage":
        seq_id = msg_obj.header.sequence_id
        dt = msg_obj.header.timestamp.ToDatetime()
        print(f"  RangeImage  seq={seq_id}  {msg_obj.width}x{msg_obj.height}  scale={msg_obj.image_pixel_scale}  ts={dt.isoformat()}")
        if dt.year < YEAR_CHECK:
            return None

        if override_stamp is not None:
            stamp = override_stamp
        elif use_sensor_stamp:
            stamp = _sec_to_ros2_time(dt.timestamp())
        else:
            now = datetime.now(timezone.utc).timestamp()
            stamp = _sec_to_ros2_time(now)

        pc_msg = range_image_to_pointcloud2(msg_obj, stamp, SONAR3D_FRAME)
        return (2, pc_msg)

    else:
        print(f"  Unknown message type: {msg_type}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# ROS2 bag writer helper
# ──────────────────────────────────────────────────────────────────────────────
class Ros2BagWriter:
    """Thin wrapper around rosbag2_py for writing."""

    def __init__(self, output_path: str, storage_id: str = "sqlite3"):
        self._writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(uri=output_path, storage_id=storage_id)
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )
        self._writer.open(storage_options, converter_options)
        self._topics_created: set[str] = set()

    def _ensure_topic(self, topic: str, msg_type_str: str):
        if topic not in self._topics_created:
            topic_info = rosbag2_py.TopicMetadata(
                name=topic,
                type=msg_type_str,
                serialization_format="cdr",
            )
            self._writer.create_topic(topic_info)
            self._topics_created.add(topic)

    def write(self, topic: str, msg, msg_type_str: str, timestamp_ns: int):
        self._ensure_topic(topic, msg_type_str)
        self._writer.write(topic, serialize_message(msg), timestamp_ns)

    def close(self):
        del self._writer


class Ros2BagReader:
    """Thin wrapper around rosbag2_py for reading."""

    def __init__(self, bag_path: str, storage_id: str = "sqlite3"):
        self._reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )
        self._reader.open(storage_options, converter_options)

        # build topic to type map
        self._topic_type_map: dict[str, str] = {}
        for topic_info in self._reader.get_all_topics_and_types():
            self._topic_type_map[topic_info.name] = topic_info.type

    def topic_type_map(self) -> dict[str, str]:
        return dict(self._topic_type_map)

    def has_next(self) -> bool:
        return self._reader.has_next()

    def read_next(self):
        """Returns (topic, serialized_data, timestamp_ns)."""
        return self._reader.read_next()

    def get_message_type(self, topic: str):
        type_str = self._topic_type_map.get(topic)
        if type_str is None:
            return None
        return get_message(type_str)


def _stamp_to_ns(stamp: Time) -> int:
    return stamp.sec * 10**9 + stamp.nanosec


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1: sonar recording file -> ROS2 bag
# ──────────────────────────────────────────────────────────────────────────────
def file_to_bag(filename: str):
    """Read a sonar recording file and write a ROS2 bag."""
    with open(filename, "rb") as f:
        content = f.read()

    # Derive output bag path
    base = os.path.splitext(filename)[0]
    output_bag_path = base + "_ros2bag"

    print(f"Input file:  {filename}")
    print(f"Output bag:  {output_bag_path}")

    writer = Ros2BagWriter(output_bag_path)

    packets = content.split(b"RIP1")
    n_valid = 0
    n_invalid = 0

    for pkt in packets:
        if len(pkt) == 0:
            continue
        r = handle_packet(b"RIP1" + pkt, use_sensor_stamp=True)
        if r is None:
            n_invalid += 1
            continue
        n_valid += 1
        msg_type_int, ros_msg = r
        stamp_ns = _stamp_to_ns(ros_msg.header.stamp)
        if msg_type_int == 1:
            writer.write(SONAR_RANGE_IMAGE_TOPIC, ros_msg, "sensor_msgs/msg/Image", stamp_ns)
        elif msg_type_int == 2:
            writer.write(SONAR_POINT_CLOUD_TOPIC, ros_msg, "sensor_msgs/msg/PointCloud2", stamp_ns)

    writer.close()
    print(f"\nDone. Valid packets: {n_valid}, Invalid/skipped: {n_invalid}")
    print(f"Output bag written to: {output_bag_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Mode 2: ROS2 bag (with raw_data_multibyte) to new ROS2 bag with decoded topics
#         also copies all other topics
# ──────────────────────────────────────────────────────────────────────────────
def bag_to_bag(input_bag_path: str, raw_topic: str = SONAR_RAW_DATA_TOPIC):
    """
    Read a ROS2 bag that contains raw sonar data (UInt8MultiArray), decode it,
    and write a new bag with point_cloud + range_image + all other topics.
    """
    output_bag_path = input_bag_path.rstrip("/") + "_w_sonar"

    print(f"Input bag:   {input_bag_path}")
    print(f"Output bag:  {output_bag_path}")
    print(f"Raw topic:   {raw_topic}")

    reader = Ros2BagReader(input_bag_path)
    writer = Ros2BagWriter(output_bag_path)

    topic_type_map = reader.topic_type_map()
    n_sonar = 0
    n_other = 0

    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()

        if topic == raw_topic:
            # Deserialize UInt8MultiArray
            msg_cls = get_message("std_msgs/msg/UInt8MultiArray")
            raw_msg = deserialize_message(data, msg_cls)
            content = bytes(raw_msg.data)

            packets = content.split(b"RIP1")
            for pkt in packets:
                if len(pkt) == 0:
                    continue
                # Use the bag timestamp as override
                override_stamp = Time()
                override_stamp.sec = timestamp_ns // 10**9
                override_stamp.nanosec = timestamp_ns % 10**9

                r = handle_packet(b"RIP1" + pkt, override_stamp=override_stamp)
                if r is None:
                    continue
                n_sonar += 1
                msg_type_int, ros_msg = r
                stamp_ns = _stamp_to_ns(ros_msg.header.stamp)
                if msg_type_int == 1:
                    writer.write(SONAR_RANGE_IMAGE_TOPIC, ros_msg, "sensor_msgs/msg/Image", stamp_ns)
                elif msg_type_int == 2:
                    writer.write(SONAR_POINT_CLOUD_TOPIC, ros_msg, "sensor_msgs/msg/PointCloud2", stamp_ns)
        else:
            # Pass through all other topics unchanged
            type_str = topic_type_map.get(topic, "std_msgs/msg/String")
            if topic not in writer._topics_created:
                topic_info = rosbag2_py.TopicMetadata(
                    name=topic,
                    type=type_str,
                    serialization_format="cdr",
                )
                writer._writer.create_topic(topic_info)
                writer._topics_created.add(topic)
            writer._writer.write(topic, data, timestamp_ns)
            n_other += 1

    writer.close()
    print(f"\nDone. Sonar messages decoded: {n_sonar}, Other messages copied: {n_other}")
    print(f"Output bag: {output_bag_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Mode 3: Same as Mode 2 but uses sensor timestamp instead of bag timestamp
# ──────────────────────────────────────────────────────────────────────────────
def multibyte_bag_to_bag(input_bag_path: str, raw_topic: str = SONAR_RAW_DATA_TOPIC):
    """
    Same as bag_to_bag but uses the sensor-embedded timestamp from each packet
    rather than the bag-recorded timestamp.
    """
    output_bag_path = input_bag_path.rstrip("/") + "_w_sonar"

    print(f"Input bag:   {input_bag_path}")
    print(f"Output bag:  {output_bag_path}")
    print(f"Raw topic:   {raw_topic}")

    reader = Ros2BagReader(input_bag_path)
    writer = Ros2BagWriter(output_bag_path)

    topic_type_map = reader.topic_type_map()
    n_sonar = 0
    n_other = 0

    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()

        if topic == raw_topic:
            msg_cls = get_message("std_msgs/msg/UInt8MultiArray")
            raw_msg = deserialize_message(data, msg_cls)
            content = bytes(raw_msg.data)

            packets = content.split(b"RIP1")
            for pkt in packets:
                if len(pkt) == 0:
                    continue
                # Use sensor timestamp (embedded in protobuf)
                r = handle_packet(b"RIP1" + pkt, use_sensor_stamp=True)
                if r is None:
                    continue
                n_sonar += 1
                msg_type_int, ros_msg = r
                stamp_ns = _stamp_to_ns(ros_msg.header.stamp)
                if msg_type_int == 1:
                    writer.write(SONAR_RANGE_IMAGE_TOPIC, ros_msg, "sensor_msgs/msg/Image", stamp_ns)
                elif msg_type_int == 2:
                    writer.write(SONAR_POINT_CLOUD_TOPIC, ros_msg, "sensor_msgs/msg/PointCloud2", stamp_ns)
        else:
            type_str = topic_type_map.get(topic, "std_msgs/msg/String")
            if topic not in writer._topics_created:
                topic_info = rosbag2_py.TopicMetadata(
                    name=topic,
                    type=type_str,
                    serialization_format="cdr",
                )
                writer._writer.create_topic(topic_info)
                writer._topics_created.add(topic)
            writer._writer.write(topic, data, timestamp_ns)
            n_other += 1

    writer.close()
    print(f"\nDone. Sonar messages decoded: {n_sonar}, Other messages copied: {n_other}")
    print(f"Output bag: {output_bag_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ROS2 Sonar 3D-15 bag converter: decode raw sonar data into PointCloud2 + Image topics."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[m.value for m in Mode],
        help="Conversion mode: file_to_bag | bag_to_bag | multibyte_bag_to_bag",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="",
        help="Sonar recording file (for file_to_bag mode).",
    )
    parser.add_argument(
        "--rosbag",
        type=str,
        default="",
        help="Input ROS2 bag directory (for bag_to_bag / multibyte_bag_to_bag modes).",
    )
    parser.add_argument(
        "--raw-topic",
        type=str,
        default=SONAR_RAW_DATA_TOPIC,
        help=f"Topic name for raw sonar data (default: {SONAR_RAW_DATA_TOPIC}).",
    )

    args = parser.parse_args()
    mode = Mode(args.mode)

    if mode == Mode.FILE_TO_BAG:
        if not args.file:
            print("ERROR: --file is required for file_to_bag mode.")
            sys.exit(1)
        if not os.path.isfile(args.file):
            print(f"ERROR: File not found: {args.file}")
            sys.exit(1)
        file_to_bag(args.file)

    elif mode == Mode.BAG_TO_BAG:
        if not args.rosbag:
            print("ERROR: --rosbag is required for bag_to_bag mode.")
            sys.exit(1)
        if not os.path.exists(args.rosbag):
            print(f"ERROR: Bag not found: {args.rosbag}")
            sys.exit(1)
        bag_to_bag(args.rosbag, raw_topic=args.raw_topic)

    elif mode == Mode.MULTIBYTE_BAG_TO_BAG:
        if not args.rosbag:
            print("ERROR: --rosbag is required for multibyte_bag_to_bag mode.")
            sys.exit(1)
        if not os.path.exists(args.rosbag):
            print(f"ERROR: Bag not found: {args.rosbag}")
            sys.exit(1)
        multibyte_bag_to_bag(args.rosbag, raw_topic=args.raw_topic)


if __name__ == "__main__":
    main()
