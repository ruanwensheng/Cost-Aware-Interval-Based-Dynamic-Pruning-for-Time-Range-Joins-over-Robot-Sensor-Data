"""Fast decoders for the two message types of the queries (ROS 1 serialization, little endian).

std_msgs/Header: seq u32, stamp (sec u32, nsec u32), frame_id (u32 length + bytes).
Checked against rosbags in tests/test_decode.py.
"""
from __future__ import annotations

import struct

_U32 = struct.Struct("<I")
_NAVSAT = struct.Struct("<bH3d")          # status i8, service u16, latitude, longitude, altitude
_F8x3 = struct.Struct("<3d")
_F8x4 = struct.Struct("<4d")

IMU_FIELDS = ("qx", "qy", "qz", "qw", "wx", "wy", "wz", "ax", "ay", "az")
NAVSAT_FIELDS = ("lat", "lon", "alt", "status")


def skip_header(buf, off: int = 0) -> int:
    """Offset right after a std_msgs/Header that starts at `off`."""
    n = _U32.unpack_from(buf, off + 12)[0]          # seq (4) + stamp (8), then frame_id length
    return off + 16 + n


def navsatfix(buf) -> tuple[float, float, float, int]:
    """sensor_msgs/NavSatFix -> (lat, lon, alt, status)."""
    status, _service, lat, lon, alt = _NAVSAT.unpack_from(buf, skip_header(buf))
    return lat, lon, alt, status


def imu(buf) -> tuple[float, ...]:
    """sensor_msgs/Imu -> (qx, qy, qz, qw, wx, wy, wz, ax, ay, az)."""
    o = skip_header(buf)
    q = _F8x4.unpack_from(buf, o)                   # orientation
    w = _F8x3.unpack_from(buf, o + 32 + 72)         # after orientation_covariance (9 f8)
    a = _F8x3.unpack_from(buf, o + 32 + 72 + 24 + 72)  # after angular_velocity_covariance
    return q + w + a


DECODERS = {"sensor_msgs/NavSatFix": (navsatfix, NAVSAT_FIELDS), "sensor_msgs/Imu": (imu, IMU_FIELDS)}
