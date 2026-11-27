"""roslab.decode against rosbags on random messages.   Run:  python -m unittest discover -s tests -v"""
import random
import sys
import unittest
from pathlib import Path

import numpy as np
from rosbags.typesys import Stores, get_typestore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roslab import decode  # noqa: E402

TS = get_typestore(Stores.ROS1_NOETIC)
T = TS.types
RNG = random.Random(0)


def header():
    frame = "".join(RNG.choice("abcxyz_/0") for _ in range(RNG.randint(0, 40)))
    return T["std_msgs/msg/Header"](seq=RNG.randint(0, 2**32 - 1),
                                    stamp=T["builtin_interfaces/msg/Time"](sec=RNG.randint(0, 2**31), nanosec=RNG.randint(0, 999_999_999)),
                                    frame_id=frame)


def f8(n):
    return np.array([RNG.uniform(-1e3, 1e3) for _ in range(n)])


class TestDecode(unittest.TestCase):
    def test_navsatfix(self):
        for _ in range(200):
            m = T["sensor_msgs/msg/NavSatFix"](
                header=header(), status=T["sensor_msgs/msg/NavSatStatus"](status=RNG.randint(-1, 2), service=RNG.randint(0, 15)),
                latitude=RNG.uniform(-90, 90), longitude=RNG.uniform(-180, 180), altitude=RNG.uniform(-100, 5000),
                position_covariance=f8(9), position_covariance_type=RNG.randint(0, 3))
            buf = TS.serialize_ros1(m, "sensor_msgs/msg/NavSatFix")
            ref = TS.deserialize_ros1(buf, "sensor_msgs/msg/NavSatFix")
            self.assertEqual(decode.navsatfix(memoryview(buf)),
                             (ref.latitude, ref.longitude, ref.altitude, ref.status.status))

    def test_imu(self):
        V3, Q = T["geometry_msgs/msg/Vector3"], T["geometry_msgs/msg/Quaternion"]
        for _ in range(200):
            q, w, a = f8(4), f8(3), f8(3)
            m = T["sensor_msgs/msg/Imu"](
                header=header(), orientation=Q(x=q[0], y=q[1], z=q[2], w=q[3]), orientation_covariance=f8(9),
                angular_velocity=V3(x=w[0], y=w[1], z=w[2]), angular_velocity_covariance=f8(9),
                linear_acceleration=V3(x=a[0], y=a[1], z=a[2]), linear_acceleration_covariance=f8(9))
            buf = TS.serialize_ros1(m, "sensor_msgs/msg/Imu")
            r = TS.deserialize_ros1(buf, "sensor_msgs/msg/Imu")
            ref = (r.orientation.x, r.orientation.y, r.orientation.z, r.orientation.w,
                   r.angular_velocity.x, r.angular_velocity.y, r.angular_velocity.z,
                   r.linear_acceleration.x, r.linear_acceleration.y, r.linear_acceleration.z)
            self.assertEqual(decode.imu(bytes(buf)), ref)

    def test_field_names(self):
        self.assertEqual(len(decode.IMU_FIELDS), 10)
        self.assertEqual(decode.NAVSAT_FIELDS, ("lat", "lon", "alt", "status"))


if __name__ == "__main__":
    unittest.main()
