"""Small synthetic bag for the engine tests (bz2, ~64 KB chunks, 120 s).

Topics (same names as bag 5, config/topics.yaml):
    /realsense/imu              200 Hz, az ~ 9.81 with a few shocks
    /reach_1/fix                5 Hz, the robot drives back and forth along x (0..100 m, period 40 s),
                                so a strip of x is visited several times
    /realsense/color/image_raw  15 Hz, 150x150 rgb8 = 67.5 KB > chunk threshold, so about one image per
                                chunk (~67 ms): every chunk holds IMU, only ~1/3 of them hold GNSS

make_fake_bag() writes the bag and returns the values that were written, so a test can compute
the expected answer without any engine.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

T0_NS = 1_700_000_000 * 10**9
DURATION = 120.0
LAT0, LON0 = -33.04, -60.88
M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON = np.cos(np.deg2rad(LAT0)) * 111320.0
SHOCKS = (30.0, 75.5, 101.2)          # s, +5 m/s^2 on az for 3 samples
IMG = 150                             # image side (pixels)


def x_of_t(t):
    """Triangle wave: 0 -> 100 m in 20 s, back in 20 s."""
    ph = np.mod(t, 40.0)
    return np.where(ph < 20, ph * 5.0, (40 - ph) * 5.0)


def box_x(x0: float, x1: float, y0: float = -5.0, y1: float = 5.0) -> dict:
    """lat/lon box of a rectangle in metres (x east, y north) around (LAT0, LON0)."""
    return {"lat_min": LAT0 + y0 / M_PER_DEG_LAT, "lat_max": LAT0 + y1 / M_PER_DEG_LAT,
            "lon_min": LON0 + x0 / M_PER_DEG_LON, "lon_max": LON0 + x1 / M_PER_DEG_LON}


def make_fake_bag(path: Path, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ts = get_typestore(Stores.ROS1_NOETIC)
    T = ts.types
    Header, Time = T["std_msgs/msg/Header"], T["builtin_interfaces/msg/Time"]
    V3, Q = T["geometry_msgs/msg/Vector3"], T["geometry_msgs/msg/Quaternion"]
    cov = np.zeros(9)

    def hdr(t_ns, frame):
        return Header(seq=0, stamp=Time(sec=t_ns // 10**9, nanosec=t_ns % 10**9), frame_id=frame)

    # ---- values
    t_imu = np.arange(0, DURATION, 0.005)
    acc = np.column_stack([rng.normal(0, 0.05, len(t_imu)), rng.normal(0, 0.05, len(t_imu)),
                           9.81 + rng.normal(0, 0.05, len(t_imu))])
    for s in SHOCKS:
        i = int(round(s / 0.005))
        acc[i:i + 3, 2] += 5.0
    gyro = rng.normal(0, 0.01, (len(t_imu), 3))
    t_gnss = np.arange(0, DURATION, 0.2)
    x, y = x_of_t(t_gnss), rng.normal(0, 0.05, len(t_gnss))
    lat, lon = LAT0 + y / M_PER_DEG_LAT, LON0 + x / M_PER_DEG_LON
    t_img = np.arange(0, DURATION, 1 / 15)

    imu = pd.DataFrame({"t_ns": T0_NS + np.round(t_imu * 1e9).astype(np.int64),
                        "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                        "wx": gyro[:, 0], "wy": gyro[:, 1], "wz": gyro[:, 2],
                        "ax": acc[:, 0], "ay": acc[:, 1], "az": acc[:, 2]})
    gnss = pd.DataFrame({"t_ns": T0_NS + np.round(t_gnss * 1e9).astype(np.int64),
                         "lat": lat, "lon": lon, "alt": 25.0, "status": 2})

    # ---- write, in time order
    events = sorted([(int(v), 0, i) for i, v in enumerate(imu["t_ns"])]
                    + [(int(v), 1, i) for i, v in enumerate(gnss["t_ns"])]
                    + [(T0_NS + int(round(t * 1e9)), 2, i) for i, t in enumerate(t_img)])
    img_base = (np.arange(IMG * IMG * 3) % 251).astype(np.uint8)
    iv, gv = imu.to_numpy(), gnss.to_numpy()          # rows as plain arrays (iloc is slow)
    path.unlink(missing_ok=True)
    w = Writer(path)
    w.set_compression(Writer.CompressionFormat.BZ2)
    w.chunk_threshold = 64 * 1024
    with w:
        c_imu = w.add_connection("/realsense/imu", "sensor_msgs/msg/Imu", typestore=ts)
        c_fix = w.add_connection("/reach_1/fix", "sensor_msgs/msg/NavSatFix", typestore=ts)
        c_img = w.add_connection("/realsense/color/image_raw", "sensor_msgs/msg/Image", typestore=ts)
        for t_ns, kind, i in events:
            if kind == 0:
                _, qx, qy, qz, qw, wx, wy, wz, ax, ay, az = (float(v) for v in iv[i])
                m = T["sensor_msgs/msg/Imu"](
                    header=hdr(t_ns, "imu"), orientation=Q(x=qx, y=qy, z=qz, w=qw),
                    orientation_covariance=cov, angular_velocity=V3(x=wx, y=wy, z=wz),
                    angular_velocity_covariance=cov, linear_acceleration=V3(x=ax, y=ay, z=az),
                    linear_acceleration_covariance=cov)
                w.write(c_imu, t_ns, ts.serialize_ros1(m, c_imu.msgtype))
            elif kind == 1:
                _, la, lo, al, _ = gv[i]
                m = T["sensor_msgs/msg/NavSatFix"](
                    header=hdr(t_ns, "gps"), status=T["sensor_msgs/msg/NavSatStatus"](status=2, service=1),
                    latitude=float(la), longitude=float(lo), altitude=float(al), position_covariance=cov,
                    position_covariance_type=0)
                w.write(c_fix, t_ns, ts.serialize_ros1(m, c_fix.msgtype))
            else:
                data = img_base.copy()
                data[rng.integers(0, len(data), 400)] = rng.integers(0, 255, 400, dtype=np.uint8)
                m = T["sensor_msgs/msg/Image"](header=hdr(t_ns, "cam"), height=IMG, width=IMG, encoding="rgb8",
                                               is_bigendian=0, step=3 * IMG, data=data)
                w.write(c_img, t_ns, ts.serialize_ros1(m, c_img.msgtype))
    for df in (imu, gnss):
        df.insert(1, "t", (df["t_ns"] - T0_NS) / 1e9)       # = (t_ns - bag.start_ns) / 1e9
    return {"imu": imu, "gnss": gnss}


_CACHE: dict = {}


def cached_fake_bag(key: str) -> tuple[Path, Path, dict[str, pd.DataFrame]]:
    """Write the fake bag once per test process, at <tmp root>/<key> (key like "bags/x.bag").
    Returns (root dir, bag path, written values). The directory is removed at exit."""
    if not _CACHE:
        import atexit
        import shutil
        import tempfile

        root = Path(tempfile.mkdtemp(prefix="roslab_fake_"))
        atexit.register(shutil.rmtree, root, ignore_errors=True)
        path = root.joinpath(*key.split("/"))
        path.parent.mkdir(parents=True)
        _CACHE.update(root=root, path=path, data=make_fake_bag(path))
    return _CACHE["root"], _CACHE["path"], _CACHE["data"]


def gnss_stats(gnss: pd.DataFrame, ell: float) -> pd.DataFrame:
    """Per ell-second bucket j: [t0, t1) and the lat/lon box (format of data/meta/gnss_stats_bag<N>.csv)."""
    j = np.floor(gnss["t"] / ell).astype(int)
    g = gnss.groupby(j).agg(lat_min=("lat", "min"), lat_max=("lat", "max"),
                            lon_min=("lon", "min"), lon_max=("lon", "max"))
    g.insert(0, "t1", (g.index + 1) * ell)
    g.insert(0, "t0", g.index * ell)
    return g.rename_axis("j").reset_index()


def expected(spec, data: dict[str, pd.DataFrame], w0: float, w1: float) -> pd.DataFrame:
    """Brute-force answer of a test: target rows in [w0, w1] within tol of a cond row that satisfies
    the condition and lies in [w0 - tol, w1 + tol]."""
    c, t = data[spec.cond], data[spec.target]
    c = c[(c["t"] >= w0 - spec.tol) & (c["t"] <= w1 + spec.tol)]
    g = c["t"].to_numpy()[spec.cond_mask(c)]
    tt = t["t"].to_numpy()
    ok = np.zeros(len(tt), bool)
    for v in g:
        ok |= (tt >= v - spec.tol) & (tt <= v + spec.tol)
    ok &= (tt >= w0) & (tt <= w1)
    return t[ok]
