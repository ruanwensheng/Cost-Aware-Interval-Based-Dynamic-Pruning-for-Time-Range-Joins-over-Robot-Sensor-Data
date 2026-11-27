"""Convert the IMU and GNSS topics of a bag to Parquet, once (approach 2).

    python tools/convert.py --bag 5 [--ell 10] [--limit-s N] [--bag-file F] [--out data]

Writes (under --out):
    parquet/imu.parquet    t_ns, t, qx, qy, qz, qw, wx, wy, wz, ax, ay, az
    parquet/gnss.parquet   t_ns, t, lat, lon, alt, status
    meta/gnss_stats_bag<N>.csv   j, t0, t1, lat_min, lat_max, lon_min, lon_max (one row per GNSS bucket)
and appends one line to results/convert/convert_log.jsonl (convert_once, rows, row groups, peak RSS).

t = (t_ns - bag.start_ns) / 1e9, exactly as the bag engine computes it (bag.start_ns = start of the
first chunk). Row group j holds the rows with j*ell <= t < (j+1)*ell, in both files (common grid);
empty buckets have no row group. Chunks are read one at a time, and a bucket is written as soon as
no later chunk can add rows to it, so RAM does not grow with the bag size.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roslab import decode  # noqa: E402
from roslab.bagfile import BagFile, iter_messages  # noqa: E402
from roslab.config import RESULTS_DIR, ROOT, bag_path, topics  # noqa: E402
from roslab.queries import COLUMNS  # noqa: E402
from roslab.storage import is_outlier  # noqa: E402

SCHEMAS = {role: pa.schema([("t_ns", pa.int64())] + [(c, pa.float64()) for c in COLUMNS[role][1:]])
           for role in ("imu", "gnss")}
SCHEMAS["gnss"] = SCHEMAS["gnss"].set(SCHEMAS["gnss"].get_field_index("status"), pa.field("status", pa.int8()))


class PeakRSS:
    """Samples the RSS of this process every 50 ms in a thread."""

    def __init__(self) -> None:
        self.peak, self._stop = 0, threading.Event()
        self._p = psutil.Process()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._p.memory_info().rss)
            self._stop.wait(0.05)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self._p.memory_info().rss)


def bucket_of(t: np.ndarray, ell: float) -> np.ndarray:
    """j with j*ell <= t < (j+1)*ell, also when t/ell rounds across a boundary."""
    j = np.floor(t / ell).astype(np.int64)
    j -= (j * ell > t)
    j += ((j + 1) * ell <= t)
    return j


def convert(bag_file: Path, out: Path, ell: float = 10.0, limit_s: float | None = None, bag_no: int = 5,
            log: Path | None = RESULTS_DIR / "convert" / "convert_log.jsonl") -> dict:
    t_start = time.perf_counter()
    tp = topics()
    with PeakRSS() as mem, BagFile(bag_file) as bag:
        t_open = time.perf_counter() - t_start
        role_of = {k: "imu" for k in bag.conns_for_topic(tp["imu"])}
        role_of.update({k: "gnss" for k in bag.conns_for_topic(tp["gnss"])})
        dec = {k: decode.DECODERS[bag.connections[k].msgtype][0] for k in role_of}
        chunks = [c for c in bag.chunks if role_of.keys() & c.counts.keys()]
        if limit_s is not None:
            s0 = bag.start_ns              # property = min over all chunks: compute once (was O(n^2), ~150 s)
            chunks = [c for c in chunks if (c.start_ns - s0) / 1e9 < limit_s]

        (out / "parquet").mkdir(parents=True, exist_ok=True)
        (out / "meta").mkdir(parents=True, exist_ok=True)
        writers = {r: pq.ParquetWriter(out / "parquet" / f"{r}.parquet", SCHEMAS[r], compression="snappy",
                                       write_statistics=True) for r in ("imu", "gnss")}
        pending = {r: {} for r in writers}           # role -> bucket j -> list of (t_ns, values)
        n_rows = {r: 0 for r in writers}
        n_rg = {r: 0 for r in writers}
        stats = []
        nbytes = 0

        def flush(upto: float) -> None:
            """Write every pending bucket j with (j+1)*ell <= upto, in order of j."""
            for r, buckets in pending.items():
                for j in sorted(b for b in buckets if (b + 1) * ell <= upto):
                    rows = buckets.pop(j)
                    t_ns = np.array([x[0] for x in rows], dtype=np.int64)
                    vals = np.array([x[1] for x in rows], dtype=float).reshape(len(rows), -1)
                    o = np.argsort(t_ns, kind="stable")
                    t_ns, vals = t_ns[o], vals[o]
                    t = (t_ns - bag.start_ns) / 1e9                 # same expression as the bag engine
                    if limit_s is not None:
                        keep = t < limit_s
                        t_ns, vals, t = t_ns[keep], vals[keep], t[keep]
                        if not len(t):
                            continue
                    cols = {"t_ns": t_ns, "t": t}
                    for i, c in enumerate(COLUMNS[r][2:]):
                        cols[c] = vals[:, i].astype(np.int8) if c == "status" else vals[:, i]
                    writers[r].write_table(pa.table(cols, schema=SCHEMAS[r]), row_group_size=len(t))
                    n_rows[r] += len(t)
                    n_rg[r] += 1
                    if r == "gnss":
                        stats.append((j, j * ell, (j + 1) * ell, cols["lat"].min(), cols["lat"].max(),
                                      cols["lon"].min(), cols["lon"].max()))

        t0 = time.perf_counter()
        for i, c in enumerate(chunks):
            flush((c.start_ns - bag.start_ns) / 1e9)                # no later chunk starts earlier
            ch, raw = bag.read_chunk(c)
            nbytes += ch.data_len
            for conn, t_ns, data in iter_messages(raw, set(role_of)):
                j = int(bucket_of(np.array([(np.int64(t_ns) - bag.start_ns) / 1e9]), ell)[0])
                pending[role_of[conn]].setdefault(j, []).append((t_ns, dec[conn](data)))
            if i % 500 == 0 or i == len(chunks) - 1:
                el = time.perf_counter() - t0
                print(f"\r  {i + 1}/{len(chunks)} chunks, {nbytes / 1e6:.0f} MB, {el:.0f} s, "
                      f"{el / (i + 1) * 1e3:.1f} ms/chunk, ETA {el / (i + 1) * (len(chunks) - i - 1) / 60:.1f} min   ",
                      end="", flush=True)
        flush(np.inf)
        t_loop = time.perf_counter() - t0
        print()
        for w in writers.values():
            w.close()
        stats_path = out / "meta" / f"gnss_stats_bag{bag_no}.csv"
        import pandas as pd
        pd.DataFrame(stats, columns=["j", "t0", "t1", "lat_min", "lat_max", "lon_min", "lon_max"]).to_csv(
            stats_path, index=False, float_format="%.17g")
        start_ns, chunks_total = bag.start_ns, len(bag.chunks)

    row = {"bag": bag_no, "bag_file": bag_file.name, "ell": ell, "limit_s": limit_s,
           "convert_once": time.perf_counter() - t_start, "t_open": t_open, "outlier": is_outlier(t_open),
           "t_loop": t_loop,
           "chunks_read": len(chunks), "chunks_total": chunks_total,
           "bytes_read": nbytes, "ms_per_chunk": t_loop / max(len(chunks), 1) * 1e3,
           "rows_imu": n_rows["imu"], "rows_gnss": n_rows["gnss"], "rg_imu": n_rg["imu"], "rg_gnss": n_rg["gnss"],
           "size_imu": (out / "parquet" / "imu.parquet").stat().st_size,
           "size_gnss": (out / "parquet" / "gnss.parquet").stat().st_size,
           "start_ns": start_ns, "peak_rss": mem.peak, "python": platform.python_version(),
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=int, default=5)
    ap.add_argument("--ell", type=float, default=10.0, help="row group length (s)")
    ap.add_argument("--limit-s", type=float, default=None, help="only the first N seconds (trial)")
    ap.add_argument("--bag-file", type=Path, default=None, help="bag path (default: from config/bags.yaml)")
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    a = ap.parse_args()
    row = convert(a.bag_file or bag_path(a.bag), a.out, a.ell, a.limit_s, a.bag)
    for k in ("convert_once", "t_open", "t_loop", "ms_per_chunk", "chunks_read", "chunks_total", "rows_imu", "rows_gnss",
              "rg_imu", "rg_gnss", "size_imu", "size_gnss", "peak_rss"):
        print(f"{k:13s} {row[k]}")


if __name__ == "__main__":
    main()
