"""Time the start of a bag read, step by step, in a fresh process (cold and warm).

    python tools/bench_open.py --bag 5 [--repeat 3] [--evict-gb 12]

Each measured run is a NEW Python process (as a convert or a query would be), which times:
    t_import     process start -> imports done (numpy, pyarrow, roslab)
    t_open       BagFile(path): open + read the bag header + read the whole index
    t_read1      read the compressed bytes of the first chunk with IMU or GNSS (as convert does)
    t_dec1       bz2-decompress that chunk
    t_read2      same for the next such chunk (is the first read special?)
    t_writer     create a ParquetWriter in a new directory under results/bench/tmp_open
The parent also records t_process (wall time of the whole child process).

cold: roslab.storage.drop_cache("ssd"): $ROSLAB_DROP_CACHE_CMD if set ("cold"), else read the whole
      evict file data/evict.bin ("evict", tools/make_evict_file.py), else nothing ("unknown").
warm: the same child run again right after, without eviction.
Output: results/bench/open_cold.csv (one row per run).
"""
from __future__ import annotations

import time

T_PROC0 = time.time()

import argparse  # noqa: E402
import bz2  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402,F401
import psutil  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_read import read_chunk  # noqa: E402
from roslab.bagfile import BagFile  # noqa: E402
from roslab.config import RESULTS_DIR, bag_path, topics  # noqa: E402
from roslab.storage import drop_cache  # noqa: E402

TMP = RESULTS_DIR / "bench" / "tmp_open"


def child(path: Path) -> dict:
    t_import = time.time() - psutil.Process().create_time()
    t0 = time.perf_counter()
    with BagFile(path) as bag:
        t_open = time.perf_counter() - t0
        tp = topics()
        want = set(bag.conns_for_topic(tp["imu"])) | set(bag.conns_for_topic(tp["gnss"]))
        c1, c2 = [c for c in bag.chunks if want & c.counts.keys()][:2]
    out = {"t_import": t_import, "t_open": t_open}
    with open(path, "rb") as f:
        for name, c in (("1", c1), ("2", c2)):
            t0 = time.perf_counter()
            _, dlen, data = read_chunk(f, c.chunk_pos, with_data=True)
            t1 = time.perf_counter()
            bz2.decompress(data)
            out[f"t_read{name}"], out[f"t_dec{name}"] = t1 - t0, time.perf_counter() - t1
            out[f"bytes{name}"] = dlen
    d = TMP / f"run_{time.time_ns()}"
    t0 = time.perf_counter()
    d.mkdir(parents=True)
    w = pq.ParquetWriter(d / "x.parquet", pa.schema([("t", pa.float64())]))
    w.write_table(pa.table({"t": [0.0]}))
    w.close()
    out["t_writer"] = time.perf_counter() - t0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "bench" / "open_cold.csv")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    path = bag_path(a.bag)
    if a.child:
        print(json.dumps(child(path)))
        return

    import pandas as pd
    rows = []
    for i in range(a.repeat):
        for first in (True, False):
            if first:
                t0 = time.perf_counter()
                cache = drop_cache("ssd")
                method = f"drop_cache('ssd') -> {cache} ({time.perf_counter() - t0:.0f} s)"
            else:
                cache, method = "warm", "right after the previous run"
            t0 = time.perf_counter()
            p = subprocess.run([sys.executable, __file__, "--bag", str(a.bag), "--child"],
                               capture_output=True, text=True, check=True)
            r = {"run": i, "cache": cache, "t_process": time.perf_counter() - t0, **json.loads(p.stdout),
                 "cache_method": method, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
            rows.append(r)
            print(f"run {i} {cache}: process {r['t_process']:.2f} s | import {r['t_import']:.2f} | "
                  f"open+index {r['t_open']:.3f} | read1 {r['t_read1'] * 1e3:.1f} ms dec1 {r['t_dec1'] * 1e3:.0f} ms | "
                  f"read2 {r['t_read2'] * 1e3:.1f} ms | writer {r['t_writer'] * 1e3:.1f} ms   [{method}]", flush=True)
    shutil.rmtree(TMP, ignore_errors=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
