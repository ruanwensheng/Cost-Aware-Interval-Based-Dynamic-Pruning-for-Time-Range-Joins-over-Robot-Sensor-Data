"""Measure the cost of reading bag chunks (input for the cost model).

    python tools/bench_read.py --bag 5 [--window 300 420] [--n 500] [--seed 0] [--evict-gb 12]

Cases (per-chunk time, median over N chunks):
    header       read only the CHUNK record header           (buffering = default and = 0)
    raw          read header + whole compressed data
    raw_bz2      raw + bz2 decompression (read and decompress timed separately)
    seq_bz2      raw_bz2 on N CONSECUTIVE chunks (file order) starting at the window start
Random cases use N chunks drawn at random inside the window, visited in file order.

Each case runs cold and warm:
    cold  run $ROSLAB_DROP_CACHE_CMD if set; otherwise evict the OS file cache by reading
          --evict-gb of the same bag OUTSIDE the window (more than the RAM size).
    warm  run the case once (not measured), then measure.

Output: results/bench/read_costs.csv (one row per case x cache, then the cost-model constants).
"""
from __future__ import annotations

import argparse
import bz2
import os
import platform
import random
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roslab.bagfile import BagFile, ChunkInfo  # noqa: E402
from roslab.config import RESULTS_DIR, bag_path  # noqa: E402

U32 = struct.Struct("<I")


def read_exact(f, n: int) -> bytes:
    buf = f.read(n)
    while len(buf) < n:                 # raw (buffering=0) reads may return less
        more = f.read(n - len(buf))
        if not more:
            raise EOFError
        buf += more
    return buf


def read_chunk(f, pos: int, with_data: bool) -> tuple[int, int, bytes | None]:
    """Read the CHUNK record at pos. Returns (bytes read, data_len, compressed data or None)."""
    f.seek(pos)
    hlen = U32.unpack(read_exact(f, 4))[0]
    read_exact(f, hlen)
    dlen = U32.unpack(read_exact(f, 4))[0]
    n = 8 + hlen
    data = None
    if with_data:
        data = read_exact(f, dlen)
        n += dlen
    return n, dlen, data


def run_case(path: Path, chunks: list[ChunkInfo], case: str, buffering: int) -> dict:
    read_t, dec_t, nbytes, comp = [], [], 0, 0
    with open(path, "rb", buffering=buffering) as f:
        for c in chunks:
            t0 = time.perf_counter()
            n, dlen, data = read_chunk(f, c.chunk_pos, with_data=case != "header")
            t1 = time.perf_counter()
            read_t.append(t1 - t0)
            nbytes += n
            if case in ("raw_bz2", "seq_bz2"):
                bz2.decompress(data)
                dec_t.append(time.perf_counter() - t1)
                comp += dlen
    r = np.array(read_t)
    tot = r + (np.array(dec_t) if dec_t else 0)
    out = {"n": len(chunks), "bytes_read": nbytes,
           "read_median_ms": np.median(r) * 1e3, "read_mean_ms": r.mean() * 1e3,
           "read_p95_ms": np.percentile(r, 95) * 1e3, "read_total_s": r.sum(),
           "read_mb_per_s": nbytes / 1e6 / r.sum(),
           "total_median_ms": np.median(tot) * 1e3, "total_s": float(np.sum(tot)),
           "_read": r, "_bytes": None}
    if dec_t:
        d = np.array(dec_t)
        out.update({"decomp_median_ms": np.median(d) * 1e3, "decomp_total_s": d.sum(),
                    "decomp_mb_per_s": comp / 1e6 / d.sum()})
    return out


def evict(bag: BagFile, w_lo: int, w_hi: int, gb: float, rr: list[int]) -> str:
    cmd = os.environ.get("ROSLAB_DROP_CACHE_CMD")
    if cmd:
        subprocess.run(cmd, shell=True, check=True)
        return f"ROSLAB_DROP_CACHE_CMD ({cmd})"
    # Read `gb` GB of the bag outside [w_lo, w_hi) in 8 MB blocks, rotating the start
    # so successive evictions do not re-read only what is already cached.
    regions = [(bag.chunks[0].chunk_pos, w_lo), (w_hi, bag.index_pos)]
    blocks = [(p, min(8 << 20, e - p)) for s, e in regions for p in range(s, e, 8 << 20)]
    need, i, done = int(gb * 1e9), rr[0], 0
    t0 = time.perf_counter()
    with open(bag.path, "rb", buffering=0) as f:
        while done < need:
            p, n = blocks[i % len(blocks)]
            f.seek(p)
            done += len(f.read(n))
            i += 1
    rr[0] = i
    return f"read {done / 1e9:.1f} GB of the bag outside the window ({time.perf_counter() - t0:.0f} s)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=int, default=5)
    ap.add_argument("--window", type=float, nargs=2, default=[300.0, 420.0])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--evict-gb", type=float, default=12.0)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "bench" / "read_costs.csv")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    path = bag_path(a.bag)
    with BagFile(path) as bag:
        lo, hi = bag.start_ns + int(a.window[0] * 1e9), bag.start_ns + int(a.window[1] * 1e9)
        win = [c for c in bag.chunks if c.end_ns >= lo and c.start_ns <= hi]
        rnd = sorted(random.Random(a.seed).sample(win, a.n), key=lambda c: c.chunk_pos)
        seq = win[:a.n]
        w_lo, w_hi = win[0].chunk_pos, bag.chunks[win[-1].idx + 1].chunk_pos
        ram = psutil.virtual_memory().total / 1e9
        print(f"{len(win)} chunks in window {a.window}; random sample {len(rnd)}, sequential {len(seq)}; "
              f"RAM {ram:.1f} GB")
        if not os.environ.get("ROSLAB_DROP_CACHE_CMD") and a.evict_gb < ram:
            print(f"!! --evict-gb {a.evict_gb} < RAM {ram:.1f} GB: cold runs may still hit the cache")

        plan = [("header", -1, rnd), ("header", 0, rnd), ("raw", -1, rnd),
                ("raw_bz2", -1, rnd), ("seq_bz2", -1, seq)]
        rows, rr = [], [0]
        for case, buf, chunks in plan:
            for cache in ("cold", "warm"):
                if cache == "cold":
                    method = evict(bag, w_lo, w_hi, a.evict_gb, rr)
                else:
                    run_case(path, chunks, case, buf)          # warm-up, not measured
                    method = "warm-up run of the same case"
                r = run_case(path, chunks, case, buf)
                row = {"case": case, "cache": cache, "buffering": "default" if buf == -1 else "0",
                       "order": "consecutive" if case == "seq_bz2" else "random, file order",
                       **{k: v for k, v in r.items() if not k.startswith("_")}, "cache_method": method}
                row["_read"] = r["_read"]
                rows.append(row)
                print(f"{case:8s} buf={row['buffering']:7s} {cache}: read median {row['read_median_ms']:.3f} ms, "
                      f"p95 {row['read_p95_ms']:.3f} ms, {row['read_mb_per_s']:.0f} MB/s"
                      + (f", bz2 median {row['decomp_median_ms']:.1f} ms = {row['decomp_mb_per_s']:.1f} MB/s"
                         if "decomp_median_ms" in row else "") + f"   [{method}]", flush=True)

        # chunk sizes of the random sample, for the linear fit time = a + bytes / B
        sizes = np.array([bag.read_chunk_header(c).data_len for c in rnd], dtype=float)

    def pick(case, cache, buf="default"):
        return next(r for r in rows if r["case"] == case and r["cache"] == cache and r["buffering"] == buf)

    consts = []
    for cache in ("cold", "warm"):
        fixed = pick("header", cache)["read_median_ms"]
        raw = pick("raw", cache)
        slope, icpt = np.polyfit(sizes, raw["_read"], 1)        # seconds per byte, seconds
        mbps = 1 / slope / 1e6 if slope > 0 else raw["read_mb_per_s"]
        consts += [{"const": "fixed_cost_per_read", "cache": cache, "value": fixed, "unit": "ms",
                    "note": "median time to read one chunk header at a random position"},
                   {"const": "read_speed", "cache": cache, "value": mbps, "unit": "MB/s",
                    "note": f"slope of read time vs chunk size (raw case); intercept {icpt * 1e3:.3f} ms"}]
    dec = [r["decomp_mb_per_s"] for r in rows if "decomp_mb_per_s" in r]
    consts.append({"const": "bz2_decompress_speed", "cache": "any", "value": float(np.median(dec)),
                   "unit": "MB/s of compressed input", "note": f"median over {len(dec)} runs (raw_bz2, seq_bz2)"})

    meta = {"bag": a.bag, "window": f"{a.window[0]:g}-{a.window[1]:g}", "seed": a.seed,
            "machine": f"{platform.processor()} / RAM {psutil.virtual_memory().total / 2**30:.1f} GiB",
            "python": platform.python_version(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    df = pd.concat([pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]),
                    pd.DataFrame(consts)], ignore_index=True)
    for k, v in meta.items():
        df[k] = v
    df.to_csv(a.out, index=False, float_format="%.4f")
    print("\ncost-model constants:")
    for c in consts:
        print(f"  {c['const']:22s} {c['cache']:5s} {c['value']:10.3f} {c['unit']}   ({c['note']})")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
