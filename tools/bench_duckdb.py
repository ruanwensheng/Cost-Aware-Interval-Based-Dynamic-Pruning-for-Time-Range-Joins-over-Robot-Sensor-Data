"""Measure the DuckDB constants of the cost function for one layout: c_q0 and c_int.

    python tools/bench_duckdb.py --layout ssd [--n 30] [--seed 0] [--metadata-cache]

c_query = median time of  SELECT count(*) FROM read_parquet(gnss) WHERE t BETWEEN -2 AND -1
          (matches no row group: the cost of a query that reads only the footer). New connection
          for every run; the time starts once the connection is ready, as in pq_engine.
c_q0    = c_query - c_req   (c_req of the layout: read_costs.csv for ssd, remote summary for lan)
c_int   = slope of the median time of k = 1..8 empty UNION ALL branches on imu.parquet
          (each branch: WHERE t BETWEEN -2 AND -1), least squares on the k medians.
The k values are visited in a random order in every round, so a slow drift does not bias the slope.
One warm-up run before measuring: the footers are in the OS cache (cache = warm, written in the
cache and note columns).
--metadata-cache (diagnosis of c_int, not the default): SET parquet_metadata_cache = true on each
connection, so the k branches of one query can share the parsed footer; output goes to
duckdb_<layout>_metacache.csv, which layout_constants() never reads.

Appends to results/bench/duckdb_<layout>.csv (const, value in s, ...): c_query, c_q0, c_int, and
one row per k (const = union_k<k>). layout_constants() uses the latest c_q0 / c_int rows.
"""
import argparse
import platform
import random
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roslab.config import RESULTS_DIR  # noqa: E402
from roslab.storage import duckdb_connect, layout_constants, parquet_uri  # noqa: E402

EMPTY = "t BETWEEN -2 AND -1"
KS = range(1, 9)


SETTINGS: list[str] = []          # extra SET statements on every connection (--metadata-cache)


def timed(layout: str, sql: str) -> float:
    con = duckdb_connect(layout)
    for s in SETTINGS:
        con.execute(s)
    t0 = time.perf_counter()
    con.execute(sql).fetchall()
    dt = time.perf_counter() - t0
    con.close()
    return dt


def union_sql(uri: str, k: int) -> str:
    return ("SELECT count(*) FROM (" + " UNION ALL ".join(
        f"SELECT t FROM read_parquet('{uri}') WHERE {EMPTY}" for _ in range(k)) + ")")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", default="ssd")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metadata-cache", action="store_true",
                    help="diagnosis only: SET parquet_metadata_cache = true on every connection; "
                         "writes duckdb_<layout>_metacache.csv (never read by layout_constants)")
    a = ap.parse_args()
    if a.metadata_cache:
        SETTINGS.append("SET parquet_metadata_cache = true")

    c_req = layout_constants(a.layout, need_duckdb=False).c_req
    g_uri, i_uri = parquet_uri(a.layout, "parquet/gnss.parquet"), parquet_uri(a.layout, "parquet/imu.parquet")
    q_sql = f"SELECT count(*) FROM read_parquet('{g_uri}') WHERE {EMPTY}"
    con = duckdb_connect(a.layout)
    if con.execute(q_sql).fetchone()[0] != 0:
        sys.exit(f"the filter {EMPTY} matches rows of gnss.parquet")
    con.close()
    for k in KS:                                             # warm-up
        timed(a.layout, union_sql(i_uri, k))

    q = [timed(a.layout, q_sql) for _ in range(a.n)]
    rng = random.Random(a.seed)
    tk = {k: [] for k in KS}
    for _ in range(a.n):
        order = list(KS)
        rng.shuffle(order)
        for k in order:
            tk[k].append(timed(a.layout, union_sql(i_uri, k)))
    med = {k: float(np.median(v)) for k, v in tk.items()}
    slope, icpt = np.polyfit(list(med), list(med.values()), 1)
    c_query = float(np.median(q))

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    meta = {"unit": "s", "n": a.n, "cache": "warm", "layout": a.layout,
            "settings": "; ".join(SETTINGS) or "default",
            "note": "warm: one warm-up run first, footers in the OS cache; new connection per run",
            "duckdb": duckdb.__version__, "python": platform.python_version(), "timestamp": stamp}
    rows = [{"const": "c_query", "value": c_query, **meta},
            {"const": "c_q0", "value": c_query - c_req, **meta},
            {"const": "c_int", "value": float(slope), **meta}]
    rows += [{"const": f"union_k{k}", "value": v, **meta} for k, v in med.items()]
    out = RESULTS_DIR / "bench" / f"duckdb_{a.layout}{'_metacache' if a.metadata_cache else ''}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if out.exists():                              # append; older rows may lack the newer columns
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)

    print(f"c_query   {c_query * 1e3:.3f} ms  (median of {a.n}; p10 {np.percentile(q, 10) * 1e3:.3f}, "
          f"p90 {np.percentile(q, 90) * 1e3:.3f})")
    print(f"c_req     {c_req * 1e3:.3f} ms  (layout {a.layout})")
    print(f"c_q0      {(c_query - c_req) * 1e3:.3f} ms")
    print("k branches -> median ms: " + ", ".join(f"{k}: {v * 1e3:.2f}" for k, v in med.items()))
    print(f"c_int     {slope * 1e3:.3f} ms per branch  (intercept {icpt * 1e3:.3f} ms)")
    print(f"appended to {out}")


if __name__ == "__main__":
    main()
