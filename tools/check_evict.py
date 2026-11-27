"""Does drop_cache empty the OS cache enough? Compare runs after drop_cache with warm runs.

    python tools/check_evict.py [--layout ssd] [--test 1] [--mode auto] [--repeat 3]

Approach pq (DuckDB on the Parquet files). Order: `repeat` runs, each after drop_cache(layout)
(cache = "cold", "evict" or "unknown"); then one warm-up run (not kept) and `repeat` warm runs.
Then a direct probe: `repeat` times, drop_cache, read the whole target file unbuffered (cache state
as returned), and read it again at once (warm); rows <cache>_raw_read / warm_raw_read.
Prints the median end_to_end and t_open of each group; writes results/bench/evict_check.csv.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roslab import pq_engine  # noqa: E402
from roslab.config import RESULTS_DIR  # noqa: E402
from roslab.queries import load_tests  # noqa: E402
from roslab.storage import drop_cache, layout_constants, parquet_uri  # noqa: E402

KEEP = ["test", "mode", "level", "end_to_end", "t_open", "t_phase1", "t_phase2", "t_l2", "outlier",
        "rows_out", "result_hash"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", default="ssd")
    ap.add_argument("--test", default="1")
    ap.add_argument("--mode", default="auto")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "bench" / "evict_check.csv")
    a = ap.parse_args()
    spec = load_tests([a.test])[0]
    L = layout_constants(a.layout)

    rows = []
    for i in range(a.repeat):
        t0 = time.perf_counter()
        cache = drop_cache(a.layout)
        t_drop = time.perf_counter() - t0
        r = pq_engine.run(spec, a.layout, L, a.mode)
        rows.append({"cache": cache, "repeat_idx": i, "t_drop": t_drop, **{k: r.get(k) for k in KEEP}})
        print(f"{cache:7s} #{i}: end_to_end {r['end_to_end'] * 1e3:8.1f} ms, t_open {r['t_open'] * 1e3:6.1f} ms "
              f"(drop {t_drop:.0f} s)", flush=True)
    pq_engine.run(spec, a.layout, L, a.mode)                    # warm-up, not kept
    for i in range(a.repeat):
        r = pq_engine.run(spec, a.layout, L, a.mode)
        rows.append({"cache": "warm", "repeat_idx": i, "t_drop": None, **{k: r.get(k) for k in KEEP}})
        print(f"warm    #{i}: end_to_end {r['end_to_end'] * 1e3:8.1f} ms, t_open {r['t_open'] * 1e3:6.1f} ms",
              flush=True)

    # direct probe: raw read of the target file right after drop_cache, then again (warm)
    path = Path(parquet_uri(a.layout, f"parquet/{spec.target}.parquet"))
    for i in range(a.repeat):
        cache = drop_cache(a.layout)
        for c in (cache, "warm"):
            t0 = time.perf_counter()
            with open(path, "rb", buffering=0) as f:
                n = len(f.read())
            dt = time.perf_counter() - t0
            rows.append({"cache": f"{c}_raw_read", "repeat_idx": i, "end_to_end": dt, "rows_out": None})
            print(f"raw read {path.name} ({n / 1e6:.1f} MB) {c:7s} #{i}: {dt * 1e3:6.2f} ms", flush=True)

    df = pd.DataFrame(rows)
    df.insert(0, "layout", a.layout)
    df["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    runs = df[df["result_hash"].notna()]
    print(f"\ntest {a.test}, mode {a.mode}, level {runs['level'].iloc[0]}, "
          f"hashes {'all equal' if runs['result_hash'].nunique() == 1 else 'DIFFER'} ({runs['result_hash'].iloc[0]}, "
          f"{int(runs['rows_out'].iloc[0])} rows)")
    med = df.groupby("cache", sort=False)[["end_to_end", "t_open"]].median() * 1e3
    print(med.round(1).rename(columns=lambda c: c + " median (ms)").to_string())
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
