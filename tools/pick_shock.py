"""Pick the shock condition of test 5 from the IMU Parquet file and write config/shock.yaml.

    python tools/pick_shock.py [--quantile 0.9999] [--parquet data/parquet/imu.parquet]

metric = | sqrt(ax*ax + ay*ay + az*az) - g0 |   (no assumption on which axis is vertical)
g0 = median of the acceleration magnitude over the whole bag; threshold = quantile of the metric.
A shock sample has metric > threshold (same expression as Spec.cond_mask / cond_sql).
Events = shock samples grouped while consecutive ones are at most `gap` s apart (config/tests.yaml).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roslab.config import CONFIG_DIR, ROOT, load_yaml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quantile", type=float, default=0.9999)
    ap.add_argument("--parquet", type=Path, default=ROOT / "data" / "parquet" / "imu.parquet")
    ap.add_argument("--out", type=Path, default=CONFIG_DIR / "shock.yaml")
    a = ap.parse_args()

    tab = pq.read_table(a.parquet, columns=["t", "ax", "ay", "az"])
    t, ax, ay, az = (tab[c].to_numpy() for c in ("t", "ax", "ay", "az"))
    mag = np.sqrt(ax * ax + ay * ay + az * az)
    g0 = float(np.median(mag))
    metric = np.abs(mag - g0)
    thr = float(np.quantile(metric, a.quantile))
    shock = metric > thr
    ts = t[shock]
    gap = float(load_yaml("tests.yaml")["gap"])
    n_events = int(len(ts) > 0) + int((np.diff(ts) > gap).sum())

    out = {"g0": g0, "threshold": thr, "quantile": a.quantile, "n_samples": int(len(t)),
           "n_shock_samples": int(shock.sum()), "n_events": n_events, "event_gap_s": gap,
           "source": a.parquet.relative_to(ROOT).as_posix() if a.parquet.is_relative_to(ROOT) else str(a.parquet)}
    a.out.write_text("# Written by tools/pick_shock.py. Condition of test 5: |sqrt(ax^2+ay^2+az^2) - g0| > threshold\n"
                     + yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    print(f"IMU samples      {len(t)}")
    print(f"g0               {g0:.6f} m/s^2 (median magnitude)")
    print(f"threshold        {thr:.6f} m/s^2 (quantile {a.quantile})")
    print(f"shock samples    {int(shock.sum())}")
    print(f"shock events     {n_events} (samples at most {gap} s apart are one event)")
    print(f"event times (s)  {np.round(ts[np.r_[True, np.diff(ts) > gap]], 2).tolist() if len(ts) else []}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
