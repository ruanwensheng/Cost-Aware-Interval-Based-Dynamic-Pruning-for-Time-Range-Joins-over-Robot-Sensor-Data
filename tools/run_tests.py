"""Run the tests of config/tests.yaml on one layout, with one or both approaches.

    python tools/run_tests.py --layout ssd --estimate-only [--l2-includes-condition true]
    python tools/run_tests.py --layout ssd --dry-run --mode all
    python tools/run_tests.py --layout ssd --approach pq --mode all --repeat 5

--approach pq bag      (default both)     --tests 1 2 4-10 ...   (default all)
--mode auto|L2|L3|L4|all [...]             one or more; all = L2, L3, L4 and auto
--repeat N             default 5 for pq, 1 for bag (a bag run can take about an hour)
--warm                 one unmeasured run before each measured run, instead of drop_cache
--estimate-only        cost function only, no data read except metadata. Phase-1 rows come from the
                       local Parquet files (data/parquet). Writes results/tests/estimates_<layout>.csv
                       (estimates_<layout>_l2cond.csv with --l2-includes-condition true).
--dry-run              print the runs and their estimated total time (hours); runs nothing.
--l2-includes-condition true|false   override cost_function.l2_includes_condition of tests.yaml.
--point-a stats|stats_clip   point A + GNSS statistics for every test with cond = gnss
                       (suffix _stats / _stats_clip). stats: T_ub = ell * hit buckets; stats_clip:
                       T_ub = total length of the hit buckets clipped to [w0, w1] (<= D).
--resume               skip runs already in --log (same layout, approach, test, mode, repeat_idx,
                       point A variant, l2_includes_condition, warm or not).

Estimates: est_cost = cost function of the level that runs; est_with_base adds what the function leaves
out: c_q0 + c_req of the main query (pq) or the measured index read t_open (bag). The dry-run uses
est_with_base. Each measured run also logs the power state (on_ac, battery %, Windows power plan and
power mode overlay: best performance / balanced / best power efficiency).

A measured run first calls drop_cache(layout) (cache = "cold", "evict" or "unknown"), or runs once
unmeasured with --warm (cache = "warm"); then appends one JSON line to --log with layout, cache,
repeat_idx, timestamp, host, python, duckdb and the engine's log row.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roslab import bag_engine, pq_engine  # noqa: E402
from roslab.config import RESULTS_DIR  # noqa: E402
from roslab.queries import load_tests  # noqa: E402
from roslab.storage import (drop_cache, evict_path, layout_cfg, layout_constants, open_store,  # noqa: E402
                            parquet_uri)

OUT = RESULTS_DIR / "tests"
LEVELS = ("L2", "L3", "L4")
EST_OF = {"L2": "C_L2", "L3": "total_L3", "L4": "total_L4"}


PS_KEY = r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes"
POWER_MODES = {"ded574b5-45a0-4f42-8737-46345c09c238": "best performance",
               "961cc777-2547-4f9d-8174-7d86181b8a7a": "best power efficiency",
               "": "balanced", "00000000-0000-0000-0000-000000000000": "balanced"}


def power_mode(on_ac: bool | None) -> str | None:
    """Windows 11 power mode: the overlay of the AC (or DC) side, read from the registry."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            PS_KEY) as k:
            name = "ActiveOverlayAcPowerScheme" if on_ac is not False else "ActiveOverlayDcPowerScheme"
            try:
                g = str(winreg.QueryValueEx(k, name)[0]).lower()
            except FileNotFoundError:
                g = ""
        return POWER_MODES.get(g, g)
    except (ImportError, OSError):
        return None


def power_state() -> dict:
    """AC power, battery level, Windows power plan and power mode (None where unknown)."""
    b = psutil.sensors_battery()
    on_ac = None if b is None else bool(b.power_plugged)
    out = {"on_ac": on_ac, "battery_pct": None if b is None else b.percent, "power_plan": None,
           "power_mode": power_mode(on_ac)}
    try:
        txt = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"\((.+)\)\s*$", txt.strip())
        out["power_plan"] = m.group(1) if m else txt.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def local_cond_rows() -> dict[str, pd.DataFrame]:
    """Whole cond tables from the local Parquet files (phase-1 rows for --estimate-only)."""
    return {r: pq.read_table(parquet_uri("ssd", f"parquet/{r}.parquet")).to_pandas() for r in ("gnss", "imu")}


def estimate(specs, approaches, layout: str) -> pd.DataFrame:
    rows_c = local_cond_rows()
    consts = {a: layout_constants(layout, need_duckdb=(a == "pq")) for a in approaches}
    out = []
    store = open_store(layout) if "bag" in approaches else None
    try:
        for sp in specs:
            for a in approaches:
                if a == "bag":
                    r = bag_engine.run(sp, store, consts[a], "auto", sp.bag, estimate_only=True,
                                       cond_rows=rows_c[sp.cond])
                else:
                    r = pq_engine.run(sp, layout, consts[a], "auto", estimate_only=True, cond_rows=rows_c[sp.cond])
                out.append({"layout": layout, "l2_includes_condition": sp.l2_includes_condition,
                            "consts_source": consts[a].source, **r})
    finally:
        if store:
            store.close()
    return pd.DataFrame(out)


def est_seconds(e: pd.Series, mode: str) -> float:
    """Estimated time of one run of `mode` from an (auto) estimate row, with the base cost."""
    level = e["level_auto"] if mode == "auto" else mode
    v = e[EST_OF[level]]
    base = float(e["est_with_base"] - e["est_cost"])                 # c_q0 + c_req (pq) or t_open (bag)
    return (float(v) if pd.notna(v) else float(e["C_L2"])) + base


def plan(specs, approaches, modes, repeat: dict) -> list[tuple]:
    return [(sp, a, m, i) for sp in specs for a in approaches for m in modes for i in range(repeat[a])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", default="ssd")
    ap.add_argument("--approach", nargs="+", choices=("pq", "bag"), default=["pq", "bag"])
    ap.add_argument("--tests", nargs="+", default=None)
    ap.add_argument("--mode", nargs="+", choices=("auto", "L2", "L3", "L4", "all"), default=["auto"])
    ap.add_argument("--repeat", type=int, default=None)
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--l2-includes-condition", choices=("true", "false"), default=None)
    ap.add_argument("--point-a", choices=("stats", "stats_clip"), default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--log", type=Path, default=OUT / "runs.jsonl")
    a = ap.parse_args()

    specs = load_tests(a.tests)
    if a.l2_includes_condition is not None:
        specs = [replace(s, l2_includes_condition=a.l2_includes_condition == "true") for s in specs]
    if a.point_a:
        specs = [replace(s, point_a=a.point_a) if s.cond == "gnss" else s for s in specs]
    modes = [*LEVELS, "auto"] if "all" in a.mode else list(dict.fromkeys(a.mode))
    repeat = {ap_: a.repeat or (5 if ap_ == "pq" else 1) for ap_ in a.approach}
    OUT.mkdir(parents=True, exist_ok=True)

    if a.estimate_only or a.dry_run:
        est = estimate(specs, a.approach, a.layout)
    if a.estimate_only:
        l2c = specs[0].l2_includes_condition
        path = OUT / f"estimates_{a.layout}{'_l2cond' if l2c else ''}{'_' + a.point_a if a.point_a else ''}.csv"
        est.to_csv(path, index=False)
        show = est[["test", "approach", "point_a", "C_L2", "C_p1", "C_L3", "C_L4", "level_auto", "est_cost", "est_with_base",
                    "k_raw", "k", "g_star"]]
        with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
            print(show.to_string(index=False))
        print(f"wrote {path}")
        return

    if a.dry_run:
        e = est.set_index(["test", "approach"])
        cfg = layout_cfg(a.layout)
        ev = evict_path(a.layout)
        import os
        if os.environ.get(cfg.get("cache_cmd_env", "")):
            drop_s, how = 0.0, "cache-drop command (time not counted)"
        elif ev is not None and ev.exists() and not a.warm:
            drop_s, how = ev.stat().st_size / 1.2e9, f"read {ev.name} (~{ev.stat().st_size / 1.2e9:.0f} s at 1.2 GB/s)"
        else:
            drop_s, how = 0.0, "warm-up runs" if a.warm else "none (cache = unknown)"
        runs = plan(specs, a.approach, modes, repeat)
        tot = {}
        for sp, ap_, m, _ in runs:
            s = est_seconds(e.loc[(sp.id, ap_)], m) * (2 if a.warm else 1) + drop_s
            tot[ap_, sp.id, m] = tot.get((ap_, sp.id, m), 0.0) + s
        print(f"{len(runs)} runs; before each: {how}")
        print(f"{'approach':8s} {'test':5s} {'mode':4s} {'runs':>4s} {'hours':>8s}")
        for (ap_, tid, m), s in tot.items():
            print(f"{ap_:8s} {tid:5s} {m:4s} {repeat[ap_]:4d} {s / 3600:8.3f}")
        for ap_ in a.approach:
            print(f"total {ap_}: {sum(v for k, v in tot.items() if k[0] == ap_) / 3600:.2f} h")
        print(f"total: {sum(tot.values()) / 3600:.2f} h")
        return

    consts = {ap_: layout_constants(a.layout, need_duckdb=(ap_ == "pq")) for ap_ in a.approach}
    store = open_store(a.layout) if "bag" in a.approach else None
    a.log.parent.mkdir(parents=True, exist_ok=True)
    meta = {"layout": a.layout, "host": platform.node(), "python": platform.python_version(),
            "duckdb": duckdb.__version__}

    def one(sp, ap_, m):
        if ap_ == "bag":
            return bag_engine.run(sp, store, consts[ap_], m, sp.bag)
        return pq_engine.run(sp, a.layout, consts[ap_], m)

    def key(ap_, tid, m, i, variant, l2c, warm):
        return (a.layout, ap_, str(tid), m, int(i), variant, bool(l2c), bool(warm))

    done = set()
    if a.resume and a.log.exists():
        for line in a.log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add(key(r["approach"], r["test"], r["mode"], r["repeat_idx"], r.get("point_a_variant"),
                             r.get("l2_includes_condition"), r.get("cache") == "warm"))
    todo = [x for x in plan(specs, a.approach, modes, repeat)
            if key(x[1], x[0].id, x[2], x[3], x[0].point_a, x[0].l2_includes_condition, a.warm) not in done]
    print(f"{len(todo)} runs to do" + (f" ({len(done)} already in {a.log})" if a.resume else ""), flush=True)

    try:
        for sp, ap_, m, i in todo:
            if a.warm:
                one(sp, ap_, m)
                cache = "warm"
            else:
                cache = drop_cache(a.layout)
            r = one(sp, ap_, m)
            row = {**meta, **power_state(), "point_a_variant": sp.point_a, "cache": cache, "repeat_idx": i, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "l2_includes_condition": sp.l2_includes_condition, **r}
            with open(a.log, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)) + "\n")
            print(f"{ap_:3s} {sp.id:5s} {m:4s} #{i} {cache:7s} level {r['level']} {r['end_to_end']:.3f} s "
                  f"rows {r['rows_out']} hash {r['result_hash']}" + (" OUTLIER" if r["outlier"] else ""), flush=True)
    finally:
        if store:
            store.close()


if __name__ == "__main__":
    main()
