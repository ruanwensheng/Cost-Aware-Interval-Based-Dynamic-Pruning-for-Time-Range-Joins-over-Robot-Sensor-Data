"""Approach 2: run one test on the Parquet files with DuckDB (local disk or S3/MinIO).

    out = run(spec, "ssd", consts, mode="auto")

Files: parquet/imu.parquet, parquet/gnss.parquet (tools/convert.py), row groups of ell seconds.
    L2       WITH g AS (<cond query>) SELECT x.* FROM target x
             WHERE x.t BETWEEN w0 AND w1 AND EXISTS (SELECT 1 FROM g WHERE x.t BETWEEN g.t - tol AND g.t + tol)
    phase 1  SELECT t, <cond columns> FROM cond WHERE t BETWEEN w0 - tol AND w1 + tol AND <condition> ORDER BY t
    phase 2  g = phase-1 rows (con.register); L3 adds one BETWEEN a AND b; L4 replaces the target by a
             UNION ALL of one branch per merged interval; then the same EXISTS as L2.
Time is measured from a ready connection to the result (footer, point A, phase 1, point B, phase 2).
Floats go into SQL as '<repr>'::DOUBLE so DuckDB gets exactly the Python value.
"""
from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from roslab import costfn as C
from roslab import intervals as I
from roslab.queries import COLUMNS, Spec, filter_intervals, result_hash, sql_float
from roslab.storage import duckdb_connect, is_outlier, parquet_uri


def row_groups(con, uri: str, cols: list[str]) -> pd.DataFrame:
    """One row per row group: t_min, t_max, bytes of `cols`, and lat/lon min/max when present."""
    m = con.execute("SELECT row_group_id AS rg, path_in_schema AS col, stats_min, stats_max, "
                    f"total_compressed_size AS size FROM parquet_metadata('{uri}')").df()
    rg_ids = np.sort(m["rg"].unique())
    out = (m[m["col"].isin(cols)].groupby("rg")["size"].sum()
           .reindex(rg_ids, fill_value=0).rename("bytes").to_frame())
    out.index.name = "rg"
    for col in ("t", "lat", "lon"):
        s = m[m["col"] == col].set_index("rg")
        if len(s):
            out[f"{col}_min"] = s["stats_min"].astype(float)
            out[f"{col}_max"] = s["stats_max"].astype(float)
    return out.reset_index()


def overlap(rgs: pd.DataFrame, iv) -> np.ndarray:
    """Row groups whose [t_min, t_max] meets one of the intervals."""
    m = np.zeros(len(rgs), bool)
    for s, e in np.asarray(iv, float).reshape(-1, 2):
        m |= (rgs["t_max"].to_numpy() >= s) & (rgs["t_min"].to_numpy() <= e)
    return m


def with_base(est: float | None, L: C.LayoutConsts) -> float | None:
    """Estimate plus the start-up of the main query (c_q0 + c_req), which (4)-(6) leave out."""
    return None if est is None else est + L.c_q0 + L.c_req


def run(spec: Spec, layout: str, L: C.LayoutConsts, mode: str = "auto", estimate_only: bool = False,
        cond_rows: pd.DataFrame | None = None, threads: int | None = None) -> dict:
    """Run one test. mode: auto | L2 | L3 | L4. Returns one log row (dict).
    estimate_only: cost function only; cond_rows = the whole cond table (e.g. read from the local
    Parquet file), filtered here to [w0 - tol, w1 + tol] and the condition; None = run phase 1."""
    con = duckdb_connect(layout, threads)
    T: dict[str, float] = {}
    t_start = time.perf_counter()

    # ---- footer -> point A inputs
    c_uri = parquet_uri(layout, f"parquet/{spec.cond}.parquet")
    t_uri = parquet_uri(layout, f"parquet/{spec.target}.parquet")
    c_cols = ["t", *spec.cond_cols]
    rc, rt = row_groups(con, c_uri, c_cols), row_groups(con, t_uri, COLUMNS[spec.target])
    w0, w1 = spec.window_range(max(rc["t_max"].max(), rt["t_max"].max()))
    lo, hi, D, ell = w0 - spec.tol, w1 + spec.tol, w1 - w0, spec.ell
    in_t, in_c = overlap(rt, [[w0, w1]]), overlap(rc, [[lo, hi]])
    x = C.PqInputs(D=D, ell=ell, rho_c=rc.loc[in_c, "bytes"].mean() / ell if in_c.any() else 0.0,
                   rho_t=rt.loc[in_t, "bytes"].mean() / ell if in_t.any() else 0.0,
                   n_c=int(in_c.sum()), n_t=int(in_t.sum()))
    T["t_open"] = time.perf_counter() - t_start                     # open + footers of both files

    t0 = time.perf_counter()
    n_hit = None
    if spec.point_a in ("stats", "stats_clip"):
        if spec.cond != "gnss":
            raise ValueError(f"test {spec.id}: point A + GNSS statistics needs cond = gnss")
        b = spec.box
        win = overlap(rc, [[w0, w1]])
        hit = (win & (rc["lat_max"] >= b.lat_min) & (rc["lat_min"] <= b.lat_max)
               & (rc["lon_max"] >= b.lon_min) & (rc["lon_min"] <= b.lon_max)).to_numpy()
        n_hit = int(hit.sum())
        if spec.point_a == "stats_clip":         # bucket j = [j ell, (j+1) ell) clipped to the window
            j0 = np.floor(rc["t_min"].to_numpy()[hit] / ell) * ell
            clip = np.clip(np.minimum(j0 + ell, w1) - np.maximum(j0, w0), 0, None).sum()
            n_hit = min(float(clip), D) / ell    # fractional buckets: C_rest = T_ub / ell * rg(rho_t)
    a = C.pq_point_a(x, L, n_hit, spec.blocks, spec.l2_includes_condition)
    T["t_point_a"] = time.perf_counter() - t0
    run_phase1 = a.point_a == "phase1" if mode == "auto" else mode in ("L3", "L4")

    cond_sql = (f"SELECT t, {', '.join(spec.cond_cols)} FROM read_parquet('{c_uri}') "
                f"WHERE t BETWEEN {sql_float(lo)} AND {sql_float(hi)} AND {spec.cond_sql()} ORDER BY t")
    exists = (f"x.t BETWEEN {sql_float(w0)} AND {sql_float(w1)} AND EXISTS (SELECT 1 FROM g "
              f"WHERE x.t BETWEEN g.t - {sql_float(spec.tol)} AND g.t + {sql_float(spec.tol)})")

    def point_b(cdf: pd.DataFrame) -> tuple[C.Decision, dict[str, np.ndarray]]:
        t = cdf["t"].to_numpy()
        inside = np.ones(len(t), bool)                    # phase-1 rows already satisfy the condition
        l3, l4 = (filter_intervals(t, inside, lvl, spec, w0, w1) for lvl in ("L3", "L4"))
        l4m = I.merge_intervals(l4, C.merge_gap(x, L))
        n_exact = (int(overlap(rt, l3).sum()), int(overlap(rt, l4m).sum())) if spec.blocks == "exact" else None
        return C.pq_point_b(x, L, l3, l4, a, n_exact), {"L3": l3, "L4": l4m}

    out = {"approach": "pq", "test": spec.id, "label": spec.label, "mode": mode, "w0": w0, "w1": w1,
           "point_a": a.point_a, "rg_total": len(rt), "rg_target_window": x.n_t, "rg_cond_window": x.n_c,
           "threads": threads, "outlier": is_outlier(T["t_open"]), **asdict(x)}

    if estimate_only:
        if cond_rows is not None:                          # whole cond table: keep phase-1 rows only
            cdf = cond_rows[(cond_rows["t"] >= lo) & (cond_rows["t"] <= hi)]
            cdf = cdf[spec.cond_mask(cdf)]
        else:
            cdf = con.execute(cond_sql).df()
        b = point_b(cdf)[0]
        auto = "L2" if a.point_a == "L2" else b.level
        con.close()
        lv = auto if mode == "auto" else mode
        f = C.log_fields(b, lv, auto)
        return {**out, **f, "est_with_base": with_base(f["est_cost"], L), "level": lv, "level_auto": auto,
                "point_b": b.point_b, **T}

    b = None
    if not run_phase1:                                     # ---- L2: one query
        t0 = time.perf_counter()
        sql = (f"WITH g AS ({cond_sql}) SELECT x.* FROM read_parquet('{t_uri}') x WHERE {exists}")
        res = con.execute(sql).to_arrow_table()
        T["t_l2"] = time.perf_counter() - t0
        level, n_iv, rg_exp = "L2", 0, x.n_t
    else:                                                  # ---- phase 1, point B, phase 2
        t0 = time.perf_counter()
        cdf = con.execute(cond_sql).df()
        T["t_phase1"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        b, iv = point_b(cdf)
        level = b.level if mode == "auto" else mode
        filt = iv[level]
        T["t_point_b"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        con.register("g", cdf[["t"]])
        cols = ", ".join(COLUMNS[spec.target])
        base = f"read_parquet('{t_uri}')"
        if len(filt) == 0:
            sql = f"SELECT {cols} FROM {base} WHERE false"
        elif level == "L3":
            a3, b3 = filt[0]
            sql = (f"SELECT x.* FROM {base} x WHERE x.t BETWEEN {sql_float(a3)} AND {sql_float(b3)} "
                   f"AND {exists}")
        else:
            branches = " UNION ALL ".join(f"SELECT {cols} FROM {base} WHERE t BETWEEN {sql_float(s)} AND {sql_float(e)}"
                                          for s, e in filt)
            sql = f"SELECT x.* FROM ({branches}) x WHERE {exists}"
        res = con.execute(sql).to_arrow_table()
        T["t_phase2"] = time.perf_counter() - t0
        n_iv, rg_exp = len(filt), int(overlap(rt, filt).sum())
    T["end_to_end"] = time.perf_counter() - t_start

    if b is None:                                          # L2 ran: estimates of point B, not timed
        b = point_b(con.execute(cond_sql).df())[0]
    level_auto = "L2" if a.point_a == "L2" else b.level
    t_ns = res.column("t_ns").to_numpy()
    con.close()
    f = C.log_fields(b, level, level_auto)
    return {**out, **f, "est_with_base": with_base(f["est_cost"], L), "level": level, "level_auto": level_auto, "point_b": b.point_b, **T,
            "n_intervals": n_iv, "rg_expected": rg_exp, "rows_out": len(t_ns), "result_hash": result_hash(t_ns)}
