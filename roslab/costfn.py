"""Cost function that picks the pruning level L2, L3 or L4 (PROMPT_cost_function.md, section 5).

Pure functions: no I/O. Costs are in seconds, sizes in bytes, rates per second.
c = cond table (read in phase 1), t = target table (returned).

Point A runs before phase 1, point B after it:
    point A: run phase 1 only if it is cheaper than L2 (optionally with a GNSS-statistics bound);
    point B: choose L3 if C_L3 <= C_L4, else L4.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from roslab import intervals as I


class MissingConstant(RuntimeError):
    """A benchmark constant has not been measured yet; the message holds the command to run."""


@dataclass(frozen=True)
class LayoutConsts:
    name: str
    local: bool              # ssd: the b / bw term of a bag block is dropped
    c_req: float             # s, fixed cost of one read request
    bw: float                # B/s, read bandwidth
    v_dec: float             # B/s of compressed input, bz2 decompression
    c_q0: float | None       # s, DuckDB query start-up cost without the read request
    c_int: float | None      # s, extra cost of one more UNION ALL branch
    source: str = ""

    def require_duckdb(self) -> None:
        if self.c_q0 is None or self.c_int is None:
            raise MissingConstant(f"c_q0 / c_int not measured for layout {self.name!r}: run "
                                  f"python tools/bench_duckdb.py --layout {self.name}")


@dataclass
class BagInputs:
    D: float        # window length (s)
    r_c: float      # cond chunks per second, on [w0 - tol, w1 + tol]
    r_t: float      # target chunks per second, on [w0, w1]
    r_ct: float     # chunks holding both, per second
    b_c: float      # mean size of a cond chunk (B)
    b_t: float      # mean size of a target chunk (B)


@dataclass
class PqInputs:
    D: float        # window length (s)
    ell: float      # row group length (s)
    rho_c: float    # bytes per second of the cond columns read
    rho_t: float    # bytes per second of the target columns read
    n_c: int        # cond row groups intersecting the window
    n_t: int        # target row groups intersecting the window


@dataclass
class Decision:
    point_a: str                    # "phase1" or "L2"
    point_b: str | None             # "L3", "L4" or None (no phase 1)
    level: str | None               # level that runs: L2, L3 or L4 (None until point B decides)
    est: dict = field(default_factory=dict)
    filter: np.ndarray | None = None   # phase-2 interval set (L3: 1 interval, L4: k intervals)


EST_OF = {"L2": "C_L2", "L3": "total_L3", "L4": "total_L4"}


def log_fields(d: Decision, level: str | None = None, level_auto: str | None = None) -> dict:
    """Cost-function numbers to log (None when not computed).
    est_cost = estimate of `level` (the level that ran), est_auto = estimate of `level_auto`
    (C_L2 for L2, C_p1 + C_L3 for L3, C_p1 + C_L4 for L4)."""
    keys = ("C_L2", "C_p1", "point_a_lhs", "C_L3", "C_L4", "total_L3", "total_L4", "k_raw", "k", "span",
            "sum_w", "g_star", "N_L3", "N_L4", "T_ub")
    out = {k: d.est.get(k) for k in keys}
    pick = lambda lv: d.est.get(EST_OF[lv]) if lv in EST_OF else d.est.get("est")   # noqa: E731
    return {**out, "est_cost": pick(level), "est_auto": pick(level_auto)}


# ------------------------------------------------------------------ approach 1: bag
def block(b: float, k: LayoutConsts) -> float:
    """Cost of reading and decompressing one chunk of b bytes."""
    return k.c_req + (0.0 if k.local else b / k.bw) + b / k.v_dec


def bag_costs(x: BagInputs, k: LayoutConsts, l2_includes_condition: bool = False) -> dict:
    c_l2 = x.r_t * x.D * block(x.b_t, k)
    if l2_includes_condition:
        c_l2 += (x.r_c - x.r_ct) * x.D * block(x.b_c, k)
    return {"C_L2": c_l2, "C_p1": x.r_c * x.D * block(x.b_c, k)}


def bag_rest(x: BagInputs, k: LayoutConsts, T: float) -> float:
    """C_rest(T): target chunks not already read in phase 1, over T seconds."""
    return (x.r_t - x.r_ct) * T * block(x.b_t, k)


def bag_point_a(x: BagInputs, k: LayoutConsts, T_ub: float | None = None,
                l2_includes_condition: bool = False) -> Decision:
    """T_ub given: point A + GNSS statistics (phase 1 only if C_p1 + C_rest(T_ub) < C_L2)."""
    c = bag_costs(x, k, l2_includes_condition)
    lhs = c["C_p1"] + (bag_rest(x, k, T_ub) if T_ub is not None else 0.0)
    go = lhs < c["C_L2"]
    est = {**c, "T_ub": T_ub, "point_a_lhs": lhs, "est": None if go else c["C_L2"]}
    return Decision("phase1" if go else "L2", None, None if go else "L2", est)


def bag_point_b(x: BagInputs, k: LayoutConsts, l3: np.ndarray, l4: np.ndarray, a: Decision) -> Decision:
    """After phase 1. No merging on the bag: one more interval does not add chunk reads."""
    span, sum_w = I.total_length(l3), I.total_length(l4)
    c_l3, c_l4 = bag_rest(x, k, span), bag_rest(x, k, sum_w)
    level = "L3" if c_l3 <= c_l4 else "L4"
    est = {**a.est, "C_L3": c_l3, "C_L4": c_l4, "total_L3": a.est["C_p1"] + c_l3,
           "total_L4": a.est["C_p1"] + c_l4, "k": len(l4), "k_raw": len(l4), "span": span, "sum_w": sum_w,
           "g_star": None}
    est["est"] = est["total_L3"] if level == "L3" else est["total_L4"]
    return Decision(a.point_a, level, level, est, l3 if level == "L3" else l4)


# ------------------------------------------------------------------ approach 2: Parquet
def rg(rho: float, ell: float, k: LayoutConsts) -> float:
    """Cost of reading one row group of ell seconds."""
    return k.c_req + ell * rho / k.bw


def n_l2(x: PqInputs, blocks: str = "formula") -> float:
    return x.D / x.ell if blocks == "formula" else x.n_t                       # (1)


def n_l3(span: float, ell: float) -> float:
    return (span + ell) / ell                                                  # (2)


def n_l4(sum_w: float, k_iv: int, ell: float) -> float:
    return (sum_w + k_iv * ell) / ell                                          # (3)


def pq_costs(x: PqInputs, k: LayoutConsts, blocks: str = "formula", l2_includes_condition: bool = False) -> dict:
    k.require_duckdb()
    c_l2 = n_l2(x, blocks) * rg(x.rho_t, x.ell, k)                             # (4)
    if l2_includes_condition:
        c_l2 += x.n_c * rg(x.rho_c, x.ell, k)
    c_p1 = k.c_q0 + k.c_req + x.n_c * rg(x.rho_c, x.ell, k)                    # (5)
    return {"C_L2": c_l2, "C_p1": c_p1}


def merge_gap(x: PqInputs, k: LayoutConsts) -> float:
    """(7) g* = c_int * ell / rg(rho_t): two visits closer than g* are merged."""
    k.require_duckdb()
    return k.c_int * x.ell / rg(x.rho_t, x.ell, k)


def pq_point_a(x: PqInputs, k: LayoutConsts, n_hit: float | None = None, blocks: str = "formula",
               l2_includes_condition: bool = False) -> Decision:
    """n_hit given: point A + GNSS statistics, C_rest(T_ub) = n_hit * rg(rho_t), T_ub = ell * n_hit.
    n_hit may be fractional (stats_clip: clipped seconds / ell)."""
    c = pq_costs(x, k, blocks, l2_includes_condition)
    rest = n_hit * rg(x.rho_t, x.ell, k) if n_hit is not None else 0.0
    lhs = c["C_p1"] + rest
    go = lhs < c["C_L2"]
    est = {**c, "T_ub": None if n_hit is None else n_hit * x.ell, "point_a_lhs": lhs,
           "est": None if go else c["C_L2"]}
    return Decision("phase1" if go else "L2", None, None if go else "L2", est)


def pq_point_b(x: PqInputs, k: LayoutConsts, l3: np.ndarray, l4: np.ndarray, a: Decision,
               n_exact: tuple[int, int] | None = None) -> Decision:
    """l4 = raw visits (clipped); they are merged with g* first. n_exact = (rg for L3, rg for L4)
    counted from the footer, used instead of (2)-(3) when blocks = exact."""
    g = merge_gap(x, k)
    l4m = I.merge_intervals(l4, g)
    span, sum_w, k_iv = I.total_length(l3), I.total_length(l4m), len(l4m)
    N3, N4 = n_exact if n_exact else (n_l3(span, x.ell), n_l4(sum_w, k_iv, x.ell))
    c_l3 = N3 * rg(x.rho_t, x.ell, k)                                          # (6) L3
    c_l4 = N4 * rg(x.rho_t, x.ell, k) + (k_iv - 1) * k.c_int                   # (6) L4
    level = "L3" if c_l3 <= c_l4 else "L4"
    est = {**a.est, "C_L3": c_l3, "C_L4": c_l4, "total_L3": a.est["C_p1"] + c_l3,
           "total_L4": a.est["C_p1"] + c_l4, "k": k_iv, "k_raw": len(l4), "span": span, "sum_w": sum_w,
           "g_star": g, "N_L3": N3, "N_L4": N4}
    est["est"] = est["total_L3"] if level == "L3" else est["total_L4"]
    return Decision(a.point_a, level, level, est, l3 if level == "L3" else l4m)
