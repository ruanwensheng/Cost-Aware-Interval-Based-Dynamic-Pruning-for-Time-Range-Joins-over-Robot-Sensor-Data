"""Build time intervals from GNSS points (phase 1 of L3 / L4).

Times are seconds since the bag start. An interval set is a float array of shape (n, 2),
sorted, rows = [start, end]. Shared by M2 (feasibility), M4 (bag_query) and M7 (duckdb_query).

Query types (CLAUDE.md section 5):
    type 1 (Q3, range semi-join): each in-area point g gives [g.t - tol, g.t + tol].
    type 2 (Q4, ASOF join):       each in-area point g gives [g.t, t of the NEXT GNSS point),
                                  the next point being any point (inside the area or not).
Levels:
    exact  union of the per-point intervals (what the query really needs)
    L3     one interval [min start, max end]
    L4     one interval per visit; a visit ends when two consecutive in-area points are more
           than `gap` seconds apart; then intervals closer than `merge` seconds are merged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EMPTY = np.empty((0, 2), dtype=float)


@dataclass(frozen=True)
class Box:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @classmethod
    def from_dict(cls, d: dict) -> "Box":
        return cls(d["lat_min"], d["lat_max"], d["lon_min"], d["lon_max"])


def in_area(lat, lon, box: Box) -> np.ndarray:
    """Same test as the SQL in_area_X: inclusive BETWEEN on lat and lon."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    return (lat >= box.lat_min) & (lat <= box.lat_max) & (lon >= box.lon_min) & (lon <= box.lon_max)


def merge_intervals(iv, merge: float = 0.0) -> np.ndarray:
    """Union of intervals; also joins two intervals whose gap is < merge (merge=0: only overlaps/touches)."""
    iv = np.asarray(iv, float).reshape(-1, 2)
    if len(iv) == 0:
        return EMPTY.copy()
    iv = iv[np.argsort(iv[:, 0], kind="stable")]
    out = [iv[0].copy()]
    for s, e in iv[1:]:
        cur = out[-1]
        if s - cur[1] < merge or s <= cur[1]:
            cur[1] = max(cur[1], e)
        else:
            out.append(np.array([s, e]))
    return np.array(out)


def clip(iv, lo: float, hi: float) -> np.ndarray:
    """Intersect every interval with [lo, hi]; drop empty ones."""
    iv = np.asarray(iv, float).reshape(-1, 2)
    c = np.column_stack([np.maximum(iv[:, 0], lo), np.minimum(iv[:, 1], hi)])
    return c[c[:, 0] <= c[:, 1]]


def point_intervals(t, inside, qtype: int, tol: float = 0.1, t_after_last: float = np.inf) -> np.ndarray:
    """Per-point intervals of the in-area points (not merged).

    t must be sorted and contain ALL GNSS points (inside or not): type 2 needs the next point.
    t_after_last: end of the type-2 interval of the very last point (default +inf; clip later).
    """
    t = np.asarray(t, float)
    inside = np.asarray(inside, bool)
    idx = np.flatnonzero(inside)
    if qtype == 1:
        return np.column_stack([t[idx] - tol, t[idx] + tol])
    if qtype == 2:
        nxt = np.append(t[1:], t_after_last)
        return np.column_stack([t[idx], nxt[idx]])
    raise ValueError(f"qtype must be 1 or 2, got {qtype}")


def visits(t, inside, gap: float) -> list[tuple[int, int]]:
    """Group in-area points into visits. Returns (first, last) indices into t, inclusive.

    A new visit starts when two consecutive IN-AREA points are more than `gap` s apart.
    """
    t = np.asarray(t, float)
    idx = np.flatnonzero(np.asarray(inside, bool))
    if len(idx) == 0:
        return []
    cut = np.flatnonzero(np.diff(t[idx]) > gap)
    starts = np.r_[0, cut + 1]
    ends = np.r_[cut, len(idx) - 1]
    return [(int(idx[s]), int(idx[e])) for s, e in zip(starts, ends)]


def build(t, inside, qtype: int, level: str, tol: float = 0.1, gap: float = 0.3, merge: float = 0.0,
          t_after_last: float = np.inf) -> np.ndarray:
    """Interval set for one query type and level ('exact', 'L3', 'L4')."""
    t = np.asarray(t, float)
    pts = point_intervals(t, inside, qtype, tol, t_after_last)
    if len(pts) == 0:
        return EMPTY.copy()
    if level == "exact":
        return merge_intervals(pts)
    if level == "L3":
        return np.array([[pts[:, 0].min(), pts[:, 1].max()]])
    if level == "L4":
        nxt = np.append(t[1:], t_after_last)
        iv = []
        for first, last in visits(t, inside, gap):
            if qtype == 1:
                iv.append([t[first] - tol, t[last] + tol])
            else:
                iv.append([t[first], nxt[last]])
        return merge_intervals(iv, merge)
    raise ValueError(f"level must be exact, L3 or L4, got {level!r}")


def total_length(iv) -> float:
    iv = np.asarray(iv, float).reshape(-1, 2)
    return float((iv[:, 1] - iv[:, 0]).sum())


def gaps(iv) -> np.ndarray:
    """Lengths of the holes between consecutive intervals of a sorted, merged set."""
    iv = np.asarray(iv, float).reshape(-1, 2)
    return iv[1:, 0] - iv[:-1, 1]


def blocks_hit(iv, ell: float, t0: float = 0.0) -> np.ndarray:
    """Indices j of the fixed-length blocks [t0 + j*ell, t0 + (j+1)*ell) that an interval set touches."""
    iv = np.asarray(iv, float).reshape(-1, 2)
    if len(iv) == 0:
        return np.empty(0, dtype=np.int64)
    js = [np.arange(int(np.floor((s - t0) / ell)), int(np.floor((e - t0) / ell)) + 1) for s, e in iv]
    return np.unique(np.concatenate(js))
