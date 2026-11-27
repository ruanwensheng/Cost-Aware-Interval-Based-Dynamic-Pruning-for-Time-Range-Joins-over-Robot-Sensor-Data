"""The 6 tests (config/tests.yaml): one range semi-join (type 1) each.

Result of a test = rows of `target` with t in [w0, w1] that are within tol of a row of `cond`
satisfying the condition; cond rows are taken in [w0 - tol, w1 + tol].
cond_mask (numpy) and cond_sql (DuckDB) express the SAME condition, with the same
operation order, so both engines select the same rows.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from roslab import intervals as I
from roslab.config import CONFIG_DIR, load_yaml

COLUMNS = {"imu": ["t_ns", "t", "qx", "qy", "qz", "qw", "wx", "wy", "wz", "ax", "ay", "az"],
           "gnss": ["t_ns", "t", "lat", "lon", "alt", "status"]}


def sql_float(v: float) -> str:
    """A float literal DuckDB parses to exactly the same double as Python."""
    return f"'{float(v)!r}'::DOUBLE"


@dataclass
class Spec:
    id: str
    label: str
    cond: str                     # table read in phase 1: "gnss" or "imu"
    target: str                   # table returned
    window: object                # "whole" or [w0, w1]
    point_a: str                  # "plain", "stats" (T_ub = ell * hit buckets) or "stats_clip"
                                  # (T_ub = total length of hit buckets clipped to [w0, w1], <= D)
    bag: int
    ell: float
    tol: float
    gap: float
    blocks: str = "formula"
    l2_includes_condition: bool = False
    area: str | None = None
    box: I.Box | None = None
    shock: bool = False
    g0: float | None = None
    threshold: float | None = None

    def window_range(self, duration: float) -> tuple[float, float]:
        if self.window == "whole":
            return 0.0, float(duration)
        w0, w1 = self.window
        return float(w0), float(w1)

    @property
    def cond_cols(self) -> list[str]:
        return ["ax", "ay", "az"] if self.shock else ["lat", "lon"]

    def _need_shock(self) -> None:
        if self.shock and (self.g0 is None or self.threshold is None):
            raise RuntimeError(f"test {self.id}: config/shock.yaml is missing: run  python tools/pick_shock.py")

    def cond_mask(self, df: pd.DataFrame) -> np.ndarray:
        """Condition on cond rows (numpy)."""
        if self.shock:
            self._need_shock()
            ax, ay, az = (df[c].to_numpy(float) for c in ("ax", "ay", "az"))
            return np.abs(np.sqrt(ax * ax + ay * ay + az * az) - self.g0) > self.threshold
        return I.in_area(df["lat"], df["lon"], self.box)

    def cond_sql(self) -> str:
        """Condition on cond rows (SQL), same as cond_mask."""
        if self.shock:
            self._need_shock()
            return f"abs(sqrt(ax * ax + ay * ay + az * az) - {sql_float(self.g0)}) > {sql_float(self.threshold)}"
        b = self.box
        return (f"lat BETWEEN {sql_float(b.lat_min)} AND {sql_float(b.lat_max)} "
                f"AND lon BETWEEN {sql_float(b.lon_min)} AND {sql_float(b.lon_max)}")


def load_tests(ids: list[str] | None = None) -> list[Spec]:
    """Specs of config/tests.yaml, in the order of `ids` (all tests if None)."""
    cfg = load_yaml("tests.yaml")
    areas = {**load_yaml("areas.yaml"), **(cfg.get("areas_extra") or {})}
    shock = load_yaml("shock.yaml") if (CONFIG_DIR / "shock.yaml").exists() else {}
    cf = cfg.get("cost_function") or {}
    by_id = {}
    for t in cfg["tests"]:
        area = t.get("area")
        by_id[str(t["id"])] = Spec(
            id=str(t["id"]), label=t["label"], cond=t["cond"], target=t["target"], window=t["window"],
            point_a=t.get("point_a", "plain"), bag=int(cfg["bag"]), ell=float(cfg["ell"]),
            tol=float(cfg["tol"]), gap=float(cfg["gap"]), blocks=cf.get("blocks", "formula"),
            l2_includes_condition=bool(cf.get("l2_includes_condition", False)),
            area=area, box=I.Box.from_dict(areas[area]) if area else None, shock=bool(t.get("shock", False)),
            g0=shock.get("g0"), threshold=shock.get("threshold"))
    ids = ids or list(by_id)
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise KeyError(f"unknown test ids {unknown}; have {list(by_id)}")
    return [by_id[i] for i in ids]


def filter_intervals(t, inside, level: str, spec: Spec, w0: float, w1: float) -> np.ndarray:
    """Type-1 intervals of the cond rows for one level ('exact', 'L3', 'L4'), clipped to [w0, w1]."""
    return I.clip(I.build(t, inside, 1, level, tol=spec.tol, gap=spec.gap), w0, w1)


def result_hash(t_ns) -> str:
    """sha1 of the sorted int64 t_ns values, first 16 hex characters."""
    return hashlib.sha1(np.sort(np.asarray(t_ns, dtype=np.int64)).tobytes()).hexdigest()[:16]


def in_intervals(t, iv) -> np.ndarray:
    """True where t lies in one of the closed intervals of iv."""
    t = np.asarray(t, float)
    iv = I.merge_intervals(iv)
    if len(iv) == 0:
        return np.zeros(len(t), bool)
    idx = np.searchsorted(iv[:, 0], t, side="right") - 1
    ok = idx >= 0
    ok[ok] = t[ok] <= iv[idx[ok], 1]
    return ok
