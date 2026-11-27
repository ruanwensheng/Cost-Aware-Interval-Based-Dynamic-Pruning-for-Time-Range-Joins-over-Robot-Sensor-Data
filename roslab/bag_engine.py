"""Approach 1: run one test directly on the ROS bag, through a Store (local file or S3/MinIO).

    out = run(spec, store, consts, mode="auto", bag_no=5)

Reads:
    index    2 requests: the first 4 KiB (bag header -> index_pos) and [index_pos, end of file]
    a chunk  1 request:  [chunk_pos, next chunk_pos)  (last chunk: up to index_pos)
Levels:
    L2       chunks holding cond (on [w0 - tol, w1 + tol]) OR target (on [w0, w1])
    L3 / L4  phase 1 = cond chunks (target rows found there are kept);
             phase 2 = target chunks meeting the filter intervals, minus chunks already read
Final join: target rows with t in [w0, w1] and t in the "exact" intervals.
"""
from __future__ import annotations

import struct
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from roslab import costfn as C
from roslab import decode
from roslab.bagfile import MAGIC, BagError, BagFile, ChunkHeader, iter_messages, parse_header
from roslab.config import load_yaml, topics
from roslab.queries import COLUMNS, Spec, filter_intervals, in_intervals, result_hash
from roslab.storage import is_outlier

HEAD_BYTES = 4096


class SparseFile:
    """Read-only, seekable file-like object made of a few loaded byte ranges."""

    def __init__(self, segments: dict[int, bytes], size: int) -> None:
        self._segs = sorted(segments.items())
        self.size, self.pos = size, 0

    def seek(self, pos: int, whence: int = 0) -> int:
        self.pos = pos if whence == 0 else (self.pos + pos if whence == 1 else self.size + pos)
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        for start, data in self._segs:
            if start <= self.pos < start + len(data):
                off = self.pos - start
                out = data[off:] if n < 0 else data[off:off + n]
                self.pos += len(out)
                return out
        raise BagError(f"byte {self.pos} was not loaded")


def bag_key(bag_no: int) -> str:
    return "bags/" + load_yaml("bags.yaml")["bags"][bag_no]["file"]


def read_index(store, key: str) -> BagFile:
    """Bag header + index section in 2 requests."""
    head = store.read(key, 0, HEAD_BYTES)
    if not head.startswith(MAGIC):
        raise BagError(f"{key}: not a ROS bag v2.0 file")
    p = len(MAGIC)
    hlen = struct.unpack_from("<I", head, p)[0]
    index_pos = struct.unpack("<Q", parse_header(head[p + 4:p + 4 + hlen])["index_pos"])[0]
    tail = store.read(key, index_pos)
    size = index_pos + len(tail)
    bag = BagFile(key, fileobj=SparseFile({0: head, index_pos: tail}, size), size=size)
    bag.open()
    return bag


class Reader:
    """Reads, decompresses and decodes chunks; each chunk at most once."""

    def __init__(self, store, key: str, bag: BagFile, conns: dict[int, str]) -> None:
        self.store, self.key, self.bag = store, key, bag
        self.conns = conns                                   # conn -> role (gnss / imu)
        pos = [c.chunk_pos for c in bag.chunks] + [bag.index_pos]
        self.ends = np.array(pos[1:], dtype=np.int64)
        self.done = np.zeros(len(bag.chunks), bool)
        self.rows = {role: ([], []) for role in set(conns.values())}
        self.t_read = self.t_dec = self.t_decode = 0.0
        self.bytes_dec = 0
        self.bytes_in = 0                                    # compressed bytes fed to the decompressor
        self._decoders = {k: decode.DECODERS[bag.connections[k].msgtype][0] for k in conns}

    def read(self, mask: np.ndarray) -> int:
        """Read every chunk of mask not read yet. Returns the number of chunks read."""
        todo = np.flatnonzero(mask & ~self.done)
        for i in todo:
            c = self.bag.chunks[i]
            t0 = time.perf_counter()
            buf = self.store.read(self.key, c.chunk_pos, int(self.ends[i]) - c.chunk_pos)
            t1 = time.perf_counter()
            hlen = struct.unpack_from("<I", buf, 0)[0]
            h = parse_header(buf[4:4 + hlen])
            dlen = struct.unpack_from("<I", buf, 4 + hlen)[0]
            ch = ChunkHeader(h["compression"].decode(), struct.unpack("<I", h["size"])[0], c.chunk_pos + 8 + hlen, dlen)
            raw = BagFile.decompress(ch, buf[8 + hlen:8 + hlen + dlen])
            t2 = time.perf_counter()
            for conn, t_ns, data in iter_messages(raw, set(self.conns)):
                ts, vals = self.rows[self.conns[conn]]
                ts.append(t_ns)
                vals.append(self._decoders[conn](data))
            t3 = time.perf_counter()
            self.t_read += t1 - t0
            self.t_dec += t2 - t1
            self.t_decode += t3 - t2
            self.bytes_dec += len(raw)
            self.bytes_in += dlen
        self.done[todo] = True
        return len(todo)

    def frame(self, role: str) -> pd.DataFrame:
        ts, vals = self.rows[role]
        t_ns = np.array(ts, dtype=np.int64)
        df = pd.DataFrame(vals, columns=COLUMNS[role][2:])
        df.insert(0, "t", (t_ns - self.bag.start_ns) / 1e9)
        df.insert(0, "t_ns", t_ns)
        return df.sort_values("t_ns", kind="stable").reset_index(drop=True)


def gnss_stats_hits(store, spec: Spec, bag_no: int, w0: float, w1: float) -> tuple[int, float]:
    """GNSS buckets in the window whose lat/lon box meets the area: (count, total seconds of these
    buckets clipped to [w0, w1])."""
    import io

    key = f"meta/gnss_stats_bag{bag_no}.csv"
    try:
        s = pd.read_csv(io.BytesIO(store.read(key, 0)))
    except Exception as e:          # FileNotFoundError (local) or botocore ClientError (S3)
        raise RuntimeError(f"{key} is missing: run  python tools/convert.py --bag {bag_no}") from e
    b = spec.box
    hit = ((s["t1"] >= w0) & (s["t0"] <= w1) & (s["lat_max"] >= b.lat_min) & (s["lat_min"] <= b.lat_max)
           & (s["lon_max"] >= b.lon_min) & (s["lon_min"] <= b.lon_max))
    clip = (np.minimum(s["t1"], w1) - np.maximum(s["t0"], w0)).clip(lower=0)
    return int(hit.sum()), float(clip[hit].sum())


def with_base(est: float | None, t_open: float) -> float | None:
    """Estimate plus the measured index read (t_open), which the cost function leaves out."""
    return None if est is None else est + t_open


def run(spec: Spec, store, L: C.LayoutConsts, mode: str = "auto", bag_no: int = 5,
        estimate_only: bool = False, cond_rows: pd.DataFrame | None = None) -> dict:
    """Run one test. mode: auto | L2 | L3 | L4. Returns one log row (dict)."""
    T: dict[str, float] = {}
    t_start = time.perf_counter()
    store.reset()
    key = bag_key(bag_no)
    bag = read_index(store, key)
    T["t_open"] = time.perf_counter() - t_start                     # open + bag header + index

    # ---- point A: inputs from chunk info
    t0 = time.perf_counter()
    tp = topics()
    cconns, tconns = set(bag.conns_for_topic(tp[spec.cond])), set(bag.conns_for_topic(tp[spec.target]))
    w0, w1 = spec.window_range(bag.duration_s)
    lo, hi = w0 - spec.tol, w1 + spec.tol
    s0 = bag.start_ns                      # property = min over all chunks: compute once, not per chunk
    starts = np.array([(c.start_ns - s0) / 1e9 for c in bag.chunks])
    ends = np.array([(c.end_ns - s0) / 1e9 for c in bag.chunks])
    pos = np.array([c.chunk_pos for c in bag.chunks] + [bag.index_pos], dtype=np.int64)
    sizes = np.diff(pos)
    has_c = np.array([bool(cconns & c.counts.keys()) for c in bag.chunks])
    has_t = np.array([bool(tconns & c.counts.keys()) for c in bag.chunks])
    cmask = has_c & (ends >= lo) & (starts <= hi)
    tmask = has_t & (ends >= w0) & (starts <= w1)
    D = w1 - w0
    x = C.BagInputs(D=D, r_c=cmask.sum() / D, r_t=tmask.sum() / D, r_ct=(cmask & tmask).sum() / D,
                    b_c=float(sizes[cmask].mean()) if cmask.any() else 0.0,
                    b_t=float(sizes[tmask].mean()) if tmask.any() else 0.0)
    T_ub = None
    if spec.point_a in ("stats", "stats_clip"):
        if spec.cond != "gnss":
            raise ValueError(f"test {spec.id}: point A + GNSS statistics needs cond = gnss")
        n_hit, clipped = gnss_stats_hits(store, spec, bag_no, w0, w1)
        T_ub = min(clipped, w1 - w0) if spec.point_a == "stats_clip" else spec.ell * n_hit
    a = C.bag_point_a(x, L, T_ub, spec.l2_includes_condition)
    T["t_point_a"] = time.perf_counter() - t0
    run_phase1 = a.point_a == "phase1" if mode == "auto" else mode in ("L3", "L4")

    out = {"approach": "bag", "test": spec.id, "label": spec.label, "mode": mode, "w0": w0, "w1": w1,
           "point_a": a.point_a, "chunks_total": len(bag.chunks), "chunks_cond": int(cmask.sum()),
           "chunks_target": int(tmask.sum()), "outlier": is_outlier(T["t_open"]), **asdict(x)}

    def point_b(cdf: pd.DataFrame) -> tuple[C.Decision, dict[str, np.ndarray], int]:
        """Intervals of the cond rows in [lo, hi] and the point-B decision."""
        cdf = cdf[(cdf["t"] >= lo) & (cdf["t"] <= hi)]
        t, inside = cdf["t"].to_numpy(), spec.cond_mask(cdf)
        iv = {lvl: filter_intervals(t, inside, lvl, spec, w0, w1) for lvl in ("exact", "L3", "L4")}
        return C.bag_point_b(x, L, iv["L3"], iv["L4"], a), iv, int(inside.sum())

    if estimate_only:
        b = point_b(cond_rows)[0] if cond_rows is not None else None
        auto = "L2" if a.point_a == "L2" else (b.level if b else None)
        lv = auto if mode == "auto" else mode
        f = C.log_fields(b or a, lv, auto)
        return {**out, **f, "est_with_base": with_base(f["est_cost"], T["t_open"]), "level": lv, "level_auto": auto,
                "point_b": b.point_b if b else None, **T, "requests": store.requests, "bytes_read": store.bytes}

    rd = Reader(store, key, bag, {**{k: spec.cond for k in cconns}, **{k: spec.target for k in tconns}})
    t0 = time.perf_counter()
    n1 = rd.read(cmask | tmask if not run_phase1 else cmask)       # L2 read, or phase 1
    T["t_l2_read" if not run_phase1 else "t_phase1"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    b, iv, n_inside = point_b(rd.frame(spec.cond))
    T["t_point_b"] = time.perf_counter() - t0
    level_auto = "L2" if a.point_a == "L2" else b.level
    level = level_auto if mode == "auto" else mode
    n2 = 0
    if run_phase1:                                                  # phase 2
        t0 = time.perf_counter()
        meet = np.zeros(len(starts), bool)
        for s, e in iv[level]:
            meet |= (ends >= s) & (starts <= e)
        n2 = rd.read(tmask & meet)
        T["t_phase2"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tdf = rd.frame(spec.target)
    keep = ((tdf["t"] >= w0) & (tdf["t"] <= w1)).to_numpy() & in_intervals(tdf["t"], iv["exact"])
    res = tdf[keep]
    T["t_join"] = time.perf_counter() - t0
    T["end_to_end"] = time.perf_counter() - t_start
    f = C.log_fields(b, level, level_auto)
    return {**out, **f, "est_with_base": with_base(f["est_cost"], T["t_open"]), "level": level, "level_auto": level_auto, "point_b": b.point_b, **T,
            "t_read": rd.t_read, "t_decompress": rd.t_dec, "t_decode": rd.t_decode,
            "chunks_read": int(rd.done.sum()), "chunks_phase1": n1, "chunks_phase2": n2,
            "requests": store.requests, "bytes_read": store.bytes, "bytes_decompressed": rd.bytes_dec,
            "bytes_compressed_in": rd.bytes_in, "v_dec_run": rd.bytes_in / rd.t_dec if rd.t_dec > 0 else None,
            "n_cond_inside": n_inside, "rows_out": len(res), "result_hash": result_hash(res["t_ns"])}
