"""M2 feasibility check (CLAUDE.md section 11): how many IMU blocks each pruning level reads.

    python tools/m2_feasibility.py [--write-areas]

Uses only the GNSS track (results/eda/gnss_track.csv), bag metadata (results/eda/chunks.csv,
INDEX DATA offsets) and results/bench/read_costs.csv. Never decompresses a chunk.

Outputs in results/m2/: feasibility_parquet.csv, feasibility_bag.csv, intervals.txt, areas.png,
timeline_<area>.png, ratio_vs_ell.png, summary.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roslab import intervals as I  # noqa: E402
from roslab.bagfile import BagFile  # noqa: E402
from roslab.config import CONFIG_DIR, RESULTS_DIR, bag_path, load_yaml, topics  # noqa: E402

OUT = RESULTS_DIR / "m2"
ELLS = [1, 2, 5, 10, 30, 60]
M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON_EQ = 111320.0
MSG_RECORD_OVERHEAD = 46        # header_len + op/conn/time fields + data_len of a MESSAGE DATA record
COLORS = {"A": "#0072B2", "B": "#E69F00", "C": "#009E73"}     # Okabe-Ito
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"

# Areas in metres around the track centre (x = east, y = north). See results/eda/summary.md:
# the robot drives 4 east-west passes ~170 m long, 4-5 m apart in y.
AREAS_M = {
    "A": {"x": (55, 65), "y": (-10, 10), "note": "10 m strip across all passes, near the east turn"},
    "B": {"x": (30, 38), "y": (4, 8), "note": "small box on pass 2 only (control: L4 = L3)"},
    "C": {"x": (-40, 40), "y": (-10, 10), "note": "80 m strip across all passes (large area)"},
}


# ------------------------------------------------------------------ inputs
def load_track():
    g = pd.read_csv(RESULTS_DIR / "eda" / "gnss_track.csv").sort_values("t").reset_index(drop=True)
    lat0, lon0 = g["lat"].mean(), g["lon"].mean()
    g["x"] = (g["lon"] - lon0) * np.cos(np.deg2rad(lat0)) * M_PER_DEG_LON_EQ
    g["y"] = (g["lat"] - lat0) * M_PER_DEG_LAT
    return g, lat0, lon0


def boxes_latlon(lat0, lon0) -> dict[str, dict]:
    lat0, lon0 = float(lat0), float(lon0)
    k = float(np.cos(np.deg2rad(lat0)) * M_PER_DEG_LON_EQ)
    out = {}
    for name, a in AREAS_M.items():
        out[name] = {"lat_min": round(lat0 + a["y"][0] / M_PER_DEG_LAT, 9),
                     "lat_max": round(lat0 + a["y"][1] / M_PER_DEG_LAT, 9),
                     "lon_min": round(lon0 + a["x"][0] / k, 9), "lon_max": round(lon0 + a["x"][1] / k, 9),
                     "note": f"{a['note']}; x {a['x'][0]}..{a['x'][1]} m, y {a['y'][0]}..{a['y'][1]} m "
                             f"around lat0 {lat0:.9f}, lon0 {lon0:.9f}"}
    return out


def write_areas(boxes: dict) -> None:
    text = ("# Areas used by --area: a lat/lon box (degrees). Chosen in M2 (tools/m2_feasibility.py),\n"
            "# from the bag-5 GNSS track (results/eda/track.png, results/m2/areas.png).\n"
            "# X is the area used by the run matrix (CLAUDE.md section 7) = A.\n")
    data = dict(boxes)
    data["X"] = dict(boxes["A"], note="same as A")
    text += yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    (CONFIG_DIR / "areas.yaml").write_text(text, encoding="utf-8")


def load_constants() -> dict:
    df = pd.read_csv(RESULTS_DIR / "bench" / "read_costs.csv")
    c = df[df["const"].notna()]

    def val(name, cache):
        r = c[(c["const"] == name) & (c["cache"] == cache)]
        return float(r["value"].iloc[0])

    return {"fixed_ms": val("fixed_cost_per_read", "cold"), "read_mbps": val("read_speed", "cold"),
            "bz2_mbps": val("bz2_decompress_speed", "any"),
            "fixed_ms_warm": val("fixed_cost_per_read", "warm")}


def imu_bytes_per_second(bag: BagFile, imu_conns: set[int], w0: float, w1: float, n_chunks: int = 300):
    """Mean IMU message size from INDEX DATA offsets (no decompression) x IMU rate."""
    lo, hi = bag.start_ns + int(w0 * 1e9), bag.start_ns + int(w1 * 1e9)
    sel = [c for c in bag.chunks if c.end_ns >= lo and c.start_ns <= hi and imu_conns & c.counts.keys()]
    sel = [sel[i] for i in np.linspace(0, len(sel) - 1, min(n_chunks, len(sel))).astype(int)]
    sizes = []
    for c in sel:
        ch = bag.read_chunk_header(c)
        ent = bag.read_index_entries(c, ch)
        off = np.concatenate([e["offset"] for e in ent.values()]).astype(np.int64)
        conn = np.concatenate([np.full(len(e), k) for k, e in ent.items()])
        o = np.argsort(off)
        off, conn = off[o], conn[o]
        rec = np.diff(np.append(off, ch.uncompressed_size))
        sizes.append(rec[np.isin(conn, list(imu_conns))] - MSG_RECORD_OVERHEAD)
    s = np.concatenate(sizes)
    size = float(np.median(s))           # median: robust to CONNECTION records between messages
    n_msgs = sum(c.counts.get(k, 0) for c in bag.chunks for k in imu_conns)
    rate = n_msgs / bag.duration_s
    return size, rate, size * rate


# ------------------------------------------------------------------ helpers
def window_list(g, inside_A, duration):
    """whole bag, 300-420, and the shortest window holding 3 visits of A (10 s margin, 10 s grid)."""
    vis = I.visits(g["t"].to_numpy(), inside_A, gap=topics()["gap"])
    t = g["t"].to_numpy()
    best = None
    for i in range(len(vis) - 2):
        a, b = t[vis[i][0]], t[vis[i + 2][1]]
        if best is None or b - a < best[1] - best[0]:
            best = (a, b)
    wins = [("all", 0.0, float(np.ceil(duration))), ("300-420", 300.0, 420.0)]
    if best:
        w0 = max(0.0, np.floor((best[0] - 10) / 10) * 10)
        w1 = min(float(np.ceil(duration)), np.ceil((best[1] + 10) / 10) * 10)
        wins.append((f"A3:{w0:g}-{w1:g}", w0, w1))
    return wins


def clipped_sets(t, inside, qtype, w0, w1, tol, gap):
    """exact / L3 / L4 interval sets for the window. Built on the whole track, then clipped,
    so GNSS points just outside the window (within tol, or the ASOF predecessor) count."""
    after = np.inf
    ex = I.clip(I.build(t, inside, qtype, "exact", tol=tol, t_after_last=after), w0, w1)
    l4 = I.clip(I.build(t, inside, qtype, "L4", tol=tol, gap=gap, t_after_last=after), w0, w1)
    l3 = np.array([[ex[:, 0].min(), ex[:, 1].max()]]) if len(ex) else I.EMPTY
    return ex, l3, l4


def chunk_mask(starts, ends, iv) -> np.ndarray:
    m = np.zeros(len(starts), bool)
    for a, b in np.asarray(iv).reshape(-1, 2):
        m |= (ends >= a) & (starts <= b)
    return m


def runs(idx: np.ndarray) -> int:
    return 0 if len(idx) == 0 else int(1 + np.sum(np.diff(np.sort(idx)) > 1))


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-areas", action="store_true", help="write A, B, C (and X = A) to config/areas.yaml")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = topics()
    tol, gap = float(cfg["tol"]), float(cfg["gap"])

    g_all, lat0, lon0 = load_track()
    boxes = boxes_latlon(lat0, lon0)
    if a.write_areas:
        write_areas(boxes)
        print("wrote config/areas.yaml")
    consts = load_constants()

    ch = pd.read_csv(RESULTS_DIR / "eda" / "chunks.csv")
    imu_col, gnss_col = cfg["imu"], cfg["gnss"]
    with BagFile(bag_path(5)) as bag:
        duration = bag.duration_s
        imu_size, imu_rate, imu_bps = imu_bytes_per_second(bag, set(bag.conns_for_topic(imu_col)), 300, 420)
    print(f"IMU: {imu_size:.0f} B/message (ROS serialized, from INDEX DATA offsets) x {imu_rate:.2f} Hz "
          f"= {imu_bps / 1e3:.1f} KB/s")
    print(f"constants: fixed {consts['fixed_ms']:.3f} ms/read (cold), bz2 {consts['bz2_mbps']:.1f} MB/s compressed")

    starts, ends = ch["start_s"].to_numpy(), ch["end_s"].to_numpy()
    has_imu, has_gnss = ch[imu_col].to_numpy() > 0, ch[gnss_col].to_numpy() > 0
    comp, uncomp = ch["bytes_compressed"].to_numpy(), ch["bytes_uncompressed"].to_numpy()

    def cost(mask):
        n = int(mask.sum())
        cb = int(comp[mask].sum())
        return {"chunks": n, "comp_MB": cb / 1e6, "uncomp_MB": uncomp[mask].sum() / 1e6,
                "runs": runs(np.flatnonzero(mask)),
                "time_s": n * consts["fixed_ms"] / 1e3 + cb / (consts["bz2_mbps"] * 1e6)}

    inside_A_all = I.in_area(g_all["lat"], g_all["lon"], I.Box.from_dict(boxes["A"]))
    windows = window_list(g_all, inside_A_all, duration)
    print("windows:", [(n, w0, w1) for n, w0, w1 in windows])

    variants = {"all": g_all, "rtk": g_all[g_all["status"] == 2].reset_index(drop=True)}
    rows_pq, rows_bag, lines = [], [], []
    for var, g in variants.items():
        t = g["t"].to_numpy()
        for area in AREAS_M:
            inside = I.in_area(g["lat"], g["lon"], I.Box.from_dict(boxes[area]))
            for wname, w0, w1 in windows:
                D = w1 - w0
                for qtype in (1, 2):
                    ex, l3, l4 = clipped_sets(t, inside, qtype, w0, w1, tol, gap)
                    k, span = len(l4), I.total_length(l3)
                    sum_w = I.total_length(l4)
                    if var == "all":
                        lines.append(f"[{area} | {wname} | type {qtype}] k={k}  span={span:.1f} s  "
                                     f"sum_w={sum_w:.1f} s  exact={I.total_length(ex):.1f} s  "
                                     f"span/sum_w={span / sum_w if sum_w else float('nan'):.2f}")
                        for j, (s, e) in enumerate(l4):
                            lines.append(f"    visit {j + 1}: [{s:8.2f}, {e:8.2f}]  length {e - s:6.2f} s")
                        if k > 1:
                            lines.append("    holes: " + ", ".join(f"{h:.1f}" for h in I.gaps(l4)) + " s")

                    # ---- approach 2: simulated row groups of ell seconds
                    for ell in ELLS:
                        n2 = len(I.blocks_hit([[w0, w1]], ell))
                        n3 = len(I.blocks_hit(l3, ell))
                        nex = len(I.blocks_hit(ex, ell))
                        for m in (0.0, float(ell)):
                            l4m = I.merge_intervals(l4, m)
                            n4 = len(I.blocks_hit(l4m, ell))
                            km, sw = len(l4m), I.total_length(l4m)
                            rows_pq.append({
                                "area": area, "window": wname, "w0": w0, "w1": w1, "type": qtype, "ell": ell,
                                "m": m, "variant": var, "k": km, "span_s": span, "sum_w_s": sw,
                                "exact_s": I.total_length(ex),
                                "blocks_L2": n2, "blocks_L3": n3, "blocks_L4": n4, "blocks_exact": nex,
                                "formula_L3": (span + ell) / ell if k else 0.0,
                                "formula_L4": (sw + km * ell) / ell if k else 0.0,
                                "f1": (span + ell) / D if k else 0.0, "fk": (sw + km * ell) / D if k else 0.0,
                                "MB_L2": n2 * ell * imu_bps / 1e6, "MB_L3": n3 * ell * imu_bps / 1e6,
                                "MB_L4": n4 * ell * imu_bps / 1e6,
                                "ratio_L2_L3": n2 / n3 if n3 else np.nan,
                                "ratio_L3_L4": n3 / n4 if n4 else np.nan})

                    # ---- approach 1: real chunks
                    win = (ends >= w0) & (starts <= w1)
                    l2 = win & has_imu
                    p1 = win & has_gnss
                    row = {"area": area, "window": wname, "w0": w0, "w1": w1, "type": qtype, "ell": np.nan,
                           "m": 0.0, "variant": var, "k": k}
                    for key, v in cost(l2).items():
                        row[f"L2_{key}"] = v
                    for lvl, iv in (("L3", l3), ("L4", l4)):
                        p2 = has_imu & chunk_mask(starts, ends, iv) & ~p1
                        tot = p1 | p2
                        c1, c2, ct = cost(p1), cost(p2), cost(tot)
                        row.update({f"{lvl}_phase1_chunks": c1["chunks"], f"{lvl}_phase2_chunks": c2["chunks"],
                                    **{f"{lvl}_{kk}": vv for kk, vv in ct.items()},
                                    f"{lvl}_phase1_time_s": c1["time_s"],
                                    f"{lvl}_phase1_share": c1["time_s"] / ct["time_s"] if ct["time_s"] else np.nan})
                    row["speedup_L2_L3"] = row["L2_time_s"] / row["L3_time_s"]
                    row["speedup_L3_L4"] = row["L3_time_s"] / row["L4_time_s"]
                    rows_bag.append(row)

    pq = pd.DataFrame(rows_pq)
    bg = pd.DataFrame(rows_bag)
    pq.to_csv(OUT / "feasibility_parquet.csv", index=False, float_format="%.4f")
    bg.to_csv(OUT / "feasibility_bag.csv", index=False, float_format="%.4f")
    header = [f"tol={tol} gap={gap}; IMU {imu_size:.0f} B/msg x {imu_rate:.2f} Hz = {imu_bps / 1e3:.1f} KB/s",
              f"windows: {windows}", ""]
    (OUT / "intervals.txt").write_text("\n".join(header + lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    same = (pq[pq.variant == "all"].drop(columns="variant").reset_index(drop=True)
            .equals(pq[pq.variant == "rtk"].drop(columns="variant").reset_index(drop=True)))
    print(f"variant rtk identical to all: {same}")

    write_summary(pq, bg, windows, imu_size, imu_rate, imu_bps, consts, same, tol, gap)
    plot_areas(g_all, boxes, inside_fn=lambda b: I.in_area(g_all["lat"], g_all["lon"], I.Box.from_dict(b)), gap=gap)
    for area in AREAS_M:
        plot_timeline(g_all, boxes, area, windows[-1], tol, gap)
    plot_ratio(pq, windows)
    print(f"wrote {OUT}")


# ------------------------------------------------------------------ outputs
def write_summary(pq, bg, windows, imu_size, imu_rate, imu_bps, consts, same, tol, gap, threshold=3.0):
    """results/m2/summary.md: setup, tables at ell = 10 s, ratio vs ell, and the step-6 conclusions."""
    allv = pq[(pq.variant == "all") & (pq.m == 0)]
    p = allv[allv.ell == 10]
    b = bg[bg.variant == "all"]
    out = ["# M2: kiểm tra khả thi (bag 5)", "",
           "Sinh tự động bởi `tools/m2_feasibility.py`. Chỉ dùng GNSS và metadata, không giải nén chunk.", "",
           "## Thiết lập", "",
           f"- tol = {tol} s, gap = {gap} s; IMU {imu_size:.0f} B/message × {imu_rate:.2f} Hz = {imu_bps / 1e3:.1f} KB/s "
           "(cỡ ROS serialize, suy từ offset trong INDEX DATA).",
           f"- Hằng số đọc (`results/bench/read_costs.csv`): {consts['fixed_ms']:.3f} ms mỗi lần đọc (cold), "
           f"bz2 {consts['bz2_mbps']:.1f} MB/s nén.",
           f"- Biến thể chỉ RTK giống hệt biến thể mọi điểm: {same}.", "",
           "| Vùng | x (m) | y (m) |", "|---|---|---|",
           *[f"| {n} | {a['x'][0]}…{a['x'][1]} | {a['y'][0]}…{a['y'][1]} |" for n, a in AREAS_M.items()],
           "", "Cửa sổ: " + ", ".join(f"`{n}` = {w0:g}–{w1:g} s" for n, w0, w1 in windows)
           + " (cửa sổ cuối: ngắn nhất có 3 lần ghé A, thêm 10 s mỗi đầu).", "",
           "## Approach 2: row group giả lập, ℓ = 10 s, m = 0", "",
           "| Vùng | Cửa sổ | Type | k | span (s) | Σw (s) | L2 | L3 | L4 | công thức L3 / L4 | L2/L3 | **L3/L4** |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in p.itertuples():
        out.append(f"| {r.area} | {r.window} | {r.type} | {r.k} | {r.span_s:.1f} | {r.sum_w_s:.1f} | "
                   f"{r.blocks_L2} | {r.blocks_L3} | {r.blocks_L4} | {r.formula_L3:.1f} / {r.formula_L4:.1f} | "
                   f"{r.ratio_L2_L3:.2f} | **{r.ratio_L3_L4:.2f}** |")
    out += ["", "## Approach 1: chunk thật trên bag (ước lượng thời gian, cache cold)", "",
            "| Vùng | Cửa sổ | Type | k | L2 chunk / GB / s | L3 chunk / GB / s | L4 chunk / GB / s | pha 1 % (L3, L4) | L2/L3 | L3/L4 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in b.itertuples():
        out.append(f"| {r.area} | {r.window} | {r.type} | {r.k} | "
                   f"{r.L2_chunks} / {r.L2_comp_MB / 1e3:.2f} / {r.L2_time_s:.0f} | "
                   f"{r.L3_chunks} / {r.L3_comp_MB / 1e3:.2f} / {r.L3_time_s:.0f} | "
                   f"{r.L4_chunks} / {r.L4_comp_MB / 1e3:.2f} / {r.L4_time_s:.0f} | "
                   f"{r.L3_phase1_share:.0%}, {r.L4_phase1_share:.0%} | {r.speedup_L2_L3:.2f} | {r.speedup_L3_L4:.2f} |")

    # ratio L3/L4 vs ell (type 1)
    piv = allv[allv.type == 1].pivot_table(index=["area", "window"], columns="ell", values="ratio_L3_L4")
    out += ["", "## Tỉ lệ L3/L4 theo ℓ (type 1, m = 0)", "",
            "| Vùng, cửa sổ | " + " | ".join(f"ℓ = {e}" for e in piv.columns) + " |",
            "|---|" + "---|" * len(piv.columns)]
    out += [f"| {a}, {w} | " + " | ".join(f"{v:.2f}" for v in row) + " |" for (a, w), row in piv.iterrows()]

    # ---- check metrics (numbers only), ell = 10 s
    t12 = p.pivot_table(index=["area", "window"], columns="type", values="blocks_L4")
    m_eff = pq[(pq.variant == "all") & (pq.ell == 10)].pivot_table(
        index=["area", "window", "type"], columns="m", values="blocks_L4")
    k2 = p[p.k >= 2]
    bk = b[b.k >= 2]
    metrics = [
        (f"số tổ hợp có L3/L4 ≥ {threshold:g}", int((p.ratio_L3_L4 >= threshold).sum())),
        ("L3/L4 lớn nhất, cửa sổ 300-420", f"{p[p.window == '300-420'].ratio_L3_L4.max():.2f}"),
        ("chênh lệch khối L4 lớn nhất giữa type 1 và type 2", int((t12[1] - t12[2]).abs().max())),
        ("số tổ hợp mà m = ℓ đổi số khối L4", int((m_eff.iloc[:, 0] != m_eff.iloc[:, 1]).sum())),
        ("sai lệch lớn nhất giữa số khối đếm và công thức (k ≥ 2)",
         f"{np.maximum((k2.blocks_L3 - k2.formula_L3).abs(), (k2.blocks_L4 - k2.formula_L4).abs()).max():.2f}"),
        ("bag: pha 1 / chi phí L3 (k ≥ 2)", f"{bk.L3_phase1_share.min():.0%} – {bk.L3_phase1_share.max():.0%}"),
        ("bag: pha 1 / chi phí L4 (k ≥ 2)", f"{bk.L4_phase1_share.min():.0%} – {bk.L4_phase1_share.max():.0%}"),
        ("bag: L3/L4 lớn nhất", f"{bk.speedup_L3_L4.max():.2f}"),
    ]
    out += ["", "## Chỉ số (ℓ = 10 s, m = 0, mọi điểm GNSS)", "", "| Chỉ số | Giá trị |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in metrics]
    (OUT / "summary.md").write_text("\n".join(out) + "\n", encoding="utf-8")


def _style(ax):
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.5)


def plot_areas(g, boxes, inside_fn, gap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.plot(g["x"], g["y"], color="#c8ccd1", lw=1, zorder=1)
    sc = ax.scatter(g["x"], g["y"], c=g["t"], cmap="Greys", vmin=-200, vmax=g["t"].max(), s=3, zorder=2)
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("t (s)", color=MUTED)
    t = g["t"].to_numpy()
    for name, a in AREAS_M.items():
        (x0, x1), (y0, y1) = a["x"], a["y"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=True, fc=COLORS[name], alpha=0.12,
                               ec=COLORS[name], lw=1.5, zorder=3))
        ax.text(x0, y1 + 0.4, f"{name}", color=INK, fontsize=11, fontweight="bold")
        for j, (f, l) in enumerate(I.visits(t, inside_fn(boxes[name]), gap)):
            seg = g.iloc[f:l + 1]
            ax.plot(seg["x"], seg["y"], color=COLORS[name], lw=2.5, zorder=4)
            mid = seg.iloc[len(seg) // 2]
            dx = (j - 1.5) * min(6.0, (a["x"][1] - a["x"][0]) / 4)     # passes 1 and 3 overlap: spread labels
            ax.annotate(f"{name}{j + 1}\n{seg['t'].iloc[0]:.0f}s", (mid["x"] + dx, mid["y"]), fontsize=7, color=INK,
                        ha="center", va="center", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=COLORS[name], lw=0.8))
    ax.set_aspect("equal")
    ax.set_xlabel("x east (m)", color=MUTED)
    ax.set_ylabel("y north (m)", color=MUTED)
    ax.set_title("GNSS track and areas A, B, C (visits numbered, with start time)", color=INK, loc="left")
    _style(ax)
    fig.savefig(OUT / "areas.png", dpi=130)
    plt.close(fig)


def plot_timeline(g, boxes, area, window, tol, gap, ell=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    wname, w0, w1 = window
    t = g["t"].to_numpy()
    inside = I.in_area(g["lat"], g["lon"], I.Box.from_dict(boxes[area]))
    col = COLORS[area]
    fig, axes = plt.subplots(2, 1, figsize=(14, 5.5), constrained_layout=True, sharex=True)
    for ax, qtype in zip(axes, (1, 2)):
        ex, l3, l4 = clipped_sets(t, inside, qtype, w0, w1, tol, gap)
        rows = [("GNSS in area", None), ("exact", ex), ("L3: 1 interval", l3), (f"L4: {len(l4)} visits", l4),
                ("L2 row groups", np.array([[w0, w1]])), ("L3 row groups", l3), ("L4 row groups", l4)]
        for i, (label, iv) in enumerate(rows):
            y = len(rows) - 1 - i
            if iv is None:
                m = inside & (t >= w0) & (t <= w1)
                ax.vlines(t[m], y - 0.3, y + 0.3, color=col, lw=0.6)
            elif "row groups" in label:
                for j in I.blocks_hit(iv, ell):
                    ax.add_patch(Rectangle((j * ell, y - 0.35), ell, 0.7, fc=col if label[1] != "2" else "#9aa4ae",
                                           alpha=0.35 if label[1] != "2" else 0.4, ec="white", lw=1))
            else:
                for s, e in iv:
                    ax.add_patch(Rectangle((s, y - 0.3), max(e - s, 0.3), 0.6, fc=col, ec="none"))
        n = {lbl: len(I.blocks_hit(iv, ell)) for lbl, iv in rows[4:]}
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8, color=MUTED)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_xlim(w0, w1)
        ax.set_title(f"type {qtype}: row groups read (ℓ = {ell} s): "
                     + ", ".join(f"{k.split()[0]} = {v}" for k, v in n.items()), color=INK, loc="left", fontsize=10)
        _style(ax)
    axes[-1].set_xlabel(f"t since bag start (s), window {wname}", color=MUTED)
    fig.suptitle(f"Area {area}", color=INK, x=0.01, ha="left")
    fig.savefig(OUT / f"timeline_{area}.png", dpi=130)
    plt.close(fig)


def plot_ratio(pq, windows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pq[(pq.m == 0) & (pq.variant == "all")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True, sharey=True)
    styles = {windows[0][0]: ":", windows[1][0]: "--", windows[-1][0]: "-"}
    markers = {"A": "o", "B": "s", "C": "^"}
    zorder = {"A": 3, "B": 5, "C": 2}          # B (often exactly 1.0) drawn on top of C
    series = [(a, w) for a in AREAS_M for w in styles]
    for ax, qtype in zip(axes, (1, 2)):
        for i, (area, wname) in enumerate(series):
            s = d[(d.area == area) & (d.window == wname) & (d.type == qtype)].sort_values("ell")
            dodge = 1 + 0.03 * (i - (len(series) - 1) / 2)       # small x shift so equal values stay visible
            empty = s.ratio_L3_L4.isna().all()
            ax.plot(s.ell * dodge, s.ratio_L3_L4, styles[wname], marker=markers[area], ms=5, lw=2,
                    color=COLORS[area], zorder=zorder[area],
                    label=f"{area}, {wname}" + (" (no visit)" if empty else ""))
        ax.axhline(3, color=MUTED, lw=1)
        ax.text(1, 3.1, "threshold 3", color=MUTED, fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks(ELLS)
        ax.set_xticklabels([str(e) for e in ELLS])
        ax.set_xlabel("row group length ℓ (s)", color=MUTED)
        ax.set_title(f"type {qtype}", color=INK, loc="left")
        _style(ax)
    axes[0].set_ylabel("L3 / L4 row groups read", color=MUTED)
    axes[1].legend(frameon=False, fontsize=8, ncol=1, loc="upper right")
    fig.savefig(OUT / "ratio_vs_ell.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
