"""EDA of a ROS 1 bag, without ever loading the whole file into RAM.

    python eda_bag.py --bag 5 --action info        # index only: topics, types, counts, Hz
    python eda_bag.py --bag 5 --action chunks      # one line per chunk -> results/eda/chunks.csv
    python eda_bag.py --bag 5 --action topic-mix   # how mixed the chunks are (is L1 useful?)
    python eda_bag.py --bag 5 --action rates       # Hz, period jitter, message size (sampled chunks)
    python eda_bag.py --bag 5 --action track       # decode GNSS only -> results/eda/track.png

What each action reads from disk:
    info, topic-mix : the index section at the end of the file only.
    chunks          : + the small CHUNK record header of every chunk (no decompression).
    rates           : + INDEX DATA of the chunks in --window, and decompresses --limit of them.
    track           : decompresses every chunk that holds a GNSS message, one at a time.

Time t = seconds since the first message of the bag (bag record time, same clock as chunk info).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from roslab.bagfile import BagError, BagFile, iter_messages
from roslab.config import RESULTS_DIR, bag_path, topics

EDA_DIR = RESULTS_DIR / "eda"
ROLES = ("imu", "gnss", "image", "odom")


# ---------------------------------------------------------------- helpers
def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n:.0f} B"
        n /= 1024
    return f"{n:.1f} GB"


def topic_table(bag: BagFile) -> pd.DataFrame:
    """One row per topic (several connections can share a topic)."""
    counts = bag.topic_counts()
    rows: dict[str, dict] = {}
    for c in bag.connections.values():
        r = rows.setdefault(c.topic, {"topic": c.topic, "type": c.msgtype, "conns": [], "messages": 0})
        r["conns"].append(c.conn)
        r["messages"] += counts.get(c.conn, 0)
    df = pd.DataFrame(rows.values()).sort_values("topic").reset_index(drop=True)
    df["hz"] = df["messages"] / bag.duration_s
    return df


def role_conns(bag: BagFile) -> dict[str, set[int]]:
    """Role (imu, gnss, ...) -> connection ids, using config/topics.yaml."""
    cfg = topics()
    return {role: set(bag.conns_for_topic(cfg[role])) for role in ROLES if cfg.get(role)}


def chunk_bytes_approx(bag: BagFile) -> np.ndarray:
    """On-disk bytes per chunk from the gap between consecutive chunk positions.

    Includes the chunk's INDEX DATA records (small). No extra disk reads.
    """
    pos = np.array([c.chunk_pos for c in bag.chunks] + [bag.index_pos], dtype=np.int64)
    return np.diff(pos)


def chunks_in_window(bag: BagFile, w0: float, w1: float):
    lo, hi = bag.start_ns + int(w0 * 1e9), bag.start_ns + int(w1 * 1e9)
    return [c for c in bag.chunks if c.end_ns >= lo and c.start_ns <= hi]


def progress(i: int, n: int, t0: float, nbytes: int | None = None) -> None:
    el = time.perf_counter() - t0
    rate = "" if nbytes is None else f", {fmt_bytes(nbytes)} read, {nbytes / 1e6 / max(el, 1e-9):.0f} MB/s"
    print(f"\r  {i}/{n} chunks, {el:.0f} s{rate}   ", end="", flush=True)


# ---------------------------------------------------------------- actions
def act_info(bag: BagFile, a) -> None:
    ch0 = bag.read_chunk_header(bag.chunks[0])
    print(f"file          {bag.path.name}  ({fmt_bytes(bag.file_size)})")
    print(f"index_pos     {bag.index_pos}  (index section = {fmt_bytes(bag.index_bytes)})")
    print(f"connections   {bag.conn_count}")
    print(f"chunks        {bag.chunk_count}  (compression of chunk 0: {ch0.compression})")
    print(f"start / end   {bag.start_ns / 1e9:.3f} / {bag.end_ns / 1e9:.3f}  (unix s)")
    print(f"duration      {bag.duration_s:.1f} s")
    print(f"mean chunk    {fmt_bytes((bag.index_pos - bag.chunks[0].chunk_pos) / bag.chunk_count)} on disk, "
          f"{bag.duration_s / bag.chunk_count * 1e3:.0f} ms of data\n")

    df = topic_table(bag)
    print(f"{'topic':50s} {'type':40s} {'messages':>9s} {'Hz':>8s}")
    for r in df.itertuples():
        print(f"{r.topic:50s} {r.type:40s} {r.messages:9d} {r.hz:8.2f}")

    print("\nconfig/topics.yaml check:")
    cfg = topics()
    for role in ROLES:
        name = cfg.get(role)
        if name in set(df["topic"]):
            print(f"  {role:6s} {name:45s} OK")
        else:
            want = {"imu": "Imu", "gnss": "NavSatFix", "image": "Image", "odom": "Odometry"}[role]
            cand = [t for t, ty in zip(df["topic"], df["type"]) if ty.endswith(want)]
            print(f"  {role:6s} {name!s:45s} !! NOT IN BAG. candidates ({want}): {cand}")

    out = EDA_DIR / "topics.csv"
    df.assign(conns=df["conns"].map(lambda v: " ".join(map(str, v)))).to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(RESULTS_DIR.parent)}")


def act_chunks(bag: BagFile, a) -> None:
    df_t = topic_table(bag)
    conn2topic = {c.conn: c.topic for c in bag.connections.values()}
    topic_list = list(df_t["topic"])
    rows, t0 = [], time.perf_counter()
    n = len(bag.chunks)
    for i, c in enumerate(bag.chunks):
        h = bag.read_chunk_header(c)        # small read, no decompression
        per_topic = dict.fromkeys(topic_list, 0)
        for conn, k in c.counts.items():
            per_topic[conn2topic[conn]] += k
        rows.append({"chunk": c.idx, "chunk_pos": c.chunk_pos, "compression": h.compression,
                     "bytes_compressed": h.data_len, "bytes_uncompressed": h.uncompressed_size,
                     "start_s": (c.start_ns - bag.start_ns) / 1e9, "end_s": (c.end_ns - bag.start_ns) / 1e9,
                     "n_messages": c.n_messages, "n_topics": sum(v > 0 for v in per_topic.values()),
                     **per_topic})
        if i % 2000 == 0 or i == n - 1:
            progress(i + 1, n, t0)
    print()
    df = pd.DataFrame(rows)
    out = EDA_DIR / "chunks.csv"
    df.to_csv(out, index=False)

    show = ["chunk", "chunk_pos", "compression", "bytes_compressed", "bytes_uncompressed",
            "start_s", "end_s", "n_messages", "n_topics"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[show].head(a.limit).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    dur = df["end_s"] - df["start_s"]
    print(f"\n{n} chunks, compression: {df['compression'].value_counts().to_dict()}")
    print(f"compressed   total {fmt_bytes(df['bytes_compressed'].sum())}, "
          f"mean {fmt_bytes(df['bytes_compressed'].mean())}, max {fmt_bytes(df['bytes_compressed'].max())}")
    print(f"uncompressed total {fmt_bytes(df['bytes_uncompressed'].sum())}, "
          f"ratio {df['bytes_uncompressed'].sum() / df['bytes_compressed'].sum():.2f}x")
    print(f"time span per chunk: median {dur.median() * 1e3:.0f} ms, max {dur.max() * 1e3:.0f} ms")
    print(f"topics per chunk: median {df['n_topics'].median():.0f}, min {df['n_topics'].min()}, "
          f"max {df['n_topics'].max()}")
    print(f"wrote {out.relative_to(RESULTS_DIR.parent)}  ({len(df.columns)} columns, per-topic counts included)")


def act_topic_mix(bag: BagFile, a) -> None:
    roles = role_conns(bag)
    missing = [r for r, s in roles.items() if not s]
    if missing:
        print(f"!! roles not found in bag (fix config/topics.yaml, see --action info): {missing}")
    has = {r: np.array([bool(s & c.counts.keys()) for c in bag.chunks]) for r, s in roles.items() if s}
    nbytes = chunk_bytes_approx(bag)
    total_b, n = nbytes.sum(), len(bag.chunks)

    def line(label, mask):
        print(f"  {label:32s} {mask.sum():7d} / {n} chunks ({mask.mean():6.1%})   "
              f"{fmt_bytes(nbytes[mask].sum()):>10s} ({nbytes[mask].sum() / total_b:6.1%} of bytes)")

    print("chunks that contain ... (bytes = on-disk size, from chunk positions)")
    for r, m in has.items():
        line(r, m)
    if "imu" in has and "image" in has:
        line("image AND imu", has["imu"] & has["image"])
    if "imu" in has and "gnss" in has:
        line("gnss AND imu", has["imu"] & has["gnss"])
    if all(r in has for r in ("imu", "gnss", "image")):
        line("image AND imu AND gnss", has["imu"] & has["gnss"] & has["image"])

    print("\nwhat L1 (skip chunks without the needed topics) would read:")
    for need in (("imu",), ("gnss",), ("imu", "gnss")):
        if all(r in has for r in need):
            m = np.logical_or.reduce([has[r] for r in need])
            print(f"  topics {'+'.join(need):10s} -> read {m.mean():6.1%} of chunks, "
                  f"{nbytes[m].sum() / total_b:6.1%} of bytes")

    n_topics = np.array([len({bag.connections[c].topic for c in ch.counts}) for ch in bag.chunks])
    vals, cnts = np.unique(n_topics, return_counts=True)
    print("\ntopics per chunk (count: chunks): " + ", ".join(f"{v}: {k}" for v, k in zip(vals, cnts)))


def act_rates(bag: BagFile, a) -> None:
    w0, w1 = a.window
    sel = chunks_in_window(bag, w0, w1)
    if not sel:
        sys.exit(f"no chunk in window {w0}..{w1} s")
    conn2topic = {c.conn: c.topic for c in bag.connections.values()}
    print(f"window {w0}..{w1} s: {len(sel)} chunks (reading their INDEX DATA, no decompression)")

    times: dict[str, list[np.ndarray]] = {}
    for c in sel:
        for conn, t in bag.read_index_data(c).items():
            times.setdefault(conn2topic[conn], []).append(t)

    # message sizes: decompress `limit` chunks spread evenly over the window
    k = min(a.limit, len(sel))
    sample = [sel[i] for i in np.linspace(0, len(sel) - 1, k).astype(int)]
    sizes: dict[str, list[int]] = {}
    comp_b = decomp_b = 0
    t_dec = 0.0
    for c in sample:
        h, raw = bag.read_chunk_raw(c)
        t0 = time.perf_counter()
        buf = bag.decompress(h, raw)
        t_dec += time.perf_counter() - t0
        comp_b += len(raw)
        decomp_b += len(buf)
        for conn, _, data in iter_messages(buf):
            sizes.setdefault(conn2topic[conn], []).append(len(data))
        del raw, buf

    df_all = topic_table(bag).set_index("topic")
    rows = []
    lo, hi = bag.start_ns + int(w0 * 1e9), bag.start_ns + int(w1 * 1e9)
    for topic in df_all.index:
        t = np.sort(np.concatenate(times.get(topic, [np.empty(0, np.int64)])))
        t = t[(t >= lo) & (t <= hi)]
        dt = np.diff(t) / 1e9
        s = np.array(sizes.get(topic, []))
        rows.append({"topic": topic, "type": df_all.loc[topic, "type"], "hz_bag": df_all.loc[topic, "hz"],
                     "n_window": len(t), "hz_window": len(t) / (w1 - w0),
                     "dt_median": np.median(dt) if len(dt) else np.nan,
                     "dt_p95": np.percentile(dt, 95) if len(dt) else np.nan,
                     "dt_max": dt.max() if len(dt) else np.nan,
                     "size_mean": s.mean() if len(s) else np.nan,
                     "size_max": s.max() if len(s) else np.nan, "size_n": len(s)})
    df = pd.DataFrame(rows)
    df["mb_per_s"] = df["hz_bag"] * df["size_mean"] / 1e6
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(df.drop(columns="type").to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nsampled {k} chunks: {fmt_bytes(comp_b)} -> {fmt_bytes(decomp_b)} "
          f"(ratio {decomp_b / max(comp_b, 1):.2f}x), decompress {t_dec:.2f} s "
          f"= {comp_b / 1e6 / max(t_dec, 1e-9):.0f} MB/s of compressed input")

    g = topics().get("gnss")
    row = df[df["topic"] == g]
    if len(row) and not np.isnan(row["dt_median"].iloc[0]):
        print(f"GNSS {g}: median period {row['dt_median'].iloc[0]:.3f} s -> suggested --tol "
              f"{row['dt_median'].iloc[0] / 2:.3f} (half a period)")
    out = EDA_DIR / "rates.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out.relative_to(RESULTS_DIR.parent)}")


def act_track(bag: BagFile, a) -> None:
    from rosbags.typesys import Stores, get_typestore

    topic = a.topic or topics().get("gnss")
    conns = set(bag.conns_for_topic(topic))
    if not conns:
        cand = [c.topic for c in bag.connections.values() if c.msgtype.endswith("NavSatFix")]
        sys.exit(f"!! topic {topic!r} not in bag. NavSatFix topics: {sorted(set(cand))}")
    msgtype = bag.connections[next(iter(conns))].typestore_name
    store = get_typestore(Stores.ROS1_NOETIC)

    sel = [c for c in bag.chunks if conns & c.counts.keys()]
    if a.limit_track:
        sel = sel[:a.limit_track]
    print(f"{topic}: {sum(sum(c.counts.get(k, 0) for k in conns) for c in sel)} messages "
          f"in {len(sel)} / {len(bag.chunks)} chunks; decompressing one chunk at a time")

    rows, nbytes, t0 = [], 0, time.perf_counter()
    for i, c in enumerate(sel):
        h, buf = bag.read_chunk(c)
        nbytes += h.data_len
        for _, t_ns, data in iter_messages(buf, conns):
            m = store.deserialize_ros1(bytes(data), msgtype)
            rows.append((t_ns, m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec,
                         m.latitude, m.longitude, m.altitude, m.status.status))
        del buf
        if i % 200 == 0 or i == len(sel) - 1:
            progress(i + 1, len(sel), t0, nbytes)
    print()

    df = pd.DataFrame(rows, columns=["t_ns", "stamp_ns", "lat", "lon", "alt", "status"])
    df.insert(0, "t", (df["t_ns"] - bag.start_ns) / 1e9)
    df = df.sort_values("t_ns").reset_index(drop=True)
    out_csv = EDA_DIR / "gnss_track.csv"
    df.to_csv(out_csv, index=False, float_format="%.9f")

    dt = np.diff(df["t"].to_numpy())
    print(f"{len(df)} fixes, t {df['t'].min():.1f}..{df['t'].max():.1f} s, "
          f"period median {np.median(dt):.3f} s, max gap {dt.max():.2f} s")
    print(f"lat {df['lat'].min():.7f}..{df['lat'].max():.7f}, lon {df['lon'].min():.7f}..{df['lon'].max():.7f}")
    print(f"status counts: {df['status'].value_counts().sort_index().to_dict()}  (-1 = no fix)")
    print(f"stamp - record time: median {np.median((df['stamp_ns'] - df['t_ns']) / 1e9):.3f} s")
    plot_track(df, a.window, EDA_DIR / "track.png", topic)
    print(f"wrote {out_csv.relative_to(RESULTS_DIR.parent)} and results/eda/track.png")


def plot_track(df: pd.DataFrame, window, out: Path, topic: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[df["status"] >= 0] if (df["status"] >= 0).any() else df
    w0, w1 = window
    win = d[(d["t"] >= w0) & (d["t"] <= w1)]
    ink, muted, grid = "#1f2328", "#6e7781", "#d0d7de"

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1])
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(d["lon"], d["lat"], color="#c8ccd1", lw=1, zorder=1, label="whole bag")
    sc = ax.scatter(win["lon"], win["lat"], c=win["t"], cmap="Blues", vmin=w0 - (w1 - w0) * 0.3, vmax=w1,
                    s=10, zorder=2, label=f"window {w0:g}..{w1:g} s")
    ax.plot(d["lon"].iloc[0], d["lat"].iloc[0], "o", ms=8, mfc="white", mec=ink, zorder=3, label="start")
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label("t in window (s)", color=muted)
    ax.set_xlabel("longitude (deg)", color=muted)
    ax.set_ylabel("latitude (deg)", color=muted)
    ax.set_title(f"GNSS track ({topic})", color=ink, loc="left")
    ax.set_aspect(1 / np.cos(np.deg2rad(d["lat"].mean())))   # metres look square
    ax.ticklabel_format(useOffset=False)
    ax.grid(color=grid, lw=0.5)
    ax.legend(loc="best", frameon=False)

    for k, (col, lab) in enumerate((("lat", "latitude (deg)"), ("lon", "longitude (deg)"))):
        a2 = fig.add_subplot(gs[k, 1])
        a2.plot(d["t"], d[col], color="#2b6cb0", lw=1.5)
        a2.axvspan(w0, w1, color="#2b6cb0", alpha=0.08, lw=0)
        a2.set_ylabel(lab, color=muted)
        a2.ticklabel_format(useOffset=False)
        a2.grid(color=grid, lw=0.5)
        if k == 1:
            a2.set_xlabel("t since bag start (s)", color=muted)
    for axis in fig.axes:
        for s in axis.spines.values():
            s.set_color(grid)
        axis.tick_params(colors=muted, labelsize=8)
    fig.savefig(out, dpi=130)
    plt.close(fig)


ACTIONS = {"info": act_info, "chunks": act_chunks, "topic-mix": act_topic_mix,
           "rates": act_rates, "track": act_track}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=int, help="sequence number in config/bags.yaml")
    ap.add_argument("--file", type=Path, help="bag file path (overrides --bag; for tests)")
    ap.add_argument("--action", required=True, choices=ACTIONS)
    ap.add_argument("--limit", type=int, default=None,
                    help="chunks: rows printed (20); rates: chunks decompressed (10); track: max chunks (all)")
    ap.add_argument("--window", type=float, nargs=2, default=None, metavar=("W0", "W1"),
                    help="rates: time range used (default 300 360); track: highlighted range (default 300 420)")
    ap.add_argument("--topic", help="track: GNSS topic (default: gnss in config/topics.yaml)")
    a = ap.parse_args()

    path = a.file or (bag_path(a.bag) if a.bag else None)
    if path is None:
        ap.error("give --bag or --file")
    if a.window is None:
        a.window = [300.0, 360.0] if a.action == "rates" else [300.0, 420.0]
    a.limit_track = a.limit if a.action == "track" else None
    if a.limit is None:
        a.limit = 10 if a.action == "rates" else 20
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    try:
        with BagFile(path) as bag:
            print(f"[index read in {time.perf_counter() - t0:.2f} s]")
            ACTIONS[a.action](bag, a)
    except BagError as e:
        sys.exit(f"!! {path.name}: {e}")
    print(f"[{a.action} done in {time.perf_counter() - t0:.1f} s]")


if __name__ == "__main__":
    main()
