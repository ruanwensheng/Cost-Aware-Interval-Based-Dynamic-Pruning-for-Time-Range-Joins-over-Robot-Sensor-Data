"""Derived EDA tables and results/eda/summary.md, computed from the EDA CSV files only (no bag access).

    python tools/eda_summary.py

Inputs (results/eda/): topics.csv, chunks.csv, rates.csv (from `rates --window 0 797 --limit 0`),
gnss_track.csv, gnss_outages.csv; config/topics.yaml.
Outputs (results/eda/): topic_chunks.csv, chunk_stats.csv, summary.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roslab.config import RESULTS_DIR, topics  # noqa: E402

EDA = RESULTS_DIR / "eda"
N_FIXED_COLS = 9      # chunks.csv columns before the per-topic message counts


def topic_chunks(ch: pd.DataFrame) -> pd.DataFrame:
    """Per topic: in how many chunks it appears, and the bytes of those chunks."""
    cols = ch.columns[N_FIXED_COLS:]
    has = ch[cols] > 0
    total_b = ch["bytes_compressed"].sum()
    return pd.DataFrame({
        "chunks": has.sum(), "pct_chunks": 100 * has.mean(),
        "pct_bytes": [100 * ch.loc[has[c], "bytes_compressed"].sum() / total_b for c in cols],
        "messages": ch[cols].sum(), "max_msgs_per_chunk": ch[cols].max()}).rename_axis("topic")


def chunk_stats(ch: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({"bytes_compressed": ch["bytes_compressed"], "bytes_uncompressed": ch["bytes_uncompressed"],
                      "ratio": ch["bytes_uncompressed"] / ch["bytes_compressed"],
                      "dur_ms": (ch["end_s"] - ch["start_s"]) * 1e3,
                      "n_messages": ch["n_messages"], "n_topics": ch["n_topics"]})
    return d.describe(percentiles=[0.05, 0.5, 0.95]).T


def track_xy(g: pd.DataFrame) -> pd.DataFrame:
    lat0, lon0 = g["lat"].mean(), g["lon"].mean()
    return g.assign(x=(g["lon"] - lon0) * np.cos(np.deg2rad(lat0)) * 111320, y=(g["lat"] - lat0) * 110540)


def passes(g: pd.DataFrame, min_travel_m: float = 100) -> pd.DataFrame:
    """Straight runs along x: segments of constant direction of the smoothed x, longer than min_travel_m."""
    xs = g["x"].rolling(25, center=True, min_periods=1).mean().to_numpy()     # 25 fixes = 5 s
    d = np.sign(np.diff(xs))
    cut = np.flatnonzero(np.diff(d)) + 1
    rows = []
    for s, e in zip(np.r_[0, cut], np.r_[cut, len(d)]):
        if abs(xs[e] - xs[s]) >= min_travel_m:
            seg = g.iloc[s:e + 1]
            rows.append({"t_start": seg["t"].iloc[0], "t_end": seg["t"].iloc[-1],
                         "direction": "east" if xs[e] > xs[s] else "west", "y_mean": seg["y"].mean()})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = topics()
    tp = pd.read_csv(EDA / "topics.csv")
    ch = pd.read_csv(EDA / "chunks.csv")
    rates = pd.read_csv(EDA / "rates.csv").set_index("topic")
    g = track_xy(pd.read_csv(EDA / "gnss_track.csv"))
    outages = pd.read_csv(EDA / "gnss_outages.csv")

    tc = topic_chunks(ch)
    st = chunk_stats(ch)
    tc.to_csv(EDA / "topic_chunks.csv", float_format="%.4f")
    st.to_csv(EDA / "chunk_stats.csv", float_format="%.4f")

    imu, gnss = cfg["imu"], cfg["gnss"]
    duration = ch["end_s"].max()
    win = ch[(ch["end_s"] >= 300) & (ch["start_s"] <= 420)]
    both = (ch[imu] > 0) & (ch[cfg["image"]] > 0)
    gnss_alts = rates[rates["type"] == "sensor_msgs/NavSatFix"]["dt_max"].sort_values()
    step = np.hypot(np.diff(g["x"]), np.diff(g["y"]))
    speed = step / np.diff(g["t"])
    status = g["status"].value_counts(normalize=True).sort_index()
    ps = passes(g)
    KB, MB, GB = 1024, 1024 ** 2, 1e9

    md = [
        "# EDA bag 5: tóm tắt", "",
        "Sinh tự động bởi `tools/eda_summary.py` từ các CSV trong `results/eda/`. "
        "GB = 10⁹ byte; KB, MB theo cơ số 1024.", "",
        "## Tham số (`config/topics.yaml`)", "",
        "| Tham số | Giá trị | Số liệu |", "|---|---|---|",
        f"| IMU | `{imu}` | {rates.loc[imu, 'hz_bag']:.2f} Hz, chu kỳ trung vị {rates.loc[imu, 'dt_median'] * 1e3:.2f} ms, "
        f"khoảng trống lớn nhất {rates.loc[imu, 'dt_max'] * 1e3:.1f} ms |",
        f"| GNSS | `{gnss}` | {rates.loc[gnss, 'hz_bag']:.2f} Hz; khoảng trống lớn nhất: "
        + ", ".join(f"`{t}` {v:.3f} s" for t, v in gnss_alts.items()) + " |",
        f"| tol | {cfg['tol']} s | chu kỳ GNSS trung vị {rates.loc[gnss, 'dt_median']:.3f} s |",
        f"| gap | {cfg['gap']} s | khoảng trống GNSS lớn nhất {rates.loc[gnss, 'dt_max']:.3f} s |", "",
        "## Bag và chunk (chỉ metadata)", "",
        "| Mục | Giá trị |", "|---|---|",
        f"| Thời lượng, số topic | {duration:.1f} s, {len(tp)} topic |",
        f"| Chunk | {len(ch):,}, nén {', '.join(ch['compression'].unique())}; "
        f"{ch['bytes_compressed'].sum() / GB:.1f} GB nén → {ch['bytes_uncompressed'].sum() / GB:.1f} GB sau giải nén |",
        f"| Thời gian mỗi chunk | trung bình {st.loc['dur_ms', 'mean']:.1f} ms, trung vị {st.loc['dur_ms', '50%']:.2f} ms, "
        f"max {st.loc['dur_ms', 'max']:.0f} ms; chunk chồng lên chunk trước: "
        f"{int((ch['start_s'].values[1:] < ch['end_s'].values[:-1]).sum())} |",
        f"| Byte mỗi chunk (trung vị) | {st.loc['bytes_compressed', '50%'] / KB:.0f} KB nén, "
        f"{st.loc['bytes_uncompressed', '50%'] / MB:.2f} MB sau giải nén |",
        f"| GNSS có mặt trong | {tc.loc[gnss, 'pct_chunks']:.1f}% số chunk, {tc.loc[gnss, 'pct_bytes']:.1f}% số byte "
        f"(tối đa {tc.loc[gnss, 'max_msgs_per_chunk']} điểm mỗi chunk) |",
        f"| IMU có mặt trong (= L1 với query cần IMU) | {tc.loc[imu, 'pct_chunks']:.1f}% số chunk, "
        f"{tc.loc[imu, 'pct_bytes']:.1f}% số byte |",
        f"| Chunk có cả IMU và ảnh | {100 * both.mean():.1f}% số chunk |",
        f"| Cửa sổ 300–420 s | {len(win):,} chunk, {win['bytes_compressed'].sum() / GB:.2f} GB "
        f"({100 * win['bytes_compressed'].sum() / ch['bytes_compressed'].sum():.1f}% số byte) |", "",
        "## Quỹ đạo GNSS", "",
        "| Mục | Giá trị |", "|---|---|",
        f"| Số điểm | {len(g):,}, t từ {g['t'].min():.1f} đến {g['t'].max():.1f} s |",
        f"| Chu kỳ | trung vị {np.median(np.diff(g['t'])):.3f} s, khoảng trống lớn nhất {np.diff(g['t']).max():.2f} s |",
        "| Status | " + ", ".join(f"`{s}`: {100 * p:.1f}%" for s, p in status.items()) + " |",
        f"| Đoạn mất tín hiệu | {len(outages)} (`gnss_outages.csv`) |",
        f"| Phạm vi | x từ {g['x'].min():.0f} đến {g['x'].max():.0f} m, y từ {g['y'].min():.0f} đến {g['y'].max():.0f} m |",
        f"| Tốc độ | trung vị {np.median(speed):.2f} m/s, p95 {np.percentile(speed, 95):.2f} m/s; "
        f"quãng đường {step.sum():.0f} m |", "",
        f"Các lượt chạy thẳng (đi được ≥ 100 m theo x): {len(ps)}", "",
        "| Lượt | t (s) | Hướng | y trung bình (m) |", "|---|---|---|---|",
        *[f"| {i + 1} | {r.t_start:.0f} → {r.t_end:.0f} | {r.direction} | {r.y_mean:+.1f} |"
          for i, r in enumerate(ps.itertuples())],
    ]
    (EDA / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("\nwrote results/eda/topic_chunks.csv, chunk_stats.csv, summary.md")


if __name__ == "__main__":
    main()
