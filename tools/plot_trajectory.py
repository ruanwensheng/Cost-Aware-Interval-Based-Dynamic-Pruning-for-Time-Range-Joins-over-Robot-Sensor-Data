"""Plot the GNSS trajectory in metres (no areas), coloured by time.

    python tools/plot_trajectory.py [--out results/eda/trajectory.png]

Input: results/eda/gnss_track.csv (from `eda_bag.py --action track`).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roslab.config import RESULTS_DIR  # noqa: E402

INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "eda" / "trajectory.png")
    ap.add_argument("--arrow-every", type=float, default=25.0, help="metres of path between direction arrows")
    a = ap.parse_args()

    g = pd.read_csv(RESULTS_DIR / "eda" / "gnss_track.csv").sort_values("t")
    lat0, lon0 = g["lat"].mean(), g["lon"].mean()
    x = (g["lon"] - lon0) * np.cos(np.deg2rad(lat0)) * 111320      # metres east
    y = (g["lat"] - lat0) * 110540                                  # metres north

    fig, ax = plt.subplots(figsize=(14, 3.2), constrained_layout=True)
    ax.plot(x, y, color="#c8ccd1", lw=1, zorder=1)
    sc = ax.scatter(x, y, c=g["t"], cmap="Blues", vmin=-0.15 * g["t"].max(), vmax=g["t"].max(),
                    s=5, linewidths=0, zorder=2)
    # direction arrows, one every `a.arrow_every` metres of path
    xv, yv = x.to_numpy(), y.to_numpy()
    dist = np.r_[0, np.cumsum(np.hypot(np.diff(xv), np.diff(yv)))]
    for d in np.arange(a.arrow_every / 2, dist[-1], a.arrow_every):
        i = min(np.searchsorted(dist, d), len(xv) - 6)
        j = i + 5                                                   # 5 fixes = 1 s ahead gives the heading
        ax.annotate("", xy=(xv[j], yv[j]), xytext=(xv[i], yv[i]), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2, mutation_scale=12))
    ax.plot(x.iloc[0], y.iloc[0], "o", ms=8, mfc="white", mec=INK, zorder=3, label=f"start (t = {g['t'].iloc[0]:.0f} s)")
    ax.plot(x.iloc[-1], y.iloc[-1], "s", ms=8, mfc=INK, mec=INK, zorder=3, label=f"end (t = {g['t'].iloc[-1]:.0f} s)")
    cb = fig.colorbar(sc, ax=ax, pad=0.01, shrink=0.45, aspect=12)
    cb.set_label("t since bag start (s)", color=MUTED)
    ax.set_aspect("equal")
    ax.set_xlabel("x east (m)", color=MUTED)
    ax.set_ylabel("y north (m)", color=MUTED)
    ax.set_title(f"GNSS trajectory, {len(g)} fixes (lat0 {lat0:.6f}, lon0 {lon0:.6f})", color=INK, loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.grid(color=GRID, lw=0.5)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
