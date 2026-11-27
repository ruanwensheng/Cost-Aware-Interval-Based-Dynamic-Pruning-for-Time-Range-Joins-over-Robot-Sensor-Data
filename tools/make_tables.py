"""Write results/tests/tables.md from the estimates and the measured runs.

    python tools/make_tables.py [--log results/tests/runs.jsonl]

Inputs (missing ones give "–"):
    results/tests/estimates_<layout>.csv          run_tests.py --estimate-only
    results/tests/estimates_<layout>_l2cond.csv   same with --l2-includes-condition true
    results/tests/estimates_<layout>_stats.csv    same with --point-a stats
    results/tests/runs.jsonl                      measured runs
Sections:
    Estimates     per layout and approach: point A (C_L2, C_p1), point B (C_L3, C_L4), chosen level,
                  estimate, inputs of the cost function, k before / after merging, g*
    l2_includes_condition   tests whose chosen level changes when C_L2 also counts the cond table
    point A stats            tests whose level changes with point A + GNSS statistics, total estimates
    Variant evaluation       per test and variant (plain, stats, stats_clip, l2cond): level chosen by the
                             estimate, measured median of that level (forced-level runs), regret =
                             measured(chosen) - fastest of L2/L3/L4, and the sum over tests
    Table 7 (bag, s) and Table 8 (pq, ms)   per test: level | "est_with_base / measured" for ssd and lan;
                  measured = median end_to_end of the auto runs. Marks: * warm or unknown cache,
                  † evict (evict file, not a real cache drop), none = cold
    Check 1       median measured time of L2, L3, L4 (mode all), level chosen by auto, fastest level
    Check 2       every run of one test (all modes, layouts) has the same result_hash
Bag times in seconds, Parquet times in milliseconds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roslab.config import RESULTS_DIR  # noqa: E402
from roslab.queries import load_tests  # noqa: E402

OUT = RESULTS_DIR / "tests"
LAYOUTS = ("ssd", "lan")
UNIT = {"bag": (1.0, "s"), "pq": (1e3, "ms")}
DASH = "–"


def fmt(v, scale=1.0, nd=3) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return DASH
    v = float(v) * scale
    return f"{v:.{nd}g}" if abs(v) < 1e4 else f"{v:,.0f}"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def read_est(layout: str, suffix: str = "") -> pd.DataFrame | None:
    p = OUT / f"estimates_{layout}{suffix}.csv"
    return pd.read_csv(p, dtype={"test": str}) if p.exists() else None


def estimates_section(est: pd.DataFrame, layout: str) -> list[str]:
    out = []
    for ap in ("pq", "bag"):
        e = est[est["approach"] == ap]
        if e.empty:
            continue
        sc, u = UNIT[ap]
        rows = []
        for r in e.itertuples():
            base = {"test": r.test, "point A": r.point_a,
                    f"C_L2 ({u})": fmt(r.C_L2, sc), f"C_p1 ({u})": fmt(r.C_p1, sc),
                    f"C_L3 ({u})": fmt(r.C_L3, sc), f"C_L4 ({u})": fmt(r.C_L4, sc),
                    "level": r.level_auto, f"estimate ({u})": fmt(r.est_auto, sc),
                    f"with base ({u})": fmt(getattr(r, "est_with_base", None), sc)}
            if ap == "bag":
                inp = {"r_c (/s)": fmt(r.r_c), "r_t (/s)": fmt(r.r_t), "r_ct (/s)": fmt(r.r_ct),
                       "b_c (KB)": fmt(r.b_c / 1e3), "b_t (KB)": fmt(r.b_t / 1e3)}
            else:
                inp = {"ρ_t (KB/s)": fmt(r.rho_t / 1e3), "ρ_c (KB/s)": fmt(r.rho_c / 1e3),
                       "n_t": int(r.n_t), "n_c": int(r.n_c)}
            tail = {"k raw → merged": f"{fmt(r.k_raw, nd=4)} → {fmt(r.k, nd=4)}",
                    "span (s)": fmt(r.span, nd=4), "Σw (s)": fmt(r.sum_w, nd=4),
                    "g* (s)": fmt(r.g_star, nd=4), "T_ub (s)": fmt(r.T_ub, nd=4)}
            rows.append({**base, **inp, **tail})
        src = e["consts_source"].iloc[0] if "consts_source" in e else ""
        out += [f"### {layout} / {ap}  (constants: {src}; l2_includes_condition = "
                f"{bool(e['l2_includes_condition'].iloc[0])})", "", md(pd.DataFrame(rows)), ""]
    return out


def l2cond_section(base: pd.DataFrame, alt: pd.DataFrame, layout: str) -> list[str]:
    m = base.merge(alt, on=["test", "approach"], suffixes=("_f", "_t"))
    rows = [{"test": r.test, "approach": r.approach, "point A (false)": r.point_a_f, "point A (true)": r.point_a_t,
             "level (false)": r.level_auto_f, "level (true)": r.level_auto_t,
             "C_L2 false → true": f"{fmt(r.C_L2_f, UNIT[r.approach][0])} → {fmt(r.C_L2_t, UNIT[r.approach][0])} "
                                  f"{UNIT[r.approach][1]}",
             "changed": "**yes**" if r.level_auto_f != r.level_auto_t else "no"} for r in m.itertuples()]
    changed = [f"{r['test']} ({r['approach']})" for r in rows if r["changed"] != "no"]
    return [f"## l2_includes_condition: false vs true ({layout})", "",
            "Level changes: " + (", ".join(changed) if changed else "none"), "", md(pd.DataFrame(rows)), ""]


def stats_section(base: pd.DataFrame, alt: pd.DataFrame, layout: str) -> list[str]:
    m = base.merge(alt, on=["test", "approach"], suffixes=("_p", "_s"))
    rows, tot = [], {}
    for r in m.itertuples():
        sc, u = UNIT[r.approach]
        tot.setdefault(r.approach, [0.0, 0.0])
        tot[r.approach][0] += r.est_with_base_p
        tot[r.approach][1] += r.est_with_base_s
        rows.append({"test": r.test, "approach": r.approach, "point A (plain → stats)": f"{r.point_a_p} → {r.point_a_s}",
                     "T_ub (s)": fmt(r.T_ub_s, nd=4), "level (plain → stats)": f"{r.level_auto_p} → {r.level_auto_s}",
                     "est_with_base plain → stats": f"{fmt(r.est_with_base_p, sc)} → {fmt(r.est_with_base_s, sc)} {u}",
                     "changed": "**yes**" if r.level_auto_p != r.level_auto_s else "no"})
    changed = [f"{r['test']} ({r['approach']})" for r in rows if r["changed"] != "no"]
    totals = [f"total {ap}: {fmt(v[0], UNIT[ap][0])} → {fmt(v[1], UNIT[ap][0])} {UNIT[ap][1]}" for ap, v in tot.items()]
    return [f"## point A: plain vs GNSS statistics ({layout}, every test with cond = gnss)", "",
            "Level changes: " + (", ".join(changed) if changed else "none"), "", "; ".join(totals), "",
            md(pd.DataFrame(rows)), ""]


VARIANTS = (("plain", ""), ("stats", "_stats"), ("stats_clip", "_stats_clip"), ("l2cond", "_l2cond"))


def variants_section(runs: pd.DataFrame | None, layout: str, labels: dict) -> list[str]:
    """Chosen level of each variant (from its estimates file) against the measured forced-level runs."""
    ests = {v: e for v, suf in VARIANTS if (e := read_est(layout, suf)) is not None}
    if runs is None or runs.empty or not ests:
        return []
    out = [f"## Variant evaluation ({layout})", "",
           "level = level chosen by the cost function of the variant; measured = median end_to_end of the "
           "forced runs of that level (mode L2 / L3 / L4); regret = measured(level) - fastest of L2, L3, L4.", ""]
    for ap in ("pq", "bag"):
        r = runs[(runs["layout"] == layout) & (runs["approach"] == ap) & runs["mode"].isin(("L2", "L3", "L4"))]
        if r.empty:
            continue
        sc, u = UNIT[ap]
        med = r.groupby(["test", "mode"])["end_to_end"].median()
        rows, tot = [], {v: [0.0, 0] for v in ests}
        for tid in labels:
            if tid not in med.index.get_level_values(0):
                continue
            m = med.loc[tid].to_dict()
            if len(m) < 3:
                continue
            fast = min(m, key=m.get)
            row = {"test": labels[tid], "fastest": f"{fast} {fmt(m[fast], sc)}"}
            for v, e in ests.items():
                er = e[(e["test"] == tid) & (e["approach"] == ap)]
                if er.empty:
                    row[v] = DASH
                    continue
                lv = er["level_auto"].iloc[0]
                reg = m[lv] - m[fast]
                tot[v][0] += reg
                tot[v][1] += lv != fast
                row[v] = f"{lv}: {fmt(m[lv], sc)} (+{fmt(reg, sc)})"
            rows.append(row)
        rows.append({"test": "**sum of regret** (tests not fastest)", "fastest": "",
                     **{v: f"{fmt(t[0], sc)} {u} ({t[1]})" for v, t in tot.items()}})
        out += [f"### {ap} ({u}): level: measured (+regret)", "", md(pd.DataFrame(rows)), ""]
    return out


def mark(caches) -> str:
    s = set(caches)
    if s & {"warm", "unknown"}:
        return "*"
    if "evict" in s:
        return "†"
    return ""


def main_tables(runs: pd.DataFrame | None, ests: dict, ids: list[str], labels: dict) -> list[str]:
    out = []
    for ap, title in (("bag", "Table 7 (bag, s)"), ("pq", "Table 8 (pq, ms)")):
        sc, _ = UNIT[ap]
        rows = []
        for tid in ids:
            row = {"test": labels[tid]}
            for lay in LAYOUTS:
                e = ests.get(lay)
                er = e[(e["test"] == tid) & (e["approach"] == ap)] if e is not None else pd.DataFrame()
                lvl = er["level_auto"].iloc[0] if len(er) else DASH
                col = "est_with_base" if "est_with_base" in er else "est_auto"
                est = fmt(er[col].iloc[0], sc) if len(er) else DASH
                meas = DASH
                if runs is not None:
                    rr = runs[(runs["layout"] == lay) & (runs["approach"] == ap) & (runs["test"] == tid)
                              & (runs["mode"] == "auto")]
                    if len(rr):
                        meas = fmt(rr["end_to_end"].median(), sc) + mark(rr["cache"])
                        lvl = "/".join(sorted(set(rr["level"])))
                row[f"{lay.upper()}: level"] = lvl
                row[f"{lay.upper()}: time (est / measured)"] = f"{est} / {meas}"
            rows.append(row)
        out += [f"## {title}", "", md(pd.DataFrame(rows)), ""]
    out += ["Estimate = est_with_base = cost function of the auto level + what it leaves out: "
            "c_q0 + c_req of the main query (pq), the measured index read t_open (bag).",
            "Marks: * warm or unknown cache, † evict file (not a real cache drop), none = cold.", ""]
    return out


def checks(runs: pd.DataFrame | None, labels: dict) -> list[str]:
    if runs is None or runs.empty:
        return ["## Check 1", "", "no measured runs yet", "", "## Check 2", "", "no measured runs yet", ""]
    rows = []
    for (lay, ap, tid), g in runs.groupby(["layout", "approach", "test"], sort=False):
        sc, u = UNIT[ap]
        med = {m: g.loc[g["mode"] == m, "end_to_end"].median() for m in ("L2", "L3", "L4")}
        have = {m: v for m, v in med.items() if pd.notna(v)}
        auto = g.loc[g["mode"] == "auto", "level"]
        rows.append({"layout": lay, "approach": ap, "test": labels.get(tid, tid),
                     **{f"{m} ({u})": fmt(v, sc) + mark(g.loc[g['mode'] == m, 'cache']) for m, v in med.items()},
                     "auto picks": "/".join(sorted(set(auto))) if len(auto) else DASH,
                     "fastest": min(have, key=have.get) if len(have) == 3 else DASH})
    c2 = []
    for tid, g in runs.groupby("test", sort=False):
        hs = sorted(set(g["result_hash"]))
        c2.append({"test": labels.get(tid, tid), "runs": len(g), "hashes": ", ".join(hs),
                   "ok": "yes" if len(hs) == 1 else "**NO**"})
    return ["## Check 1", "", md(pd.DataFrame(rows)), "", "## Check 2", "", md(pd.DataFrame(c2)), ""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=OUT / "runs.jsonl")
    ap.add_argument("--out", type=Path, default=OUT / "tables.md")
    a = ap.parse_args()

    specs = load_tests()
    ids, labels = [s.id for s in specs], {s.id: s.label for s in specs}
    ests = {lay: e for lay in LAYOUTS if (e := read_est(lay)) is not None}
    runs = pd.read_json(a.log, lines=True, dtype={"test": str}) if a.log.exists() else None

    lines = ["# Tests: estimates and measurements", "",
             "Written by tools/make_tables.py. Bag times in s, Parquet times in ms.", ""]
    lines += main_tables(runs, ests, ids, labels)
    lines += checks(runs, labels)
    for lay in LAYOUTS:
        lines += variants_section(runs, lay, labels)
    for lay, e in ests.items():
        lines += [f"## Estimates ({lay})", ""] + estimates_section(e, lay)
        alt = read_est(lay, "_l2cond")
        if alt is not None:
            lines += l2cond_section(e, alt, lay)
        st = read_est(lay, "_stats")
        if st is not None:
            lines += stats_section(e, st, lay)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
