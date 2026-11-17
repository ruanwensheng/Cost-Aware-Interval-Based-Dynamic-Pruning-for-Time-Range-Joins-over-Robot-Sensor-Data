"""Re-create every file in results/ (EDA, read benchmark, M2) from the bag, in order.

    python tools/reproduce.py                 # all steps
    python tools/reproduce.py --list          # show the steps
    python tools/reproduce.py --only m2       # some steps (names from --list)
    python tools/reproduce.py --from track    # a step and everything after it

The console output of each step is also saved to the log file shown in --list.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roslab.config import RESULTS_DIR, bag_path  # noqa: E402

BAG = 5
EDA, BENCH, M2 = RESULTS_DIR / "eda", RESULTS_DIR / "bench", RESULTS_DIR / "m2"
BAG_FILE = str(bag_path(BAG))

# (name, arguments after `python`, log file or None)          approx. time on bag 5
STEPS = [
    ("info",        ["eda_bag.py", "--bag", str(BAG), "--action", "info"], EDA / "info.txt"),           # 1 s
    ("layout_head", ["tools/inspect_bag.py", BAG_FILE, "--limit", "15"], EDA / "layout_head.txt"),     # 1 s
    ("layout_index", ["tools/inspect_bag.py", BAG_FILE, "--index", "--limit", "42"],                   # 1 s
     EDA / "layout_index.txt"),                         # 42 = bag header + 38 connections + 3 chunk infos
    ("chunks",      ["eda_bag.py", "--bag", str(BAG), "--action", "chunks", "--limit", "5"],           # ~10 min
     EDA / "chunks.txt"),
    ("topic_mix",   ["eda_bag.py", "--bag", str(BAG), "--action", "topic-mix"], EDA / "topic_mix.txt"),  # 1 s
    ("rates",       ["eda_bag.py", "--bag", str(BAG), "--action", "rates", "--window", "0", "797",     # 0.5-10 min
                     "--limit", "0"], EDA / "rates.txt"),
    ("track",       ["eda_bag.py", "--bag", str(BAG), "--action", "track"], EDA / "track.log"),         # ~12 min
    ("eda_summary", ["tools/eda_summary.py"], None),                                                     # 1 s
    ("bench",       ["tools/bench_read.py", "--bag", str(BAG)], BENCH / "bench_run.log"),                # ~5 min
    ("tests",       ["-m", "unittest", "discover", "-s", "tests"], None),                                # 1 s
    ("m2",          ["tools/m2_feasibility.py", "--write-areas"], M2 / "m2_run.log"),                    # 10 s
]


def run(name: str, args: list[str], log: Path | None) -> None:
    print(f"\n===== {name}: python {' '.join(args)}", flush=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen([sys.executable, *args], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        lines.append(line)
    if proc.wait() != 0:
        sys.exit(f"!! step {name} failed (exit {proc.returncode})")
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="+", metavar="STEP")
    ap.add_argument("--from", dest="start", metavar="STEP")
    a = ap.parse_args()
    names = [s[0] for s in STEPS]
    if a.list:
        for name, args, log in STEPS:
            print(f"{name:13s} python {' '.join(args)}" + (f"   > {log.relative_to(ROOT)}" if log else ""))
        return
    for s in (a.only or []) + ([a.start] if a.start else []):
        if s not in names:
            sys.exit(f"unknown step {s!r}; steps: {names}")
    todo = STEPS
    if a.only:
        todo = [s for s in STEPS if s[0] in a.only]
    elif a.start:
        todo = STEPS[names.index(a.start):]
    for step in todo:
        run(*step)


if __name__ == "__main__":
    main()
