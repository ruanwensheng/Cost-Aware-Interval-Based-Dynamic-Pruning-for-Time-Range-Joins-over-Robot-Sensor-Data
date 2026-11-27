"""Create the evict file used by roslab.storage.drop_cache when no cache-drop command is set.

    python tools/make_evict_file.py [--layout ssd] [--factor 1.5] [--max-gb 30] [--yes]

Size = factor x RAM of this machine (psutil). Content = random bytes (not compressible), written in
64 MiB blocks. Path = evict_file of the layout in config/layouts.yaml (data/evict.bin for ssd).
Prints RAM and free disk space first. Stops without writing if the file would be larger than
--max-gb (unless --yes) or would leave less than 5 GB free.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roslab.storage import evict_path  # noqa: E402

BLOCK = 64 << 20
MARGIN = 5e9


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", default="ssd")
    ap.add_argument("--factor", type=float, default=1.5)
    ap.add_argument("--max-gb", type=float, default=30.0)
    ap.add_argument("--yes", action="store_true", help="allow a file larger than --max-gb")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    path = evict_path(a.layout)
    if path is None:
        sys.exit(f"layout {a.layout!r} has no evict_file in config/layouts.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    ram = psutil.virtual_memory().total
    size = int(a.factor * ram) // BLOCK * BLOCK + BLOCK
    free = shutil.disk_usage(path.parent).free + (path.stat().st_size if path.exists() else 0)
    print(f"RAM            {ram / 1e9:.2f} GB")
    print(f"free on {path.anchor:6s} {free / 1e9:.1f} GB")
    print(f"evict file     {path}  {size / 1e9:.2f} GB ({a.factor} x RAM)")
    if size > a.max_gb * 1e9 and not a.yes:
        sys.exit(f"stop: {size / 1e9:.1f} GB > --max-gb {a.max_gb}; rerun with --yes to create it")
    if free - size < MARGIN:
        sys.exit(f"stop: would leave {(free - size) / 1e9:.1f} GB free (< {MARGIN / 1e9:.0f} GB)")

    rng = np.random.default_rng(a.seed)
    tmp = path.with_suffix(".tmp")
    t0 = time.perf_counter()
    with open(tmp, "wb") as f:
        for i in range(size // BLOCK):
            f.write(rng.bytes(BLOCK))
            if i % 16 == 0:
                print(f"\r  {(i + 1) * BLOCK / 1e9:.1f} / {size / 1e9:.1f} GB", end="", flush=True)
    tmp.replace(path)
    print(f"\nwrote {path} ({path.stat().st_size / 1e9:.2f} GB) in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
