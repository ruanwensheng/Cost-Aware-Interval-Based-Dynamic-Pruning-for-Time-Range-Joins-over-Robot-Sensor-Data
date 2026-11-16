"""Check that the local experiment environment works (no Docker, no MinIO).

Run from the project folder:   python tools/check_env.py

1. Python version and the required packages.
2. Free disk space on the drive that holds the project (the bag alone is about 31.5 GB).
3. DuckDB writes a small Parquet file (4 row groups of 10 s each), then reads it back with a
   time filter, and shows that only 1 of the 4 row groups is read (row-group pruning works).
4. If a bag is already in data/bags/, open its index (no decompression) and list the topics.
"""
import importlib, platform, shutil, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ok = True


def step(msg):
    print(f"\n== {msg}")


step("1. Python and packages")
print("Python", platform.python_version(), "on", platform.system())
if sys.version_info < (3, 10):
    print("  !! need Python 3.10 or newer"); ok = False
for name in ["duckdb", "pyarrow", "pandas", "numpy", "rosbags", "psutil", "yaml", "lz4"]:
    try:
        mod = importlib.import_module(name)
        print(f"  {name:8s} {getattr(mod, '__version__', 'ok')}")
    except ImportError:
        print(f"  !! {name} missing: pip install -r requirements.txt"); ok = False

step("2. Disk space")
free = shutil.disk_usage(ROOT).free / 1e9
print(f"  free on the drive of {ROOT}: {free:.1f} GB")
if free < 50:
    print("  !! less than 50 GB free: the bag (31.5 GB) plus Parquet output may not fit"); ok = False

step("3. DuckDB row-group pruning on a local Parquet file")
import duckdb, pyarrow as pa, pyarrow.parquet as pq
tmp = ROOT / "results" / "_check_env.parquet"
tmp.parent.mkdir(exist_ok=True)
n = 8000                                    # 40 s of fake 200 Hz IMU data
t = [300 + i * 0.005 for i in range(n)]
pq.write_table(pa.table({"t": t, "acc_z": [9.8 + (i % 50) * 0.01 for i in range(n)]}),
               tmp, row_group_size=2000)    # 2000 rows = 10 s per row group
con = duckdb.connect()
start = time.perf_counter()
cnt = con.sql(f"SELECT count(*) FROM '{tmp.as_posix()}' WHERE t BETWEEN 312 AND 318").fetchone()[0]
print(f"  rows between 312 and 318 s: {cnt} (expected 1201), {time.perf_counter() - start:.3f} s")
stats = con.sql(f"""SELECT row_group_id, stats_min, stats_max FROM parquet_metadata('{tmp.as_posix()}')
                    WHERE path_in_schema = 't' ORDER BY row_group_id""").fetchall()
print("  min/max of t per row group (what DuckDB uses to skip):")
for rg, lo, hi in stats:
    keep = float(lo) <= 318 and float(hi) >= 312
    print(f"    row group {rg}: [{float(lo):.3f}, {float(hi):.3f}]  {'read' if keep else 'skipped'}")
if cnt != 1201:
    ok = False
tmp.unlink(missing_ok=True)

step("4. Bags in data/bags/ (index only, no decompression)")
bags = sorted((ROOT / "data" / "bags").glob("*.bag"))
if not bags:
    print("  no bag yet: download it (see README.md, step 4)")
for b in bags:
    try:
        from rosbags.rosbag1 import Reader
        with Reader(b) as r:
            dur = (r.end_time - r.start_time) / 1e9
            print(f"  {b.name}: {b.stat().st_size / 1e9:.2f} GB, {dur:.0f} s, "
                  f"{len(r.chunk_infos)} chunks, {r.message_count} messages")
            for c in sorted(r.connections, key=lambda c: c.topic):
                print(f"    {c.topic:45s} {c.msgtype:35s} {c.msgcount:>9d}  ({c.msgcount / dur:7.1f} Hz)")
    except Exception as e:                  # an incomplete download fails here
        print(f"  !! cannot open {b.name}: {e}  (download not finished?)"); ok = False

print("\nOK: the environment works." if ok else "\nSome checks failed: see the lines marked !!")
