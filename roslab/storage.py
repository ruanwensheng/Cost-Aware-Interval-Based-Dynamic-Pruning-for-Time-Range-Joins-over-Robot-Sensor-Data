"""Where the data lives: local disk (layout `ssd`) or MinIO over S3 (layout `lan`).

The code is the same for every layout; only config/layouts.yaml and .env differ.

    store = open_store("ssd")                  # or "lan"
    buf = store.read("bags/x.bag", 0, 4096)    # one request
    store.requests, store.bytes                # counters

Benchmark files read by layout_constants():
    results/bench/read_costs.csv          rows with const in {fixed_cost_per_read, read_speed} (cache=cold,
                                          else cache=evict)
                                          and bz2_decompress_speed  (tools/bench_read.py)
    results/bench/remote_<layout>_summary.csv   columns timestamp, c_req_s, bw_Bps; latest row
                                          (tools/bench_remote.py)
    results/bench/duckdb_<layout>.csv     columns const, value (s); const in {c_query, c_q0, c_int};
                                          latest row per const (tools/bench_duckdb.py)
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import yaml

from roslab.config import CONFIG_DIR, RESULTS_DIR, ROOT
from roslab.costfn import LayoutConsts

BENCH = RESULTS_DIR / "bench"


# ------------------------------------------------------------------ configuration
def load_dotenv(path: Path | None = None, override: bool = False) -> dict[str, str]:
    """Read KEY=VALUE lines of .env into os.environ (existing variables win unless override)."""
    path = Path(path) if path else ROOT / ".env"
    found: dict[str, str] = {}
    if not path.exists():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        found[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return found


def layout_cfg(name: str) -> dict:
    """Layout entry of config/layouts.yaml (or of the file in $ROSLAB_LAYOUTS)."""
    path = Path(os.environ.get("ROSLAB_LAYOUTS") or CONFIG_DIR / "layouts.yaml")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if name not in cfg:
        raise KeyError(f"layout {name!r} not in {path} (have {sorted(cfg)})")
    return {**cfg[name], "name": name}


def _s3_env(cfg: dict) -> dict[str, str]:
    load_dotenv()
    out = {}
    for field, var in cfg["env"].items():
        if not os.environ.get(var):
            raise RuntimeError(f"environment variable {var} is not set (layout {cfg['name']!r}, see .env.example)")
        out[field] = os.environ[var]
    return out


# ------------------------------------------------------------------ stores
class Store:
    """Byte-range reads by key. Counts requests and bytes."""

    def __init__(self) -> None:
        self.requests = 0
        self.bytes = 0

    def reset(self) -> None:
        self.requests = self.bytes = 0

    def size(self, key: str) -> int:
        raise NotImplementedError

    def read(self, key: str, offset: int, length: int | None = None) -> bytes:
        """`length` bytes from `offset` (None = to the end), in one request."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class LocalStore(Store):
    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = Path(root)
        self._files: dict[str, object] = {}

    def path(self, key: str) -> Path:
        return self.root.joinpath(*key.split("/"))

    def _file(self, key: str):
        if key not in self._files:
            self._files[key] = open(self.path(key), "rb", buffering=0)
        return self._files[key]

    def size(self, key: str) -> int:
        return self.path(key).stat().st_size

    def read(self, key: str, offset: int, length: int | None = None) -> bytes:
        f = self._file(key)
        if length is None:
            length = self.size(key) - offset
        buf = bytearray(length)
        mv = memoryview(buf)
        f.seek(offset)
        got = 0
        while got < length:
            n = f.readinto(mv[got:])
            if not n:
                break
            got += n
        self.requests += 1
        self.bytes += got
        return bytes(buf[:got])

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()


class S3Store(Store):
    """MinIO / S3. Each read is one GET with a Range header."""

    def __init__(self, endpoint: str, bucket: str, region: str, key_id: str, secret: str, prefix: str = "") -> None:
        super().__init__()
        import boto3
        from botocore.config import Config

        self.bucket, self.prefix = bucket, prefix
        self.client = boto3.client(
            "s3", endpoint_url=endpoint, region_name=region, aws_access_key_id=key_id,
            aws_secret_access_key=secret,
            config=Config(max_pool_connections=1, s3={"addressing_style": "path"}))

    def size(self, key: str) -> int:
        self.requests += 1
        return int(self.client.head_object(Bucket=self.bucket, Key=self.prefix + key)["ContentLength"])

    def read(self, key: str, offset: int, length: int | None = None) -> bytes:
        rng = f"bytes={offset}-" if length is None else f"bytes={offset}-{offset + length - 1}"
        body = self.client.get_object(Bucket=self.bucket, Key=self.prefix + key, Range=rng)["Body"].read()
        self.requests += 1
        self.bytes += len(body)
        return body


def open_store(layout: str) -> Store:
    cfg = layout_cfg(layout)
    if cfg["kind"] == "local":
        root = Path(cfg["root"])
        return LocalStore(root if root.is_absolute() else ROOT / root)
    if cfg["kind"] == "s3":
        e = _s3_env(cfg)
        return S3Store(e["endpoint"], e["bucket"], e["region"], e["key_id"], e["secret"], cfg.get("prefix", ""))
    raise ValueError(f"unknown layout kind {cfg['kind']!r}")


# ------------------------------------------------------------------ DuckDB
def duckdb_connect(layout: str, threads: int | None = None):
    """A new DuckDB connection that can read the layout's Parquet files."""
    import duckdb

    cfg = layout_cfg(layout)
    con = duckdb.connect()
    if threads:
        con.execute(f"SET threads = {int(threads)}")
    if cfg["kind"] == "s3":
        e = _s3_env(cfg)
        url = urlparse(e["endpoint"])
        use_ssl = "true" if url.scheme == "https" else "false"
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute(f"CREATE SECRET (TYPE s3, KEY_ID '{e['key_id']}', SECRET '{e['secret']}', "
                    f"REGION '{e['region']}', ENDPOINT '{url.netloc}', URL_STYLE 'path', USE_SSL {use_ssl})")
    return con


def parquet_uri(layout: str, key: str) -> str:
    cfg = layout_cfg(layout)
    if cfg["kind"] == "local":
        root = Path(cfg["root"])
        return (root if root.is_absolute() else ROOT / root).joinpath(*key.split("/")).as_posix()
    bucket = _s3_env(cfg)["bucket"]
    return f"s3://{bucket}/{cfg.get('prefix', '')}{key}"


# ------------------------------------------------------------------ constants and cache
def _need(path: Path, command: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path.relative_to(ROOT)} is missing: run  {command}")
    return path


def _latest(df, const: str):
    rows = df[df["const"] == const]
    return None if rows.empty else float(rows["value"].iloc[-1])


def layout_constants(layout: str, need_duckdb: bool = True) -> LayoutConsts:
    """Constants of the cost function for one layout (PROMPT_cost_function.md, section 3).

    need_duckdb=False allows c_q0 / c_int = None (bag approach does not use them).
    """
    import pandas as pd

    cfg = layout_cfg(layout)
    local = cfg["kind"] == "local"
    rc = pd.read_csv(_need(BENCH / "read_costs.csv", "python tools/bench_read.py --bag 5"))
    v_dec = float(rc.loc[rc["const"] == "bz2_decompress_speed", "value"].iloc[-1]) * 1e6
    if local:
        # not-warm rows of bench_read: "cold" (cache-drop command) if measured, else "evict" (evict file)
        states = set(rc["cache"])
        c = rc[rc["cache"] == ("cold" if "cold" in states else "evict")]
        c_req = float(c.loc[c["const"] == "fixed_cost_per_read", "value"].iloc[-1]) / 1e3
        bw = float(c.loc[c["const"] == "read_speed", "value"].iloc[-1]) * 1e6
        source = f"read_costs.csv ({c['cache'].iloc[0]})"
    else:
        rs = pd.read_csv(_need(BENCH / f"remote_{layout}_summary.csv",
                               f"python tools/bench_remote.py --layout {layout}")).sort_values("timestamp")
        c_req, bw = float(rs["c_req_s"].iloc[-1]), float(rs["bw_Bps"].iloc[-1])
        source = f"remote_{layout}_summary.csv"
    c_q0 = c_int = None
    dpath = BENCH / f"duckdb_{layout}.csv"
    if dpath.exists():
        d = pd.read_csv(dpath)
        c_q0, c_int = _latest(d, "c_q0"), _latest(d, "c_int")
        source += f" + {dpath.name}"
    elif need_duckdb:
        _need(dpath, f"python tools/bench_duckdb.py --layout {layout}")
    return LayoutConsts(layout, local, c_req, bw, v_dec, c_q0, c_int, source)


def evict_path(layout: str) -> Path | None:
    """The layout's evict file (config key evict_file), or None."""
    p = layout_cfg(layout).get("evict_file")
    if not p:
        return None
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def drop_cache(layout: str) -> str:
    """Empty the OS file cache before a measured run. Returns the cache state to log:
    "cold"     the layout's cache-drop command ran ($ROSLAB_DROP_CACHE_CMD for ssd);
    "evict"    no command, but the evict file exists: it was read whole (unbuffered, 8 MiB blocks), which
               pushes most other pages out of the cache (not guaranteed: see results/bench/evict_check.csv);
    "unknown"  neither."""
    load_dotenv()
    cfg = layout_cfg(layout)
    cmd = os.environ.get(cfg.get("cache_cmd_env", ""), "")
    if cmd:
        subprocess.run(cmd, shell=True, check=True)
        return "cold"
    ev = evict_path(layout)
    if ev is None or not ev.exists():
        return "unknown"
    buf = bytearray(EVICT_BLOCK)
    with open(ev, "rb", buffering=0) as f:
        while f.readinto(buf):
            pass
    return "evict"


EVICT_BLOCK = 8 << 20
OUTLIER_T_OPEN = 5.0          # s; a run whose t_open (open + index / footer) is longer is flagged, not dropped


def is_outlier(t_open: float | None) -> bool:
    return t_open is not None and t_open > OUTLIER_T_OPEN
