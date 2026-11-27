"""Engine tests on a small synthetic bag (tests/fakebag.py), both approaches.

Setup (once per module): write the fake bag, convert it with tools/convert.py (Parquet + gnss_stats),
and serve all files from moto.server.ThreadedMotoServer over real HTTP (so every read is a GET with a
Range header).

Bag engine, local store and S3 (moto). For each fake test (kinds 1, 4, 5, 6s) and each mode (L2, L3, L4, auto):
    - result_hash equals the brute-force answer, rows_out > 0, same hash local vs S3;
    - chunks_read(L4) <= chunks_read(L3) <= chunks_read(L2);
    - requests = 2 (index) + chunks_read (+1 for the gnss_stats table in test 6s);
    - phase 2 never re-reads a phase-1 chunk: requests count one GET per distinct chunk.
Parquet engine (DuckDB), layout ssd on the converted files, same tests and modes:
    - result_hash equals the bag engine hash (and the brute-force answer);
    - rg_expected(L4) <= rg_expected(L3) <= rg_expected(L2);
    - through S3 (DuckDB httpfs + moto): same hashes. Skipped with a message if httpfs cannot load.
Slow (~10 min): the bag is written once (~25 s); a run reads up to 1800 chunks, and through moto
each GET costs ~15-20 ms. Set ROSLAB_FAST_TESTS=1 to skip every S3 part (~1.5 min).
Run:  python -m unittest discover -s tests -v
"""
import logging
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import fakebag as F  # noqa: E402
from convert import convert  # noqa: E402
from roslab import bag_engine, pq_engine  # noqa: E402
from roslab import costfn as C  # noqa: E402
from roslab import intervals as I  # noqa: E402
from roslab.queries import Spec, result_hash  # noqa: E402
from roslab.storage import LocalStore, S3Store, open_store  # noqa: E402

MODES = ("L2", "L3", "L4", "auto")
BUCKET = "rosario"
FAST = os.environ.get("ROSLAB_FAST_TESTS") == "1"           # skip the S3 (moto) runs
K = C.LayoutConsts("test", True, c_req=1e-4, bw=5e8, v_dec=6e6, c_q0=4e-3, c_int=5e-3)


def spec(id, **kw) -> Spec:
    base = dict(id=id, label=id, cond="gnss", target="imu", window="whole", point_a="plain", bag=5,
                ell=10.0, tol=0.1, gap=0.3)
    return Spec(**{**base, **kw})


SPECS = [
    spec("f1", area="A", box=I.Box.from_dict(F.box_x(40, 50))),                       # strip visited 6 times
    spec("f4", area="A", box=I.Box.from_dict(F.box_x(40, 50)), window=[35.0, 75.0]),  # sub-window
    spec("f5", cond="imu", target="gnss", shock=True, g0=9.81, threshold=2.0),        # shocks on IMU
    spec("f6s", area="W", box=I.Box.from_dict(F.box_x(-10, 110, -20, 20)), point_a="stats"),
]
# f6: area W with the plain point A, run once (local, auto) to log whether point A starts phase 1
F6 = spec("f6", area="W", box=I.Box.from_dict(F.box_x(-10, 110, -20, 20)), point_a="plain")

S = {}          # shared fixtures, filled by setUpModule


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def setUpModule():
    key = bag_engine.bag_key(5)
    root, bag_path, data = F.cached_fake_bag(key)
    convert(bag_path, root, 10.0, bag_no=5, log=None)          # parquet/*.parquet + meta/gnss_stats_bag5.csv
    S.update(key=key, root=root, data=data, stores={"local": LocalStore(root)}, server=None, httpfs=None)

    tmp = tempfile.TemporaryDirectory()
    layouts = Path(tmp.name) / "layouts.yaml"
    layouts.write_text(
        f"ssd: {{kind: local, root: '{root.as_posix()}', cache_cmd_env: ROSLAB_TEST_NO_DROP}}\n"
        "lan: {kind: s3, prefix: '', cache_cmd_env: ROSLAB_TEST_NO_DROP,\n"
        "      env: {endpoint: S3_ENDPOINT, bucket: S3_BUCKET, region: S3_REGION,\n"
        "            key_id: AWS_ACCESS_KEY_ID, secret: AWS_SECRET_ACCESS_KEY}}\n", encoding="utf-8")
    env = {"ROSLAB_LAYOUTS": str(layouts)}
    if not FAST:
        from moto.server import ThreadedMotoServer

        logging.getLogger("werkzeug").setLevel(logging.ERROR)     # one log line per GET otherwise
        port = free_port()
        server = ThreadedMotoServer(ip_address="127.0.0.1", port=port, verbose=False)
        server.start()
        endpoint = f"http://127.0.0.1:{port}"
        s3 = S3Store(endpoint, BUCKET, "us-east-1", "test", "test")
        s3.client.create_bucket(Bucket=BUCKET)
        for f in sorted(p for p in root.rglob("*") if p.is_file()):
            s3.client.upload_file(str(f), BUCKET, f.relative_to(root).as_posix())
        S["stores"]["s3"] = s3
        S.update(server=server, endpoint=endpoint)
        env.update({"S3_ENDPOINT": endpoint, "S3_BUCKET": BUCKET, "S3_REGION": "us-east-1",
                    "AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test"})
    patch = mock.patch.dict(os.environ, env)
    patch.start()
    S.update(tmp=tmp, patch=patch)


def tearDownModule():
    S["stores"]["local"].close()
    if S["server"]:
        S["server"].stop()
    S["patch"].stop()
    S["tmp"].cleanup()


def table(runs: dict, extra: list[str]) -> str:
    lines = [f"{'test':5s} {'where':5s} {'mode':4s} {'point_a':7s} {'level':5s} {'hash':16s} {'rows':>5s} "
             + " ".join(f"{c:>8s}" for c in extra)]
    for (tid, where, mode), r in runs.items():
        lines.append(f"{tid:5s} {where:5s} {mode:4s} {r['point_a']:7s} {r['level']:5s} {r['result_hash']:16s} "
                     f"{r['rows_out']:5d} " + " ".join(f"{r[c]:8d}" for c in extra))
    return "\n".join(lines)


class TestBagEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.stores, cls.key = S["data"], S["stores"], S["key"]
        cls.runs = {}          # (spec id, store, mode) -> log row
        for sp in SPECS:
            for name, store in cls.stores.items():
                for mode in MODES:
                    cls.runs[sp.id, name, mode] = bag_engine.run(sp, store, K, mode, 5)
        cls.f6 = bag_engine.run(F6, cls.stores["local"], K, "auto", 5)
        S["bag_runs"] = cls.runs
        sys.stderr.write("\nbag engine\n" + table({**cls.runs, ("f6", "local", "auto"): cls.f6},
                                                   ["chunks_read", "chunks_phase1", "chunks_phase2", "requests"])
                         + f"\nf6 point A: C_p1 = {cls.f6['C_p1']:.4f} s vs C_L2 = {cls.f6['C_L2']:.4f} s -> "
                           f"{cls.f6['point_a']}\n")

    def test_hash_equals_brute_force(self):
        for sp in SPECS:
            w0, w1 = self.runs[sp.id, "local", "L2"]["w0"], self.runs[sp.id, "local", "L2"]["w1"]
            want = F.expected(sp, self.data, w0, w1)
            self.assertGreater(len(want), 0, sp.id)
            for name in self.stores:
                for mode in MODES:
                    r = self.runs[sp.id, name, mode]
                    with self.subTest(test=sp.id, store=name, mode=mode):
                        self.assertEqual(r["rows_out"], len(want))
                        self.assertEqual(r["result_hash"], result_hash(want["t_ns"]))

    @unittest.skipIf(FAST, "ROSLAB_FAST_TESTS=1")
    def test_local_equals_s3(self):
        for (tid, name, mode), r in self.runs.items():
            if name == "s3":
                loc = self.runs[tid, "local", mode]
                with self.subTest(test=tid, mode=mode):
                    for k in ("result_hash", "level", "chunks_read", "requests", "bytes_read", "bytes_decompressed"):
                        self.assertEqual(r[k], loc[k], k)

    def test_chunks_l4_le_l3_le_l2(self):
        for sp in SPECS:
            for name in self.stores:
                n = {m: self.runs[sp.id, name, m]["chunks_read"] for m in ("L2", "L3", "L4")}
                with self.subTest(test=sp.id, store=name, chunks=n):
                    self.assertLessEqual(n["L4"], n["L3"])
                    self.assertLessEqual(n["L3"], n["L2"])

    def test_requests_one_per_distinct_chunk(self):
        for (tid, name, mode), r in self.runs.items():
            stats = 1 if tid == "f6s" else 0
            with self.subTest(test=tid, store=name, mode=mode):
                # one GET per chunk read + 2 for the index (+ the gnss_stats table)
                self.assertEqual(r["requests"], 2 + r["chunks_read"] + stats)
                # phase 2 adds only chunks not read in phase 1 (chunks_read counts distinct chunks)
                self.assertEqual(r["chunks_phase1"] + r["chunks_phase2"], r["chunks_read"])

    def test_f6_plain_point_a(self):
        """Area W covers the whole path: the plain point A only compares C_p1 with C_L2 (logged above);
        whatever it picks, the answer must be right."""
        want = F.expected(F6, self.data, self.f6["w0"], self.f6["w1"])
        self.assertEqual(self.f6["result_hash"], result_hash(want["t_ns"]))
        self.assertIn(self.f6["point_a"], ("phase1", "L2"))
        self.assertEqual(self.f6["point_a"] == "phase1", self.f6["C_p1"] < self.f6["C_L2"])

    def test_pruning_really_happens(self):
        r = self.runs["f1", "local", "L4"]
        self.assertGreater(r["chunks_phase2"], 0)
        self.assertLess(r["chunks_read"], r["chunks_total"] / 2)
        self.assertEqual(r["k"], 6)                                    # 3 periods x 2 passes over the strip

    def test_log_t_open(self):
        r = self.runs["f1", "local", "L2"]
        self.assertGreater(r["t_open"], 0)
        self.assertIs(r["outlier"], r["t_open"] > 5.0)

    @unittest.skipIf(FAST, "ROSLAB_FAST_TESTS=1")
    def test_s3_range_get(self):
        s3, loc = self.stores["s3"], self.stores["local"]
        for off, n in ((0, 13), (12345, 1000), (loc.size(self.key) - 10, None)):
            self.assertEqual(s3.read(self.key, off, n), loc.read(self.key, off, n))
        self.assertEqual(s3.size(self.key), loc.size(self.key))

    @unittest.skipIf(FAST, "ROSLAB_FAST_TESTS=1")
    def test_open_store_lan(self):
        s = open_store("lan")
        self.assertIsInstance(s, S3Store)
        self.assertEqual(s.read(self.key, 0, 13), b"#ROSBAG V2.0\n")
        self.assertEqual((s.requests, s.bytes), (1, 13))


def httpfs_error() -> str | None:
    """None if DuckDB can load httpfs (downloads it the first time), else the error."""
    import duckdb

    try:
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.close()
        return None
    except Exception as e:                       # no internet, blocked download, ...
        return str(e).splitlines()[0]


class TestPqEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = S["data"]
        cls.layouts = ["ssd"]
        cls.s3_skip = "ROSLAB_FAST_TESTS=1" if FAST else None
        if not FAST:
            err = httpfs_error()
            if err:
                cls.s3_skip = f"DuckDB httpfs not available: {err}"
                sys.stderr.write(f"\n!! pq engine over S3 skipped: {cls.s3_skip}\n")
            else:
                cls.layouts.append("lan")
        cls.runs = {}          # (spec id, layout, mode) -> log row
        for sp in SPECS:
            for lay in cls.layouts:
                for mode in MODES:
                    cls.runs[sp.id, lay, mode] = pq_engine.run(sp, lay, K, mode)
        cls.f6 = pq_engine.run(F6, "ssd", K, "auto")
        sys.stderr.write("\npq engine\n" + table({**cls.runs, ("f6", "ssd", "auto"): cls.f6},
                                                  ["rg_total", "rg_expected", "n_intervals"])
                         + f"\nf6 point A: C_p1 = {cls.f6['C_p1'] * 1e3:.3f} ms vs C_L2 = {cls.f6['C_L2'] * 1e3:.3f} ms"
                           f" -> {cls.f6['point_a']}\n")

    def test_hash_equals_bag_engine(self):
        bag = S.get("bag_runs")
        for sp in SPECS:
            w0, w1 = self.runs[sp.id, "ssd", "L2"]["w0"], self.runs[sp.id, "ssd", "L2"]["w1"]
            want = result_hash(F.expected(sp, self.data, w0, w1)["t_ns"])
            for lay in self.layouts:
                for mode in MODES:
                    r = self.runs[sp.id, lay, mode]
                    with self.subTest(test=sp.id, layout=lay, mode=mode):
                        self.assertGreater(r["rows_out"], 0)
                        self.assertEqual(r["result_hash"], want)
                        if bag:
                            self.assertEqual(r["result_hash"], bag[sp.id, "local", mode]["result_hash"])
        self.assertEqual(self.f6["result_hash"], result_hash(F.expected(F6, self.data, self.f6["w0"],
                                                                        self.f6["w1"])["t_ns"]))

    def test_same_window_as_bag_engine(self):
        bag = S.get("bag_runs")
        if not bag:
            self.skipTest("bag engine runs not available")
        for sp in SPECS:
            r, b = self.runs[sp.id, "ssd", "L2"], bag[sp.id, "local", "L2"]
            with self.subTest(test=sp.id):
                self.assertEqual(r["w0"], b["w0"])
                self.assertLessEqual(r["w1"], b["w1"])        # pq: max t in the footers; bag: end of last chunk

    def test_row_groups_l4_le_l3_le_l2(self):
        for sp in SPECS:
            n = {m: self.runs[sp.id, "ssd", m]["rg_expected"] for m in ("L2", "L3", "L4")}
            with self.subTest(test=sp.id, rg=n):
                self.assertLessEqual(n["L4"], n["L3"])
                self.assertLessEqual(n["L3"], n["L2"])

    def test_pruning_really_happens(self):
        r = self.runs["f4", "ssd", "L3"]
        self.assertLess(r["rg_expected"], self.runs["f4", "ssd", "L2"]["rg_expected"])

    def test_l4_without_merging(self):
        """With K, g* ~ 100 s merges every visit of f1 into one interval (logged above). With c_int ~ 0,
        g* ~ 0: L4 runs one UNION ALL branch per visit (6) and must give the same answer."""
        k0 = C.LayoutConsts("test0", True, c_req=K.c_req, bw=K.bw, v_dec=K.v_dec, c_q0=K.c_q0, c_int=1e-12)
        r = pq_engine.run(SPECS[0], "ssd", k0, "L4")
        self.assertEqual(self.runs["f1", "ssd", "L4"]["n_intervals"], 1)
        self.assertEqual(r["n_intervals"], 6)
        self.assertEqual(r["result_hash"], self.runs["f1", "ssd", "L2"]["result_hash"])

    def test_log_t_open(self):
        r = self.runs["f1", "ssd", "L2"]
        self.assertGreater(r["t_open"], 0)
        self.assertIs(r["outlier"], r["t_open"] > 5.0)

    def test_s3_equals_ssd(self):
        if self.s3_skip:
            self.skipTest(self.s3_skip)
        for sp in SPECS:
            for mode in MODES:
                with self.subTest(test=sp.id, mode=mode):
                    a, b = self.runs[sp.id, "lan", mode], self.runs[sp.id, "ssd", mode]
                    for k in ("result_hash", "level", "rg_expected", "n_intervals"):
                        self.assertEqual(a[k], b[k], k)


if __name__ == "__main__":
    unittest.main()
