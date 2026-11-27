"""roslab.storage: .env, layouts, LocalStore, constants, cache.   Run:  python -m unittest discover -s tests -v"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roslab import storage  # noqa: E402


class TestStorage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        layouts = self.dir / "layouts.yaml"
        layouts.write_text(
            f"ssd: {{kind: local, root: '{self.dir.as_posix()}', cache_cmd_env: ROSLAB_TEST_DROP}}\n"
            "lan: {kind: s3, prefix: '', cache_cmd_env: ROSLAB_TEST_DROP,\n"
            "      env: {endpoint: T_EP, bucket: T_BK, region: T_RG, key_id: T_ID, secret: T_SC}}\n",
            encoding="utf-8")
        self.env = mock.patch.dict(os.environ, {"ROSLAB_LAYOUTS": str(layouts)})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_dotenv(self):
        p = self.dir / ".env"
        p.write_text("# comment\nT_A=1\nexport T_B = 'two words'\nT_C=\"x=y\"\n\nbad line\n", encoding="utf-8")
        with mock.patch.dict(os.environ, {"T_A": "keep"}):
            got = storage.load_dotenv(p)
            self.assertEqual(got, {"T_A": "1", "T_B": "two words", "T_C": "x=y"})
            self.assertEqual((os.environ["T_A"], os.environ["T_B"]), ("keep", "two words"))   # no override

    def test_layout_cfg(self):
        self.assertEqual(storage.layout_cfg("ssd")["kind"], "local")
        with self.assertRaises(KeyError):
            storage.layout_cfg("cloud")

    def test_local_store(self):
        (self.dir / "sub").mkdir()
        (self.dir / "sub" / "f.bin").write_bytes(bytes(range(100)))
        s = storage.open_store("ssd")
        try:
            self.assertEqual(s.size("sub/f.bin"), 100)
            self.assertEqual(s.read("sub/f.bin", 10, 5), bytes([10, 11, 12, 13, 14]))
            self.assertEqual(s.read("sub/f.bin", 95), bytes(range(95, 100)))           # to the end
            self.assertEqual((s.requests, s.bytes), (2, 10))
            s.reset()
            self.assertEqual((s.requests, s.bytes), (0, 0))
        finally:
            s.close()

    def test_parquet_uri_local(self):
        self.assertEqual(storage.parquet_uri("ssd", "parquet/imu.parquet"),
                         (self.dir / "parquet" / "imu.parquet").as_posix())

    def test_s3_env_missing(self):
        with self.assertRaises(RuntimeError) as e:
            storage.parquet_uri("lan", "parquet/imu.parquet")
        self.assertIn("is not set", str(e.exception))

    def test_drop_cache(self):
        with mock.patch.dict(os.environ, {"ROSLAB_TEST_DROP": ""}):
            self.assertEqual(storage.drop_cache("ssd"), "unknown")
        with mock.patch.dict(os.environ, {"ROSLAB_TEST_DROP": f'"{sys.executable}" -c "pass"'}):
            self.assertEqual(storage.drop_cache("ssd"), "cold")

    def test_drop_cache_evict(self):
        ev = self.dir / "evict.bin"
        layouts = self.dir / "layouts2.yaml"
        layouts.write_text(f"ssd: {{kind: local, root: '{self.dir.as_posix()}', cache_cmd_env: ROSLAB_TEST_DROP, "
                           f"evict_file: '{ev.as_posix()}'}}\n", encoding="utf-8")
        with mock.patch.dict(os.environ, {"ROSLAB_LAYOUTS": str(layouts), "ROSLAB_TEST_DROP": ""}):
            self.assertEqual(storage.drop_cache("ssd"), "unknown")             # file missing
            ev.write_bytes(os.urandom(3 * storage.EVICT_BLOCK // 2))
            self.assertEqual(storage.drop_cache("ssd"), "evict")
            with mock.patch.dict(os.environ, {"ROSLAB_TEST_DROP": f'"{sys.executable}" -c "pass"'}):
                self.assertEqual(storage.drop_cache("ssd"), "cold")            # the command wins

    def test_outlier(self):
        self.assertEqual([storage.is_outlier(v) for v in (None, 0.4, 5.0, 5.01)], [False, False, False, True])

    def test_layout_constants(self):
        bench = self.dir / "bench"
        bench.mkdir()
        (bench / "read_costs.csv").write_text(
            "case,cache,const,value,unit\n,cold,fixed_cost_per_read,0.2,ms\n,cold,read_speed,500,MB/s\n"
            ",warm,fixed_cost_per_read,0.005,ms\n,any,bz2_decompress_speed,6,MB/s\n", encoding="utf-8")
        with mock.patch.object(storage, "BENCH", bench), mock.patch.object(storage, "ROOT", self.dir):
            k = storage.layout_constants("ssd", need_duckdb=False)
            self.assertEqual((k.local, k.c_q0, k.c_int), (True, None, None))
            self.assertAlmostEqual(k.c_req, 0.0002)
            self.assertAlmostEqual(k.bw, 5e8)
            self.assertAlmostEqual(k.v_dec, 6e6)
            with self.assertRaises(FileNotFoundError) as e:
                storage.layout_constants("ssd")
            self.assertIn("bench_duckdb.py --layout ssd", str(e.exception))
            (bench / "duckdb_ssd.csv").write_text("const,value\nc_q0,0.5\nc_int,0.1\nc_q0,0.004\nc_int,0.003\n",
                                                  encoding="utf-8")
            k = storage.layout_constants("ssd")
            self.assertEqual((k.c_q0, k.c_int), (0.004, 0.003))                      # latest rows
            with self.assertRaises(FileNotFoundError) as e:
                storage.layout_constants("lan")
            self.assertIn("bench_remote.py --layout lan", str(e.exception))


if __name__ == "__main__":
    unittest.main()
