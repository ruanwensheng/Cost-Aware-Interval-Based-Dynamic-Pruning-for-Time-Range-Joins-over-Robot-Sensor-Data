"""tools/convert.py on the fake bag.   Run:  python -m unittest discover -s tests -v

- column t of imu.parquet / gnss.parquet is bit-identical to the t the bag engine computes;
- row group j holds only rows with j*ell <= t < (j+1)*ell, in both files (common grid);
- every value is the one written in the bag; gnss_stats_bag5.csv matches the GNSS buckets;
- --limit-s keeps only t < limit.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import fakebag as F  # noqa: E402
from convert import bucket_of, convert  # noqa: E402
from roslab import bag_engine  # noqa: E402
from roslab.config import topics  # noqa: E402
from roslab.storage import LocalStore  # noqa: E402

ELL = 10.0


class TestConvert(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = bag_engine.bag_key(5)
        cls.root, cls.bag, cls.data = F.cached_fake_bag(cls.key)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "full"
        cls.row = convert(cls.bag, cls.out, ELL, log=None)
        cls.out30 = Path(cls.tmp.name) / "limit30"
        cls.row30 = convert(cls.bag, cls.out30, ELL, limit_s=30.0, log=None)
        cls.pq = {r: pq.read_table(cls.out / "parquet" / f"{r}.parquet").to_pandas() for r in ("imu", "gnss")}

        store = LocalStore(cls.root)                 # what the bag engine decodes from the same bag
        try:
            bag = bag_engine.read_index(store, cls.key)
            tp = topics()
            conns = {k: "imu" for k in bag.conns_for_topic(tp["imu"])}
            conns.update({k: "gnss" for k in bag.conns_for_topic(tp["gnss"])})
            rd = bag_engine.Reader(store, cls.key, bag, conns)
            rd.read(np.ones(len(bag.chunks), bool))
            cls.engine = {r: rd.frame(r) for r in ("imu", "gnss")}
        finally:
            store.close()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_t_bit_identical_to_bag_engine(self):
        for r in ("imu", "gnss"):
            a, b = self.pq[r]["t"].to_numpy(), self.engine[r]["t"].to_numpy()
            self.assertEqual(a.dtype, np.float64)
            self.assertEqual(len(a), len(b))
            self.assertTrue(np.array_equal(a.view(np.int64), b.view(np.int64)), r)     # same bits
            self.assertTrue(np.array_equal(self.pq[r]["t_ns"].to_numpy(), self.engine[r]["t_ns"].to_numpy()))

    def test_row_groups_follow_the_grid(self):
        js = {}
        for r in ("imu", "gnss"):
            f = pq.ParquetFile(self.out / "parquet" / f"{r}.parquet")
            js[r] = []
            for i in range(f.num_row_groups):
                t = f.read_row_group(i, columns=["t"])["t"].to_numpy()
                j = int(bucket_of(t[:1], ELL)[0])
                with self.subTest(file=r, row_group=i):
                    self.assertTrue(np.all(t >= j * ELL) and np.all(t < (j + 1) * ELL))
                    self.assertTrue(np.all(np.diff(t) >= 0))                        # sorted by t
                    st = f.metadata.row_group(i).column(1).statistics                # column t
                    self.assertTrue(st.has_min_max)
                    self.assertEqual((st.min, st.max), (t.min(), t.max()))
                js[r].append(j)
        self.assertEqual(js["imu"], list(range(12)))                                 # 120 s / 10 s
        self.assertEqual(js["gnss"], js["imu"])                                      # common grid
        self.assertEqual((self.row["rg_imu"], self.row["rg_gnss"]), (12, 12))

    def test_values_equal_written(self):
        for r in ("imu", "gnss"):
            want, got = self.data[r].reset_index(drop=True), self.pq[r]
            self.assertEqual(list(got.columns), list(want.columns))
            for c in want.columns:
                np.testing.assert_array_equal(got[c].to_numpy(), want[c].to_numpy().astype(got[c].dtype), c)
        self.assertEqual((self.row["rows_imu"], self.row["rows_gnss"]), (len(self.data["imu"]), len(self.data["gnss"])))

    def test_gnss_stats(self):
        got = pd.read_csv(self.out / "meta" / "gnss_stats_bag5.csv")
        want = F.gnss_stats(self.data["gnss"], ELL)
        self.assertEqual(list(got.columns), list(want.columns))
        for c in want.columns:
            np.testing.assert_array_equal(got[c].to_numpy(), want[c].to_numpy(), c)   # exact (%.17g)

    def test_limit_s(self):
        for r in ("imu", "gnss"):
            t = pq.read_table(self.out30 / "parquet" / f"{r}.parquet")["t"].to_numpy()
            self.assertLess(t.max(), 30.0)
            self.assertEqual(len(t), int((self.data[r]["t"] < 30.0).sum()))
        self.assertEqual(self.row30["rg_imu"], 3)
        self.assertLess(self.row30["chunks_read"], self.row["chunks_read"])

    def test_log_row(self):
        self.assertGreater(self.row["peak_rss"], 0)
        self.assertGreater(self.row["convert_once"], 0)
        self.assertEqual(self.row["start_ns"], F.T0_NS)                              # = bag.start_ns

    def test_bucket_of_edges(self):
        t = np.array([0.0, 9.999999999999998, 10.0, 19.999999999999996, 20.0, 0.30000000000000004])
        j = bucket_of(t, 0.1)
        self.assertTrue(np.all(j * 0.1 <= t) and np.all(t < (j + 1) * 0.1))


if __name__ == "__main__":
    unittest.main()
