"""roslab.queries: tests.yaml, numpy condition == SQL condition, intervals.   Run:  python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roslab import intervals as I  # noqa: E402
from roslab import queries as Q  # noqa: E402

RNG = np.random.default_rng(0)


def spec(**kw):
    base = dict(id="x", label="x", cond="gnss", target="imu", window="whole", point_a="plain", bag=5,
                ell=10.0, tol=0.1, gap=0.3)
    return Q.Spec(**{**base, **kw})


class TestQueries(unittest.TestCase):
    def test_load_tests(self):
        specs = Q.load_tests()
        self.assertEqual([s.id for s in specs], ["1", "2", "3", "4-10", "4-20", "4-40", "4-80", "5", "6", "6s"])
        s = Q.load_tests(["6s", "4-20"])
        self.assertEqual([x.id for x in s], ["6s", "4-20"])
        self.assertEqual((s[0].area, s[0].point_a, s[0].box.lat_min), ("W", "stats", -33.040128678))
        self.assertEqual(s[1].window_range(796.5), (350.0, 370.0))
        self.assertEqual(Q.load_tests(["1"])[0].window_range(796.5), (0.0, 796.5))
        t5 = Q.load_tests(["5"])[0]
        self.assertEqual((t5.cond, t5.target, t5.shock, t5.cond_cols), ("imu", "gnss", True, ["ax", "ay", "az"]))
        with self.assertRaises(KeyError):
            Q.load_tests(["7"])

    def test_sql_float_roundtrip(self):
        vals = list(RNG.uniform(-1e3, 1e3, 50)) + [0.1, 1 / 3, -33.040128678, 1e-300]
        con = duckdb.connect()
        for v in vals:
            self.assertEqual(con.execute(f"SELECT {Q.sql_float(v)}").fetchone()[0], float(v))

    def test_area_numpy_equals_sql(self):
        box = I.Box(-33.04, -33.0399, -60.883, -60.882)
        df = pd.DataFrame({"t": np.arange(2000) * 0.2,
                           "lat": RNG.uniform(-33.0401, -33.0398, 2000), "lon": RNG.uniform(-60.8832, -60.8818, 2000)})
        df.loc[:3, ["lat", "lon"]] = [[box.lat_min, box.lon_min], [box.lat_max, box.lon_max],
                                      [box.lat_min, box.lon_max + 1e-12], [box.lat_max, box.lon_min]]  # edges
        s = spec(area="A", box=box)
        sql = duckdb.connect().execute(f"SELECT t FROM df WHERE {s.cond_sql()} ORDER BY t").df()["t"].to_numpy()
        np.testing.assert_array_equal(df["t"].to_numpy()[s.cond_mask(df)], sql)
        self.assertTrue(0 < len(sql) < len(df))

    def test_shock_numpy_equals_sql(self):
        n = 5000
        df = pd.DataFrame({"t": np.arange(n) * 0.005, "ax": RNG.normal(0, 0.3, n), "ay": RNG.normal(0, 0.3, n),
                           "az": RNG.normal(9.81, 0.3, n)})
        s = spec(cond="imu", target="gnss", shock=True, g0=9.81, threshold=0.6)
        sql = duckdb.connect().execute(f"SELECT t FROM df WHERE {s.cond_sql()} ORDER BY t").df()["t"].to_numpy()
        np.testing.assert_array_equal(df["t"].to_numpy()[s.cond_mask(df)], sql)
        self.assertTrue(0 < len(sql) < n)
        with self.assertRaises(RuntimeError):
            spec(cond="imu", shock=True).cond_sql()

    def test_filter_and_in_intervals(self):
        t = np.round(np.arange(0, 4.01, 0.2), 6)
        inside = ((t >= 1.0) & (t <= 1.4)) | ((t >= 3.0) & (t <= 3.2))
        s = spec()
        np.testing.assert_allclose(Q.filter_intervals(t, inside, "exact", s, 0, 3.0), [[0.9, 1.5], [2.9, 3.0]])
        np.testing.assert_allclose(Q.filter_intervals(t, inside, "L3", s, 1.0, 4.0), [[1.0, 3.3]])
        got = Q.in_intervals([0.89, 0.9, 1.2, 1.5, 1.51, 2.95, 3.3, 3.31], [[2.9, 3.3], [0.9, 1.5]])
        np.testing.assert_array_equal(got, [False, True, True, True, False, True, True, False])
        self.assertFalse(Q.in_intervals([1.0], I.EMPTY).any())

    def test_result_hash(self):
        a = Q.result_hash(np.array([3, 1, 2], dtype=np.int64))
        self.assertEqual(a, Q.result_hash([1, 2, 3]))
        self.assertEqual(len(a), 16)
        self.assertNotEqual(a, Q.result_hash([1, 2, 4]))


if __name__ == "__main__":
    unittest.main()
