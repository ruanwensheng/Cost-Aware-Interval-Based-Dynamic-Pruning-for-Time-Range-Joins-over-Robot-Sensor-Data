"""Hand-computed cases for roslab.costfn (PROMPT_cost_function.md, section 5).

Run:  python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roslab import costfn as C  # noqa: E402

# c_req 1 ms, bw 100 MB/s, bz2 1 MB/s, c_q0 10 ms, c_int 2 ms
SSD = C.LayoutConsts("ssd", True, c_req=0.001, bw=1e8, v_dec=1e6, c_q0=0.01, c_int=0.002)
LAN = C.LayoutConsts("lan", False, c_req=0.001, bw=1e8, v_dec=1e6, c_q0=0.01, c_int=0.002)


class TestLogFields(unittest.TestCase):
    def test_est_of_the_level_that_ran(self):
        d = C.Decision("phase1", "L4", "L4", {"C_L2": 5.0, "C_p1": 1.0, "total_L3": 4.0, "total_L4": 3.0,
                                              "est": 3.0})
        self.assertEqual((C.log_fields(d, "L2", "L4")["est_cost"], C.log_fields(d, "L2", "L4")["est_auto"]),
                         (5.0, 3.0))                                              # forced L2, auto L4
        self.assertEqual(C.log_fields(d, "L3", "L4")["est_cost"], 4.0)
        a = C.Decision("L2", None, "L2", {"C_L2": 5.0, "C_p1": 6.0, "est": 5.0})
        self.assertEqual(C.log_fields(a, "L2", "L2")["est_cost"], 5.0)
        self.assertIsNone(C.log_fields(a, "L3", "L2")["est_cost"])             # point B not computed


class TestBag(unittest.TestCase):
    # D = 100 s, 1 cond chunk/s of 1 MB, 10 target chunks/s of 2 MB, 1 chunk/s holds both
    X = C.BagInputs(D=100, r_c=1, r_t=10, r_ct=1, b_c=1e6, b_t=2e6)

    def test_block(self):
        self.assertAlmostEqual(C.block(2e6, SSD), 0.001 + 2.0)               # no b/bw term on ssd
        self.assertAlmostEqual(C.block(2e6, LAN), 0.001 + 0.02 + 2.0)

    def test_costs(self):
        c = C.bag_costs(self.X, SSD)
        self.assertAlmostEqual(c["C_L2"], 10 * 100 * 2.001)                  # 2001
        self.assertAlmostEqual(c["C_p1"], 1 * 100 * 1.001)                   # 100.1
        self.assertAlmostEqual(C.bag_rest(self.X, SSD, 50), 9 * 50 * 2.001)  # 900.45

    def test_l2_includes_condition(self):
        x = C.BagInputs(D=100, r_c=1, r_t=10, r_ct=0.5, b_c=1e6, b_t=2e6)
        c = C.bag_costs(x, SSD, l2_includes_condition=True)
        self.assertAlmostEqual(c["C_L2"], 2001 + 0.5 * 100 * 1.001)

    def test_point_a_phase1(self):
        a = C.bag_point_a(self.X, SSD)
        self.assertEqual((a.point_a, a.level), ("phase1", None))

    def test_point_a_l2_wins(self):
        x = C.BagInputs(D=100, r_c=30, r_t=10, r_ct=1, b_c=1e6, b_t=2e6)    # C_p1 = 3003 > 2001
        a = C.bag_point_a(x, SSD)
        self.assertEqual((a.point_a, a.level), ("L2", "L2"))
        self.assertAlmostEqual(a.est["est"], 2001)

    def test_point_a_stats(self):
        # C_p1 + C_rest(T_ub): 100.1 + 9*100*2.001 = 1901 < 2001 ; with T_ub 200: 3702 > 2001
        self.assertEqual(C.bag_point_a(self.X, SSD, T_ub=100).point_a, "phase1")
        self.assertEqual(C.bag_point_a(self.X, SSD, T_ub=200).point_a, "L2")

    def test_point_b_l4(self):
        a = C.bag_point_a(self.X, SSD)
        b = C.bag_point_b(self.X, SSD, np.array([[0, 50.]]), np.array([[0, 5.], [45, 50.]]), a)
        self.assertEqual(b.level, "L4")
        self.assertAlmostEqual(b.est["C_L3"], 9 * 50 * 2.001)
        self.assertAlmostEqual(b.est["C_L4"], 9 * 10 * 2.001)
        self.assertAlmostEqual(b.est["est"], 100.1 + 9 * 10 * 2.001)
        np.testing.assert_allclose(b.filter, [[0, 5], [45, 50]])

    def test_point_b_tie_picks_l3(self):
        a = C.bag_point_a(self.X, SSD)
        iv = np.array([[10, 20.]])
        self.assertEqual(C.bag_point_b(self.X, SSD, iv, iv, a).level, "L3")


class TestParquet(unittest.TestCase):
    # D = 100 s, ell = 10 s; rg(rho_t) = 0.001 + 10*1e5/1e8 = 0.011 ; rg(rho_c) = 0.001 + 10*1e3/1e8 = 0.0011
    X = C.PqInputs(D=100, ell=10, rho_c=1e3, rho_t=1e5, n_c=11, n_t=12)

    def test_rg_and_counts(self):
        self.assertAlmostEqual(C.rg(1e5, 10, SSD), 0.011)
        self.assertAlmostEqual(C.n_l2(self.X), 10)                           # (1) formula: D / ell
        self.assertEqual(C.n_l2(self.X, "exact"), 12)                        # (1) exact: n_t
        self.assertAlmostEqual(C.n_l3(50, 10), 6)                            # (2) (span + ell) / ell
        self.assertAlmostEqual(C.n_l4(10, 2, 10), 3)                         # (3) (sum_w + k ell) / ell

    def test_costs(self):
        c = C.pq_costs(self.X, SSD)
        self.assertAlmostEqual(c["C_L2"], 10 * 0.011)                        # (4) 0.11
        self.assertAlmostEqual(c["C_p1"], 0.01 + 0.001 + 11 * 0.0011)        # (5) 0.0231
        c = C.pq_costs(self.X, SSD, l2_includes_condition=True)
        self.assertAlmostEqual(c["C_L2"], 0.11 + 11 * 0.0011)

    def test_merge_gap(self):
        self.assertAlmostEqual(C.merge_gap(self.X, SSD), 0.002 * 10 / 0.011)  # (7) g* = 1.818 s

    def test_point_a(self):
        self.assertEqual(C.pq_point_a(self.X, SSD).point_a, "phase1")
        x = C.PqInputs(D=100, ell=10, rho_c=1e3, rho_t=1e5, n_c=100, n_t=10)  # C_p1 = 0.121 > 0.11
        self.assertEqual(C.pq_point_a(x, SSD).level, "L2")

    def test_point_a_stats(self):
        # lhs = 0.0231 + n_hit * 0.011 : n_hit 3 -> 0.0561 < 0.11 ; n_hit 9 -> 0.1221 > 0.11
        a = C.pq_point_a(self.X, SSD, n_hit=3)
        self.assertEqual((a.point_a, a.est["T_ub"]), ("phase1", 30))
        self.assertEqual(C.pq_point_a(self.X, SSD, n_hit=9).point_a, "L2")

    def test_point_b_l4(self):
        a = C.pq_point_a(self.X, SSD)
        b = C.pq_point_b(self.X, SSD, np.array([[0, 50.]]), np.array([[0, 5.], [45, 50.]]), a)
        self.assertAlmostEqual(b.est["C_L3"], 6 * 0.011)                     # (6) 0.066
        self.assertAlmostEqual(b.est["C_L4"], 3 * 0.011 + 1 * 0.002)         # (6) 0.035
        self.assertEqual((b.level, b.est["k"]), ("L4", 2))
        self.assertAlmostEqual(b.est["est"], 0.0231 + 0.035)

    def test_point_b_merge_then_tie(self):
        # visits [0,5] and [6,10]: hole 1 s < g* 1.818 s -> merged to [0,10]; then L3 = L4 -> L3
        a = C.pq_point_a(self.X, SSD)
        b = C.pq_point_b(self.X, SSD, np.array([[0, 10.]]), np.array([[0, 5.], [6, 10.]]), a)
        self.assertEqual((b.est["k"], b.est["k_raw"]), (1, 2))
        self.assertAlmostEqual(b.est["C_L3"], b.est["C_L4"])
        self.assertEqual(b.level, "L3")

    def test_point_b_exact_blocks(self):
        a = C.pq_point_a(self.X, SSD)
        b = C.pq_point_b(self.X, SSD, np.array([[0, 50.]]), np.array([[0, 5.]]), a, n_exact=(4, 1))
        self.assertAlmostEqual(b.est["C_L3"], 4 * 0.011)
        self.assertAlmostEqual(b.est["C_L4"], 1 * 0.011)

    def test_missing_duckdb_constants(self):
        k = C.LayoutConsts("ssd", True, 0.001, 1e8, 1e6, None, None)
        with self.assertRaises(C.MissingConstant) as e:
            C.pq_costs(self.X, k)
        self.assertIn("bench_duckdb.py --layout ssd", str(e.exception))


if __name__ == "__main__":
    unittest.main()
