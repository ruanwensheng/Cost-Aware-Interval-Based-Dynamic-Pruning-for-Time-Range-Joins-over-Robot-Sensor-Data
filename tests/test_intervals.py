"""Unit tests for roslab.intervals.   Run:  python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roslab import intervals as I  # noqa: E402

# GNSS every 0.2 s from t=0 to t=4; inside the area during [1.0, 1.4] and [3.0, 3.2]
T = np.round(np.arange(0, 4.01, 0.2), 6)
INSIDE = ((T >= 1.0) & (T <= 1.4)) | ((T >= 3.0) & (T <= 3.2))


class TestMerge(unittest.TestCase):
    def test_union_overlap_and_touch(self):
        got = I.merge_intervals([[5, 6], [0, 1], [0.5, 2], [2, 3]])
        np.testing.assert_allclose(got, [[0, 3], [5, 6]])

    def test_merge_gap(self):
        iv = [[0, 1], [1.5, 2], [4, 5]]
        np.testing.assert_allclose(I.merge_intervals(iv, merge=0), iv)
        np.testing.assert_allclose(I.merge_intervals(iv, merge=0.6), [[0, 2], [4, 5]])
        np.testing.assert_allclose(I.merge_intervals(iv, merge=0.5), iv)   # gap 0.5 is not < 0.5

    def test_empty(self):
        self.assertEqual(I.merge_intervals([]).shape, (0, 2))

    def test_clip(self):
        np.testing.assert_allclose(I.clip([[0, 2], [3, 4], [5, 9]], 1, 6), [[1, 2], [3, 4], [5, 6]])
        self.assertEqual(len(I.clip([[0, 1]], 2, 3)), 0)


class TestArea(unittest.TestCase):
    def test_in_area_inclusive(self):
        box = I.Box(lat_min=-1, lat_max=1, lon_min=10, lon_max=11)
        got = I.in_area([-1, 0, 1.0001, 0], [10, 11, 10.5, 9.99], box)
        np.testing.assert_array_equal(got, [True, True, False, False])


class TestType1(unittest.TestCase):
    def test_exact_is_union(self):
        got = I.build(T, INSIDE, 1, "exact", tol=0.1)
        np.testing.assert_allclose(got, [[0.9, 1.5], [2.9, 3.3]])

    def test_L3_single_hull(self):
        np.testing.assert_allclose(I.build(T, INSIDE, 1, "L3", tol=0.1), [[0.9, 3.3]])

    def test_L4_visits(self):
        got = I.build(T, INSIDE, 1, "L4", tol=0.1, gap=0.3)
        np.testing.assert_allclose(got, [[0.9, 1.5], [2.9, 3.3]])

    def test_L4_large_gap_joins_visits(self):
        got = I.build(T, INSIDE, 1, "L4", tol=0.1, gap=2.0)    # 1.4 -> 3.0 is 1.6 s < 2.0
        np.testing.assert_allclose(got, [[0.9, 3.3]])

    def test_L4_merge(self):
        got = I.build(T, INSIDE, 1, "L4", tol=0.1, gap=0.3, merge=1.5)   # hole 1.4 s < 1.5
        np.testing.assert_allclose(got, [[0.9, 3.3]])


class TestType2(unittest.TestCase):
    def test_exact_until_next_point(self):
        # last inside point of each visit maps up to the NEXT GNSS point (which is outside)
        got = I.build(T, INSIDE, 2, "exact")
        np.testing.assert_allclose(got, [[1.0, 1.6], [3.0, 3.4]])

    def test_L4(self):
        np.testing.assert_allclose(I.build(T, INSIDE, 2, "L4", gap=0.3), [[1.0, 1.6], [3.0, 3.4]])

    def test_L3(self):
        np.testing.assert_allclose(I.build(T, INSIDE, 2, "L3"), [[1.0, 3.4]])

    def test_last_point_inside(self):
        t = np.array([0.0, 0.2, 0.4])
        got = I.build(t, [False, False, True], 2, "exact", t_after_last=0.6)
        np.testing.assert_allclose(got, [[0.4, 0.6]])
        self.assertTrue(np.isinf(I.build(t, [False, False, True], 2, "exact")[0, 1]))

    def test_isolated_inside_point_between_outside(self):
        t = np.array([0.0, 0.2, 0.4, 0.6])
        got = I.build(t, [False, True, False, True], 2, "exact")
        np.testing.assert_allclose(got, [[0.2, 0.4], [0.6, np.inf]])


class TestVisitsAndBlocks(unittest.TestCase):
    def test_visits(self):
        self.assertEqual(I.visits(T, INSIDE, gap=0.3), [(5, 7), (15, 16)])
        self.assertEqual(I.visits(T, np.zeros(len(T), bool), gap=0.3), [])

    def test_no_point_inside(self):
        for lvl in ("exact", "L3", "L4"):
            self.assertEqual(I.build(T, np.zeros(len(T), bool), 1, lvl).shape, (0, 2))

    def test_total_length_and_gaps(self):
        iv = [[0, 1], [3, 4.5]]
        self.assertAlmostEqual(I.total_length(iv), 2.5)
        np.testing.assert_allclose(I.gaps(iv), [2])

    def test_blocks_hit(self):
        np.testing.assert_array_equal(I.blocks_hit([[0.5, 1.5], [9.0, 9.2]], ell=1.0), [0, 1, 9])
        np.testing.assert_array_equal(I.blocks_hit([[12, 35]], ell=10.0), [1, 2, 3])
        self.assertEqual(len(I.blocks_hit(I.EMPTY, ell=10.0)), 0)


if __name__ == "__main__":
    unittest.main()
