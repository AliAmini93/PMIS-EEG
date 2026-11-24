"""R11-A: the R10-B C0 geometry behind the overlap-label erratum (no data needed)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from experiments.pmis_r10b import geometry as geo  # noqa: E402


class C0OverlapErratum(unittest.TestCase):
    def test_c0_is_75_percent_overlap(self):
        self.assertEqual(geo.WLEN_MS, 2000)
        self.assertEqual(geo.C0_STRIDE, 500)
        self.assertEqual(geo.stride_offsets(geo.C0_STRIDE), (0, 500, 1000, 1500, 2000))
        self.assertEqual(geo.overlap_fraction(geo.C0_STRIDE), 0.75)
        self.assertEqual(geo.shared_samples(geo.C0_STRIDE), 1500)

    def test_fifty_percent_needs_stride_1000(self):
        self.assertEqual(geo.overlap_fraction(1000), 0.5)

    def test_erratum_record_states_75(self):
        from evaluation.r11_a_c0_erratum_check import recorded_geometry

        self.assertEqual(recorded_geometry()["adjacent_overlap_fraction"], 0.75)


if __name__ == "__main__":
    unittest.main()
