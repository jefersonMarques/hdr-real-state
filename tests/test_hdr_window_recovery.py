import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "poc"))

from hdr_window_recovery import create_highlight_mask, keep_large_components, recover_highlights


class WindowRecoveryTests(unittest.TestCase):
    def test_component_filter_excludes_small_lights(self):
        candidates = np.zeros((100, 100), dtype=bool)
        candidates[15:55, 20:70] = True
        candidates[70:72, 75:77] = True
        retained = keep_large_components(candidates, minimum_fraction=0.01)
        self.assertTrue(retained[30, 30])
        self.assertFalse(retained[70, 75])

    def test_window_mask_uses_both_exposures(self):
        under = np.full((200, 200, 3), 0.08, dtype=np.float32)
        reference = np.full_like(under, 0.45)
        under[30:110, 30:110] = 0.55
        reference[30:110, 30:110] = 0.99
        under[125:135, 145:155] = 0.9
        reference[125:135, 145:155] = 0.99
        mask = create_highlight_mask(under, reference)
        self.assertGreater(mask[60, 60], 0.9)
        self.assertLess(mask[160, 160], 0.01)
        self.assertLess(mask[130, 150], 0.1)

    def test_window_pixels_come_from_underexposed_source(self):
        base = np.full((5, 6, 3), 0.95, dtype=np.float32)
        under = np.full((5, 6, 3), 0.4, dtype=np.float32)
        mask = np.ones((5, 6), dtype=np.float32)
        result = recover_highlights(base, under, mask, 0.0)
        np.testing.assert_allclose(result, under, atol=1e-6)

    def test_dark_regions_remain_unchanged(self):
        base = np.full((5, 6, 3), 0.75, dtype=np.float32)
        under = np.full((5, 6, 3), 0.08, dtype=np.float32)
        mask = np.zeros((5, 6), dtype=np.float32)
        result = recover_highlights(base, under, mask, 0.0)
        np.testing.assert_allclose(result, base, atol=1e-6)

    def test_mismatch_rejected(self):
        base = np.zeros((3, 4, 3), dtype=np.float32)
        under = np.zeros((4, 4, 3), dtype=np.float32)
        mask = np.zeros((3, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            recover_highlights(base, under, mask, 0.0)


if __name__ == "__main__":
    unittest.main()
