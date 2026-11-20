import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "poc"))

from hdr_diagnostics import brightness_metrics, exposure_value, merge_linear_brackets, tone_map


class HdrDiagnosticsTests(unittest.TestCase):
    def test_exposure_value_accounts_for_iso(self) -> None:
        reference = exposure_value({"ExposureTime": 1 / 40, "ISO": 100, "FNumber": 4})
        under = exposure_value({"ExposureTime": 1 / 320, "ISO": 100, "FNumber": 4})
        over = exposure_value({"ExposureTime": 1 / 20, "ISO": 400, "FNumber": 4})

        self.assertAlmostEqual(under / reference, 0.125)
        self.assertAlmostEqual(over / reference, 8.0)

    def test_weighted_merge_preserves_radiance(self) -> None:
        scene = np.full((16, 20, 3), 0.1, dtype=np.float32)
        factors = (0.125, 1.0, 8.0)
        inputs = tuple(np.clip(scene * factor, 0.0, 1.0) for factor in factors)
        result = merge_linear_brackets(inputs, factors)

        np.testing.assert_allclose(result, scene, rtol=1e-4, atol=1e-5)

    def test_weighted_merge_avoids_clipped_exposure(self) -> None:
        scene = np.full((12, 16, 3), 0.8, dtype=np.float32)
        factors = (0.125, 1.0, 8.0)
        inputs = tuple(np.clip(scene * factor, 0.0, 1.0) for factor in factors)
        result = merge_linear_brackets(inputs, factors)

        np.testing.assert_allclose(result, scene, rtol=1e-4, atol=1e-4)

    def test_tone_map_preserves_light_order(self) -> None:
        scene = np.array([[[0.02] * 3, [0.2] * 3, [4.0] * 3]], dtype=np.float32)
        preview = tone_map(scene)

        self.assertLess(int(preview[0, 0, 0]), int(preview[0, 1, 0]))
        self.assertLess(int(preview[0, 1, 0]), int(preview[0, 2, 0]))
        self.assertLess(int(preview[0, 2, 0]), 255)

    def test_brightness_metrics_on_dark_image(self) -> None:
        dark = np.zeros((10, 10, 3), dtype=np.uint8)
        metrics = brightness_metrics(dark)

        self.assertEqual(metrics["dark_pixel_fraction"], 1.0)
        self.assertEqual(metrics["near_white_pixel_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
