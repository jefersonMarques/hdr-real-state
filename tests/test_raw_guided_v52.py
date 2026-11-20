import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "poc"))

from merge_hdr_raw_v52 import (
    build_sensor_confidence,
    calculate_sensor_confidence,
    determine_clip_code,
    merge_raw_guided,
)
from v5_core import fusion_chunk


class RawGuidedV52Tests(unittest.TestCase):
    def test_detects_repeatable_sony_style_plateau(self) -> None:
        samples = np.array(
            [2000] * 100 + [15360] * 8 + [16372] * 70 + [16596] * 2,
            dtype=np.uint16,
        )

        self.assertEqual(
            determine_clip_code(samples, 15360, 16380),
            16372.0,
        )

    def test_rejects_unsupported_or_unreliable_sensor_plateau(self) -> None:
        samples = np.arange(15000, 16500, 5, dtype=np.uint16)

        with self.assertRaisesRegex(ValueError, "not sufficiently established"):
            determine_clip_code(samples, 15360, 16380)

    def test_confidence_retains_declared_white_but_rejects_plateau(self) -> None:
        samples = np.array([512, 1000, 15360, 16000, 16372], dtype=np.uint16)
        confidence = calculate_sensor_confidence(samples, 512, 16372)

        self.assertEqual(float(confidence[0]), 0.0)
        self.assertGreater(float(confidence[1]), 0.99)
        self.assertGreater(float(confidence[2]), 0.99)
        self.assertGreater(float(confidence[3]), 0.99)
        self.assertEqual(float(confidence[4]), 0.0)

    def test_bayer_masks_exclude_high_code_saturation(self) -> None:
        mosaic = np.full((48, 64), 3500, dtype=np.uint16)
        mosaic[:24, :24] = 16372
        pattern = np.array([[0, 1], [3, 2]], dtype=np.uint8)
        colors = np.tile(pattern, (24, 32))

        camera = MagicMock()
        camera.__enter__.return_value = camera
        camera.raw_image_visible = mosaic
        camera.raw_colors_visible = colors
        camera.raw_pattern = pattern
        camera.black_level_per_channel = [512] * 4
        camera.camera_white_level_per_channel = [15360] * 4
        camera.white_level = 16380
        camera.sizes.flip = 0
        fake_rawpy = SimpleNamespace(imread=lambda path: camera)

        with patch.dict(sys.modules, {"rawpy": fake_rawpy}):
            confidence, report = build_sensor_confidence(
                Path("sample.ARW"),
                (48, 64),
                sampling_stride=2,
            )

        self.assertEqual(confidence.shape, (48, 64))
        self.assertLess(float(confidence[10, 10]), 0.05)
        self.assertGreater(float(confidence[36, 45]), 0.95)
        self.assertEqual(report["clip_codes_by_cfa_index"]["1"], 16372.0)

    def test_confidence_one_preserves_existing_v5_fusion(self) -> None:
        reference_scene = np.full((30, 36, 3), 0.3, dtype=np.float32)
        reference_scene[5:12, 8:15] = 1.4
        factors = (0.5, 1.0, 2.0)
        exposures = tuple(
            np.clip(reference_scene * exposure, 0.0, 1.0)
            for exposure in factors
        )
        baseline = fusion_chunk(exposures, factors)
        confidences = tuple(
            np.ones((30, 36), dtype=np.float32) for _ in range(3)
        )
        guided = fusion_chunk(
            exposures, factors, sensor_confidence=confidences
        )

        np.testing.assert_allclose(baseline, guided, rtol=1e-6, atol=1e-6)

    def test_confidence_rejects_untrustworthy_overexposure(self) -> None:
        exposures = (
            np.full((20, 25, 3), 0.10, dtype=np.float32),
            np.full((20, 25, 3), 0.35, dtype=np.float32),
            np.full((20, 25, 3), 0.70, dtype=np.float32),
        )
        factors = (0.5, 1.0, 2.0)
        masks = (
            np.ones((20, 25), dtype=np.float32),
            np.ones((20, 25), dtype=np.float32),
            np.zeros((20, 25), dtype=np.float32),
        )
        guided = fusion_chunk(
            exposures, factors, sensor_confidence=masks
        )

        self.assertLess(float(guided[3, 3, 0]), 0.35)

    def test_opt_in_merge_exports_independent_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for role, factor in zip(
                ("under", "reference", "over"), (0.5, 1.0, 2.0)
            ):
                linear = root / f"{role}.npy"
                raw_path = root / f"{role}.ARW"
                raw_path.touch()
                np.save(
                    linear,
                    np.full((32, 48, 3), 0.15 * factor, dtype=np.float32),
                )
                records.append({
                    "role": role,
                    "raw_path": str(raw_path),
                    "linear_path": str(linear),
                    "relative_exposure": factor,
                })

            manifest = root / "triplet_v5.json"
            manifest.write_text(
                json.dumps({"inputs": records}),
                encoding="utf-8",
            )

            def simulated_confidence(path, dimensions):
                return np.ones(dimensions, dtype=np.float32), {
                    "raw_path": str(path),
                    "clip_codes_by_cfa_index": {"0": 16372.0},
                }

            with patch(
                "merge_hdr_raw_v52.build_sensor_confidence",
                side_effect=simulated_confidence,
            ):
                report = merge_raw_guided(
                    manifest,
                    root / "guided",
                    alignment="none",
                    rows_per_chunk=10,
                )

            merged = np.load(root / "guided" / "merged_hdr.npy")
            self.assertEqual(merged.shape, (32, 48, 3))
            self.assertAlmostEqual(float(merged.mean()), 0.15, places=3)
            self.assertEqual(report["version"], "v5.2-experimental")
            self.assertTrue(
                (root / "guided" / "under_sensor_confidence.npy").exists()
            )
            self.assertFalse((root / "merged_hdr.npy").exists())


if __name__ == "__main__":
    unittest.main()
