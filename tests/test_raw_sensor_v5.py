import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "poc"))

from inspect_raw_sensor_v5 import (
    channel_samples,
    inspect_raw_sensor,
    inspect_triplet,
    sensor_channel_statistics,
    validate_sensor_levels,
)


class RawSensorAuditTests(unittest.TestCase):
    def test_sensor_fractions_distinguish_black_and_white_limits(self) -> None:
        values = np.array([100, 101, 200, 1099, 1100], dtype=np.uint16)
        metrics = sensor_channel_statistics(values, 100, 1100)

        self.assertEqual(metrics["sample_count"], 5)
        self.assertAlmostEqual(metrics["near_black_fraction"], 0.4)
        self.assertAlmostEqual(metrics["sensor_saturated_fraction"], 0.2)
        self.assertAlmostEqual(metrics["near_white_fraction"], 0.4)

    def test_detects_high_code_plateau_above_declared_white(self) -> None:
        values = np.array(
            [512, 800, 15359, 15360, 16372, 16372, 16372, 16375],
            dtype=np.uint16,
        )
        metrics = sensor_channel_statistics(values, 512, 15360)

        self.assertAlmostEqual(metrics["above_declared_white_fraction"], 5 / 8)
        self.assertAlmostEqual(metrics["strictly_above_declared_white_fraction"], 4 / 8)
        self.assertEqual(metrics["sampled_raw_max"], 16375)
        self.assertEqual(metrics["dominant_high_code"], 16372)
        self.assertAlmostEqual(metrics["dominant_high_code_sample_fraction"], 3 / 8)
        self.assertAlmostEqual(metrics["sampled_max_code_fraction"], 1 / 8)

    def test_empty_high_tail_does_not_invent_saturation_plateau(self) -> None:
        values = np.array([512, 800, 900, 1200], dtype=np.uint16)
        metrics = sensor_channel_statistics(values, 512, 15360)

        self.assertIsNone(metrics["dominant_high_code"])
        self.assertEqual(metrics["dominant_high_code_sample_fraction"], 0.0)
        self.assertEqual(metrics["above_declared_white_fraction"], 0.0)

    def test_channel_sampling_includes_all_bayer_phases(self) -> None:
        pattern = np.array([[0, 1], [3, 2]], dtype=np.uint8)
        colors = np.tile(pattern, (16, 16))
        mosaic = (colors.astype(np.uint16) * 100 + 200)

        for channel in range(4):
            samples = channel_samples(mosaic, colors, channel, 8, 2)
            self.assertGreater(samples.size, 0)
            np.testing.assert_array_equal(
                samples, np.full(samples.shape, channel * 100 + 200)
            )

    def test_levels_use_camera_white_fallback(self) -> None:
        black, white = validate_sensor_levels(
            [512, 512, 512, 512], None, 16383
        )
        self.assertEqual(black, [512.0] * 4)
        self.assertEqual(white, [16383.0] * 4)

        with self.assertRaisesRegex(ValueError, "Invalid"):
            validate_sensor_levels([200, 200, 200, 200], [150] * 4, 16383)

    def test_mosaic_audit_reads_sensor_before_postprocess(self) -> None:
        pattern = np.array([[0, 1], [3, 2]], dtype=np.uint8)
        colors = np.tile(pattern, (8, 8))
        mosaic = np.full(colors.shape, 200, dtype=np.uint16)
        mosaic[colors == 0] = 4095
        camera = MagicMock()
        camera.__enter__.return_value = camera
        camera.raw_image_visible = mosaic
        camera.raw_colors_visible = colors
        camera.raw_pattern = pattern
        camera.black_level_per_channel = [64] * 4
        camera.camera_white_level_per_channel = [4095] * 4
        camera.white_level = 4095
        camera.color_desc = b"RGBG"

        fake_rawpy = SimpleNamespace(imread=lambda path: camera)

        with patch.dict(sys.modules, {"rawpy": fake_rawpy}):
            report = inspect_raw_sensor(Path("camera.ARW"), 4)

        self.assertEqual(report["visible_mosaic_dimensions"], [16, 16])
        self.assertEqual(report["libraw_global_white_level"], 4095)
        self.assertEqual(report["channels"]["channel_0"]["dominant_high_code"], 4095)
        self.assertEqual(report["channels"]["channel_0"]["sensor_saturated_fraction"], 1.0)
        self.assertEqual(report["channels"]["channel_1"]["sensor_saturated_fraction"], 0.0)
        camera.postprocess.assert_not_called()

    def test_triplet_audit_uses_metadata_order_without_writing_raws(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            input_paths = [
                folder / f"{role}.ARW" for role in ("under", "reference", "over")
            ]
            records = [{"ISO": 100} for _ in input_paths]
            normalized = {
                "visible_mosaic_dimensions": [16, 16],
                "channels": {},
            }

            with patch(
                "inspect_raw_sensor_v5.discover_triplet",
                return_value=(input_paths, [0.5, 1.0, 2.0], records),
            ):
                with patch(
                    "inspect_raw_sensor_v5.camera_white_balance",
                    return_value=(2.0, 1.0, 1.5, 1.0),
                ):
                    with patch(
                        "inspect_raw_sensor_v5.inspect_raw_sensor",
                        return_value=normalized,
                    ):
                        report = inspect_triplet(folder, folder / "audit", 8)

            self.assertEqual(len(report["inputs"]), 3)
            self.assertEqual(
                [record["role"] for record in report["inputs"]],
                ["under", "reference", "over"],
            )
            self.assertEqual(
                [record["relative_exposure"] for record in report["inputs"]],
                [0.5, 1.0, 2.0],
            )
            self.assertEqual(set(normalized), {"visible_mosaic_dimensions", "channels"})
            self.assertEqual(len({id(record) for record in report["inputs"]}), 3)
            self.assertTrue((folder / "audit" / "raw_sensor_report.json").is_file())
            self.assertFalse((folder / "audit" / "under.ARW").exists())


if __name__ == "__main__":
    unittest.main()
