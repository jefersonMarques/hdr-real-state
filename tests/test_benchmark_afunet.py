import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "poc"))

from benchmark_afunet import (
    analyze_preview,
    mean_absolute_error,
    resize_float_image,
    run_checkpoint,
    verify_existing_baseline,
)


class AfunetCheckpointBenchmarkTests(unittest.TestCase):
    def test_absolute_difference_is_zero_for_matching_images(self) -> None:
        image = np.full((4, 5, 3), 127, dtype=np.uint8)
        self.assertEqual(mean_absolute_error(image, image), 0.0)

    def test_absolute_difference_matches_rgb_levels(self) -> None:
        first = np.zeros((4, 5, 3), dtype=np.uint8)
        second = np.full((4, 5, 3), 60, dtype=np.uint8)
        self.assertEqual(mean_absolute_error(first, second), 60.0)

    def test_baseline_must_have_same_raw_filenames_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text(
                json.dumps({"inputs": ["photos/under.ARW", "photos/reference.ARW", "photos/over.ARW"]}),
                encoding="utf-8",
            )
            paths = (
                Path("photos/under.ARW"),
                Path("photos/reference.ARW"),
                Path("photos/over.ARW"),
            )
            verify_existing_baseline(report, paths)

            wrong_paths = (paths[1], paths[0], paths[2])
            with self.assertRaisesRegex(ValueError, "different RAW filenames"):
                verify_existing_baseline(report, wrong_paths)

    def test_downsampling_keeps_float_data(self) -> None:
        image = np.full((120, 160, 3), 0.123456, dtype=np.float32)
        resized = resize_float_image(image, max_side=80)

        self.assertEqual(resized.shape, (60, 80, 3))
        self.assertEqual(resized.dtype, np.float32)
        np.testing.assert_allclose(resized, 0.123456, rtol=1e-5)

    def test_reference_error_is_explicitly_separate(self) -> None:
        baseline = np.zeros((8, 8, 3), dtype=np.uint8)
        neural = np.full((8, 8, 3), 100, dtype=np.uint8)
        professional = np.full((8, 8, 3), 80, dtype=np.uint8)
        result = analyze_preview(neural, baseline, professional)

        self.assertEqual(result["mean_absolute_difference_from_baseline_0_255"], 100.0)
        self.assertEqual(result["approximate_rgb_mae_from_professional_0_255"], 20.0)

    def test_model_inference_creates_distinct_preview_and_tiff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            checkpoint = folder / "Kalantari.pth"
            checkpoint.write_bytes(b"synthetic-checkpoint")
            image = np.full((64, 64, 3), 0.1, dtype=np.float32)
            predicted = np.full_like(image, 0.3)

            with patch("benchmark_afunet.load_afunet", return_value=object()):
                with patch("benchmark_afunet.reconstruct", return_value=predicted):
                    preview, elapsed, metadata = run_checkpoint(
                        "Kalantari",
                        checkpoint,
                        folder,
                        (image, image, image),
                        (0.5, 1.0, 2.0),
                        torch.device("cpu"),
                        128,
                        32,
                        folder,
                    )

            self.assertEqual(preview.shape, (64, 64, 3))
            self.assertGreaterEqual(elapsed, 0)
            self.assertEqual(metadata["raw_hdr_median"], 0.3)
            self.assertTrue((folder / "kalantari_hdr.tiff").is_file())
            self.assertTrue((folder / "kalantari_preview.png").is_file())


if __name__ == "__main__":
    unittest.main()
