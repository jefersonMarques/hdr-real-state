import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "poc"))

from benchmark_editor_v3 import (
    build_report,
    image_statistics,
    rgb_mae,
    run_comparison,
)


class EditorBenchmarkTests(unittest.TestCase):
    def test_matching_images_have_no_change(self) -> None:
        image = np.full((48, 64, 3), 127, dtype=np.uint8)
        self.assertEqual(rgb_mae(image, image), 0.0)

    def test_rgb_difference_is_measured_in_pixel_levels(self) -> None:
        first = np.zeros((48, 64, 3), dtype=np.uint8)
        second = np.full_like(first, 20)
        self.assertEqual(rgb_mae(first, second), 20.0)

    def test_finite_brightness_statistics(self) -> None:
        image = np.full((48, 64, 3), 128, dtype=np.uint8)
        statistics = image_statistics(image)
        self.assertAlmostEqual(statistics["brightness_median"], 128 / 255, places=4)

    def test_model_output_cannot_change_dimensions(self) -> None:
        before = np.zeros((48, 64, 3), dtype=np.uint8)
        after = np.zeros((42, 64, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "preserve original image dimensions"):
            build_report(before, after, None, {})

    def test_reference_comparison_is_explicitly_approximate(self) -> None:
        before = np.zeros((48, 64, 3), dtype=np.uint8)
        after = np.full_like(before, 100)
        reference = np.full((96, 128, 3), 100, dtype=np.uint8)
        report = build_report(before, after, reference, {})

        self.assertEqual(report["editor_change_mae_0_255"], 100.0)
        self.assertEqual(report["reference_comparison"]["editor_mae_0_255"], 0.0)
        self.assertTrue(report["reference_comparison"]["reference_resized"])

    def test_comparison_produces_files_and_rejects_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            before = base / "before.png"
            after = base / "after.png"
            reference = base / "reference.png"
            output = base / "results"
            Image.fromarray(np.full((48, 64, 3), 90, dtype=np.uint8)).save(before)
            Image.fromarray(np.full((48, 64, 3), 110, dtype=np.uint8)).save(after)
            Image.fromarray(np.full((48, 64, 3), 115, dtype=np.uint8)).save(reference)

            run_comparison(before, after, reference, output)

            self.assertTrue((output / "comparison.jpg").is_file())
            self.assertTrue((output / "report.json").is_file())

            with self.assertRaises(FileExistsError):
                run_comparison(before, after, reference, output)


if __name__ == "__main__":
    unittest.main()
