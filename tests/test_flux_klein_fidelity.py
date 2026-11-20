import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "poc"))

from flux_klein_fidelity import (
    build_prompt,
    compare_images,
    prepare_images,
    save_variation,
    validate_strengths,
)


class FluxKleinFidelityTests(unittest.TestCase):
    def test_validates_distinct_strengths_and_steps(self) -> None:
        self.assertEqual(validate_strengths([0.5, 0.75], 4), [0.5, 0.75])

        with self.assertRaisesRegex(ValueError, "too low"):
            validate_strengths([0.10], 4)

        with self.assertRaisesRegex(ValueError, "distinct"):
            validate_strengths([0.5, 0.5], 4)

        with self.assertRaisesRegex(ValueError, "within"):
            validate_strengths([1.4], 4)

    def test_optional_style_prompt_does_not_change_default(self) -> None:
        source_prompt = "Retouch only this real estate image."
        self.assertEqual(build_prompt(source_prompt, False), source_prompt)
        self.assertIn("Image 2", build_prompt(source_prompt, True))

    def test_preparation_uses_full_white_edit_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            reference_path = root / "professional.png"
            Image.new("RGB", (1400, 900), (70, 100, 140)).save(source_path)
            Image.new("RGB", (1400, 900), (90, 110, 120)).save(reference_path)
            source, mask, style = prepare_images(source_path, 1024, reference_path)

            self.assertEqual(source.size, (1024, 656))
            self.assertEqual(mask.mode, "L")
            self.assertEqual(mask.getextrema(), (255, 255))
            self.assertIsNotNone(style)
            self.assertLessEqual(max(style.size), 512)

    def test_brightness_metrics_change_with_image(self) -> None:
        source = Image.new("RGB", (64, 64), (80, 80, 80))
        edited = Image.new("RGB", (64, 64), (140, 140, 140))
        comparison = compare_images(source, edited)
        self.assertGreater(comparison["rgb_change_mae_0_255"], 50)
        self.assertLess(
            comparison["source_brightness_median"],
            comparison["edited_brightness_median"],
        )

    def test_variation_writes_diff_and_report_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Image.new("RGB", (64, 64), (120, 120, 120))
            edited = Image.new("RGB", (64, 64), (140, 115, 118))
            result = save_variation(source, edited, root, 0.5)
            self.assertTrue(Path(result["edited_file"]).is_file())
            self.assertTrue(Path(result["difference_file"]).is_file())
            self.assertEqual(result["strength"], 0.5)

    def test_detects_unexpected_geometry_dimensions(self) -> None:
        source = Image.new("RGB", (64, 64))
        edited = Image.new("RGB", (32, 32))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            save_variation(source, edited, Path("/tmp/unused"), 0.5)


if __name__ == "__main__":
    unittest.main()
