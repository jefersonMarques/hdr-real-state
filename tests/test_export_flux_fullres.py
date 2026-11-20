import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from tifffile import imread

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "poc"))

from export_flux_fullres import (
    apply_adjustments,
    box_blur,
    calculate_adjustments,
    export_photo,
    upscale_adjustments,
)


class FluxFullResolutionExportTests(unittest.TestCase):
    def test_box_blur_preserves_constant_fields(self) -> None:
        source = np.full((40, 60, 4), 0.15, dtype=np.float32)
        result = box_blur(source, radius=3)
        np.testing.assert_allclose(result, source, atol=1e-6)

    def test_identical_images_produce_neutral_adjustment_maps(self) -> None:
        pixels = np.full((64, 96, 3), 126, dtype=np.uint8)
        adjustments = calculate_adjustments(
            pixels,
            pixels,
            max_exposure_ev=0.65,
            max_color_log=0.12,
            blur_radius=2,
        )
        np.testing.assert_allclose(adjustments, 0.0, atol=1e-6)

    def test_brightness_shift_is_bounded(self) -> None:
        before = np.full((64, 96, 3), 50, dtype=np.uint8)
        after = np.full_like(before, 250)
        adjustments = calculate_adjustments(
            before,
            after,
            max_exposure_ev=0.5,
            max_color_log=0.1,
            blur_radius=2,
        )

        self.assertLessEqual(float(np.max(adjustments[..., 0])), np.log(2) * 0.5 + 1e-5)
        self.assertLessEqual(float(np.max(np.abs(adjustments[..., 1:]))), 0.1 + 1e-5)

    def test_chunked_render_preserves_detail_and_dimensions(self) -> None:
        rng = np.random.default_rng(41)
        source = rng.integers(0, 256, size=(78, 125, 3), dtype=np.uint8)
        controls = [np.zeros((78, 125), dtype=np.float32) for _ in range(4)]

        complete_8, complete_16 = apply_adjustments(source, controls, 0.0, 300)
        chunked_8, chunked_16 = apply_adjustments(source, controls, 0.0, 15)

        np.testing.assert_array_equal(complete_8, chunked_8)
        np.testing.assert_array_equal(complete_16, chunked_16)
        np.testing.assert_allclose(chunked_8, source, atol=1)
        self.assertEqual(chunked_16.dtype, np.uint16)

    def test_upscaling_adjustments_preserves_uniform_values(self) -> None:
        changes = np.full((12, 20, 4), 0.08, dtype=np.float32)
        scaled = upscale_adjustments(changes, (100, 60))

        self.assertEqual(len(scaled), 4)
        for channel in scaled:
            self.assertEqual(channel.shape, (60, 100))
            np.testing.assert_allclose(channel, 0.08, atol=1e-5)

    def test_high_resolution_jpeg_and_tiff_are_exported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            small_input = root / "flux_input.png"
            small_edit = root / "flux_edit.png"
            destination = root / "export"
            rng = np.random.default_rng(11)
            native_pixels = rng.integers(15, 240, size=(180, 300, 3), dtype=np.uint8)

            Image.fromarray(native_pixels).save(source)
            Image.new("RGB", (96, 64), (110, 120, 125)).save(small_input)
            Image.new("RGB", (96, 64), (115, 126, 130)).save(small_edit)

            export_photo(
                source,
                small_input,
                small_edit,
                destination,
                strength=0.75,
                max_exposure_ev=0.65,
                max_color_log=0.12,
                blur_radius=2,
                rows_per_chunk=40,
            )

            with Image.open(destination / "photo_fullres.jpg") as result:
                self.assertEqual(result.size, (300, 180))
                self.assertTrue(result.info.get("icc_profile"))
            tiff = imread(destination / "photo_fullres_16bit.tiff")
            self.assertEqual(tiff.dtype, np.uint16)
            self.assertEqual(tiff.shape, (180, 300, 3))

            report = json.loads(
                (destination / "export_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["source_dimensions"], [300, 180])
            self.assertEqual(report["flux_dimensions"], [96, 64])

            with self.assertRaises(FileExistsError):
                export_photo(
                    source, small_input, small_edit, destination,
                    strength=0.75, max_exposure_ev=0.65,
                    max_color_log=0.12, blur_radius=2, rows_per_chunk=40,
                )


if __name__ == "__main__":
    unittest.main()
