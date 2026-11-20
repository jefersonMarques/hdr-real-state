import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
from tifffile import imread

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "poc"))

from benchmark_v5 import evaluate_v5
from v5_core import (
    aligned_chunk,
    encode_srgb_16,
    estimate_translation,
    export_final,
    fusion_chunk,
    merge_linear_triplet,
    prepare_linear_triplet,
    read_manifest,
    tone_map_file,
    tone_map_linear,
)


class V5PipelineTests(unittest.TestCase):
    def test_merges_exposures_without_clipping_real_window_values(self) -> None:
        reference_scene = np.full((36, 48, 3), 0.2, dtype=np.float32)
        reference_scene[8:25, 15:30] = 2.4
        factors = (0.125, 1.0, 8.0)
        brackets = tuple(
            np.clip(reference_scene * factor, 0.0, 1.0)
            for factor in factors
        )

        recovered = fusion_chunk(brackets, factors)

        np.testing.assert_allclose(recovered[0, 0], 0.2, rtol=0.01)
        np.testing.assert_allclose(recovered[10, 20], 2.4, rtol=0.02)
        self.assertGreater(float(recovered[10, 20, 0]), 1.0)

    def test_exposed_interior_shadow_is_recovered(self) -> None:
        scene = np.full((32, 44, 3), 0.01, dtype=np.float32)
        factors = (0.125, 1.0, 8.0)
        images = tuple(
            np.clip(scene * factor, 0.0, 1.0)
            for factor in factors
        )
        merged = fusion_chunk(images, factors)
        self.assertAlmostEqual(float(merged[15, 25, 0]), 0.01, delta=0.002)

    def test_rejects_wrong_exposure_order(self) -> None:
        inputs = tuple(np.zeros((10, 12, 3), dtype=np.float32) for _ in range(3))
        with self.assertRaisesRegex(ValueError, "increase"):
            fusion_chunk(inputs, (1.0, 0.5, 2.0))

    def test_registers_simple_camera_translation(self) -> None:
        rng = np.random.default_rng(17)
        reference = rng.random((144, 180, 3), dtype=np.float32)
        shifted = cv2.warpAffine(
            reference,
            np.float32([[1, 0, 4], [0, 1, -3]]),
            (180, 144),
        )
        dx, dy, confidence = estimate_translation(reference, shifted)
        self.assertAlmostEqual(dx, 4.0, delta=1.5)
        self.assertAlmostEqual(dy, -3.0, delta=1.5)
        self.assertGreater(confidence, 0.04)

        aligned, valid = aligned_chunk(shifted, 0, 144, (dx, dy))
        difference = np.abs(aligned[10:-10, 10:-10] - reference[10:-10, 10:-10])
        self.assertLess(float(difference.mean()), 0.12)
        self.assertGreater(float(valid.mean()), 0.9)

    def test_preparation_preserves_camera_wb_for_all_exposures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            photos = folder / "photos"
            photos.mkdir()
            paths = [photos / f"{name}.ARW" for name in ("under", "reference", "over")]
            for path in paths:
                path.touch()

            factors = [0.25, 1.0, 4.0]
            exposures = [
                {"ExposureTime": factor, "ISO": 100, "FNumber": 4}
                for factor in factors
            ]
            seen = []

            def decode(path, white_balance):
                seen.append(tuple(white_balance))
                level = factors[paths.index(path)]
                return np.full((48, 64, 3), level / 4, dtype=np.float32)

            with patch(
                "v5_core.discover_triplet",
                return_value=(paths, factors, exposures),
            ):
                with patch("v5_core.camera_white_balance", return_value=(2, 1, 1.5, 1)):
                    with patch("v5_core.load_linear_raw", side_effect=decode):
                        manifest = prepare_linear_triplet(photos, folder / "prepared")

            self.assertEqual(len(manifest["inputs"]), 3)
            self.assertEqual(seen, [(2, 1, 1.5, 1)] * 3)
            arrays, loaded_factors, _ = read_manifest(
                folder / "prepared" / "triplet_v5.json"
            )
            self.assertEqual(arrays[0].shape, (48, 64, 3))
            self.assertEqual(loaded_factors, factors)

    def test_merge_writes_float32_hdr_with_original_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            paths = []
            for label, factor in zip(("under", "reference", "over"), (0.25, 1.0, 4.0)):
                raw = folder / f"{label}_linear.npy"
                np.save(raw, np.full((50, 72, 3), min(0.12 * factor, 1.0), dtype=np.float32))
                paths.append({
                    "role": label,
                    "linear_path": str(raw),
                    "relative_exposure": factor,
                })
            manifest = folder / "triplet_v5.json"
            manifest.write_text(json.dumps({"inputs": paths}), encoding="utf-8")

            output = folder / "merge"
            report = merge_linear_triplet(
                manifest, output, alignment="none", rows_per_chunk=13
            )
            result = np.load(output / "merged_hdr.npy")
            self.assertEqual(result.shape, (50, 72, 3))
            self.assertEqual(result.dtype, np.float32)
            self.assertEqual(report["alignment"], "none")
            self.assertAlmostEqual(float(np.median(result)), 0.12, delta=0.005)

    def test_tone_map_is_finite_and_monotonic(self) -> None:
        source = np.stack(
            [
                np.full((48, 64), 0.03, dtype=np.float32),
                np.full((48, 64), 0.3, dtype=np.float32),
                np.full((48, 64), 2.4, dtype=np.float32),
            ],
            axis=2,
        )
        result = tone_map_linear(source, local_strength=0.0)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(float(result[0, 0, 1]), float(result[0, 0, 0]))
        self.assertGreater(float(result[0, 0, 2]), float(result[0, 0, 1]))
        self.assertLessEqual(float(result.max()), 1.0)
        self.assertEqual(encode_srgb_16(result).dtype, np.uint16)

    def test_shadow_contrast_deepens_shadows_without_changing_highlights(self) -> None:
        levels = np.array([0.01, 0.06, 0.15, 0.50, 2.0], dtype=np.float32)
        hdr = np.broadcast_to(
            levels.reshape(1, 5, 1), (48, 5, 3)
        ).copy()

        baseline = tone_map_linear(
            hdr, target_mid=0.42, local_strength=0.0, shadow_contrast=0.0
        )
        protected = tone_map_linear(
            hdr, target_mid=0.42, local_strength=0.0, shadow_contrast=0.50
        )

        self.assertLess(float(protected[0, 0, 0]), float(baseline[0, 0, 0]))
        self.assertLess(float(protected[0, 1, 0]), float(baseline[0, 1, 0]))
        np.testing.assert_allclose(
            protected[0, -1], baseline[0, -1], atol=1e-6
        )
        self.assertTrue(np.isfinite(protected).all())

    def test_reduced_target_mid_darkens_interior_without_reversing_lightness(self) -> None:
        levels = np.linspace(0.02, 1.3, 96, dtype=np.float32)
        hdr = np.repeat(levels.reshape(8, 12, 1), 3, axis=2)
        brighter = tone_map_linear(
            hdr, target_mid=0.42, local_strength=0.0
        )
        balanced = tone_map_linear(
            hdr, target_mid=0.37, local_strength=0.0
        )

        self.assertLess(float(np.median(balanced)), float(np.median(brighter)))
        self.assertGreater(float(np.min(np.diff(balanced[:, :, 0].ravel()))), 0.0)

    def test_shadow_contrast_rejects_out_of_range_strength(self) -> None:
        hdr = np.full((32, 48, 3), 0.2, dtype=np.float32)

        for strength in (-0.1, 0.96):
            with self.subTest(strength=strength):
                with self.assertRaisesRegex(ValueError, "Shadow contrast"):
                    tone_map_linear(hdr, shadow_contrast=strength)

    def test_tone_map_report_records_render_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hdr_path = root / "hdr.npy"
            np.save(hdr_path, np.full((64, 96, 3), 0.2, dtype=np.float32))

            report = tone_map_file(
                hdr_path,
                root / "balanced",
                target_mid=0.37,
                local_strength=0.15,
                shadow_contrast=0.50,
            )

            self.assertEqual(
                report["tone_mapping"],
                {
                    "target_mid": 0.37,
                    "local_strength": 0.15,
                    "shadow_contrast": 0.50,
                },
            )
            self.assertTrue(Path(report["tiff"]).is_file())

    def test_export_creates_24_megapixel_equivalent_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            hdr_path = folder / "merged_hdr.npy"
            np.save(hdr_path, np.full((100, 160, 3), 0.2, dtype=np.float32))

            tone_report = tone_map_file(hdr_path, folder)
            self.assertTrue(Path(tone_report["tiff"]).is_file())
            tiff = imread(folder / "tone_mapped_16bit.tiff")
            self.assertEqual(tiff.dtype, np.uint16)

            export = export_final(
                hdr_path, folder / "tone_mapped_16bit.tiff", folder / "final"
            )
            with Image.open(export["jpeg"]) as result:
                self.assertEqual(result.size, (160, 100))
                self.assertTrue(result.info.get("icc_profile"))
            self.assertEqual(imread(export["tiff"]).shape, (100, 160, 3))

    def test_comparison_generates_report_without_fake_registration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "before.png"
            final = folder / "after.png"
            professional = folder / "professional.png"
            Image.new("RGB", (80, 60), (90, 90, 90)).save(source)
            Image.new("RGB", (160, 120), (125, 125, 125)).save(final)
            Image.new("RGB", (160, 120), (140, 140, 140)).save(professional)

            report = evaluate_v5(source, final, professional, folder / "review")
            self.assertTrue((folder / "review" / "comparison.jpg").is_file())
            self.assertIn("difference_from_professional_0_255", report)
            self.assertIn("Unregistered", report["pixel_difference_caveat"])


if __name__ == "__main__":
    unittest.main()
