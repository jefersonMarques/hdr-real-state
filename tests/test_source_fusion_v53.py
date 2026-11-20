import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from tifffile import imread

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "poc"))

from source_fusion_v53 import (
    detect_exterior_mask,
    guided_filter,
    load_window_mask,
    merge_original_brackets,
    render_scene,
    scene_tone_parameters,
    smoothstep,
    source_select_chunk,
    tone_map_chunk,
)
from v5_core import export_final


class SourceFusionV53Tests(unittest.TestCase):
    def test_underexposed_source_restores_real_window_detail(self) -> None:
        radiance = np.full((48, 64, 3), 0.23, dtype=np.float32)
        radiance[12:36, 12:44] = 1.4
        radiance[14:35:2, 16:40:2] = 1.2
        factors = (0.5, 1.0, 2.0)
        frames = tuple(
            np.clip(radiance * factor, 0.0, 1.0)
            for factor in factors
        )
        valid = tuple(np.ones((48, 64), dtype=bool) for _ in range(3))

        merged, selection = source_select_chunk(frames, factors, valid)

        np.testing.assert_allclose(
            merged[15, 20], radiance[15, 20], atol=0.001
        )
        self.assertEqual(int(selection[15, 20]), 0)
        self.assertEqual(int(selection[1, 1]), 1)
        self.assertGreater(
            float(merged[15, 20, 0]), float(frames[1][15, 20, 0])
        )

    def test_bright_bracket_supplies_real_shadow_signal(self) -> None:
        radiance = np.full((24, 32, 3), 0.09, dtype=np.float32)
        factors = (0.5, 1.0, 2.0)
        frames = tuple(radiance * factor for factor in factors)
        valid = tuple(np.ones((24, 32), dtype=bool) for _ in range(3))

        merged, selection = source_select_chunk(frames, factors, valid)

        np.testing.assert_allclose(merged, radiance, atol=0.00001)
        self.assertEqual(int(selection[0, 0]), 2)

    def test_bad_highlight_source_is_not_used(self) -> None:
        frames = (
            np.full((20, 24, 3), 0.70, dtype=np.float32),
            np.full((20, 24, 3), 1.0, dtype=np.float32),
            np.full((20, 24, 3), 1.0, dtype=np.float32),
        )
        valid = (
            np.zeros((20, 24), dtype=bool),
            np.ones((20, 24), dtype=bool),
            np.ones((20, 24), dtype=bool),
        )
        merged, selection = source_select_chunk(
            frames, (0.5, 1.0, 2.0), valid
        )

        self.assertEqual(int(selection[10, 10]), 1)
        self.assertAlmostEqual(float(merged[10, 10, 0]), 1.0)

    def test_reference_protects_mismatched_structure(self) -> None:
        frames = (
            np.full((20, 24, 3), 0.65, dtype=np.float32),
            np.full((20, 24, 3), 0.68, dtype=np.float32),
            np.full((20, 24, 3), 0.90, dtype=np.float32),
        )
        valid = tuple(np.ones((20, 24), dtype=bool) for _ in range(3))
        merged, selection = source_select_chunk(
            frames, (0.5, 1.0, 2.0), valid
        )

        self.assertEqual(int(selection[10, 10]), 1)
        self.assertAlmostEqual(float(merged[10, 10, 0]), 0.68, places=5)

    def test_exterior_detection_does_not_spill_into_dark_wall(self) -> None:
        reference = np.full((120, 160, 3), 0.12, dtype=np.float32)
        under = np.full((120, 160, 3), 0.06, dtype=np.float32)
        reference[16:96, 8:84] = 0.93
        under[16:96, 8:84] = 0.45

        mask = detect_exterior_mask(under, reference)

        self.assertGreater(float(mask[48, 38]), 0.90)
        self.assertLess(float(mask[48, 110]), 0.05)
        self.assertLess(float(mask[110, 38]), 0.05)

    def test_guided_filter_retains_edge_transition(self) -> None:
        guide = np.zeros((60, 80), dtype=np.float32)
        guide[:, 40:] = 1.0
        source = guide.copy()

        result = guided_filter(guide, source, radius=3, regularization=0.001)

        self.assertLess(float(result[30, 20]), 0.05)
        self.assertGreater(float(result[30, 60]), 0.95)

    def test_local_contrast_keeps_exterior_texture_and_darker_shadow(self) -> None:
        hdr = np.full((72, 96, 3), 0.25, dtype=np.float32)
        hdr[12:60, 8:48] = 1.8
        hdr[14:58:2, 10:46:2] = 1.25
        hdr[50:70, 55:95] = 0.04
        mask = np.zeros((72, 96), dtype=np.float32)
        mask[12:60, 8:48] = 1.0

        params = scene_tone_parameters(
            hdr, mask, 0.47, 0.56, 1.08, 1.40
        )
        result = tone_map_chunk(
            hdr,
            params["base_preview"],
            mask,
            params,
            shadow_depth=0.3,
        )

        self.assertLess(float(np.mean(result[54:68, 60:88])), 0.47)
        self.assertLess(float(np.median(result[12:60, 8:48])), 0.90)
        self.assertGreater(
            float(result[13, 9, 0]), float(result[14, 10, 0])
        )
        self.assertTrue(np.isfinite(result).all())

    def test_invalid_window_mask_geometry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            Image.new("L", (32, 40), 255).save(path)

            with self.assertRaisesRegex(ValueError, "camera preview"):
                load_window_mask(path, (64, 96))

    def test_source_fusion_end_to_end_retains_native_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            height, width = 80, 112
            radiance = np.full((height, width, 3), 0.25, dtype=np.float32)
            radiance[12:56, 8:64] = 1.4

            for role, factor in zip(
                ("under", "reference", "over"), (0.5, 1.0, 2.0)
            ):
                path = root / f"{role}.npy"
                np.save(
                    path,
                    np.clip(radiance * factor, 0.0, 1.0).astype(np.float32),
                )
                records.append({
                    "role": role,
                    "linear_path": str(path),
                    "raw_path": str(root / f"{role}.ARW"),
                    "relative_exposure": factor,
                })

            manifest = root / "triplet_v5.json"
            manifest.write_text(
                json.dumps({"inputs": records}),
                encoding="utf-8",
            )

            with patch(
                "source_fusion_v53.estimate_translation",
                return_value=(0.0, 0.0, 1.0),
            ):
                merge = merge_original_brackets(
                    manifest, root / "experiment", rows_per_chunk=24
                )

            render = render_scene(
                manifest,
                Path(merge["merged_hdr"]),
                root / "experiment" / "grade",
                target_mid=0.47,
                window_target=0.56,
                base_contrast=1.08,
                detail_contrast=1.40,
                shadow_depth=0.30,
                rows_per_chunk=24,
            )
            exported = export_final(
                Path(merge["merged_hdr"]),
                Path(render["tiff"]),
                root / "experiment" / "final",
            )

            with Image.open(exported["jpeg"]) as image:
                self.assertEqual(image.size, (width, height))
            self.assertEqual(
                imread(exported["tiff"]).shape, (height, width, 3)
            )
            self.assertEqual(
                imread(exported["tiff"]).dtype, np.uint16
            )
            self.assertEqual(
                np.load(merge["source_selection"]).shape, (height, width)
            )
            self.assertGreater(render["exterior_mask_fraction"], 0.01)
            self.assertTrue(Path(render["exterior_mask_preview"]).exists())


if __name__ == "__main__":
    unittest.main()
