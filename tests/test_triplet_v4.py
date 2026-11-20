import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "poc"))

from prepare_triplet_v4 import calculate_size, develop_raw, discover_triplet
from flux_triplet_v4 import load_references, quality_diagnostics, save_outputs


class TripletNeuralV4Tests(unittest.TestCase):
    def test_exposure_sort_includes_shutter_and_iso(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ("middle.ARW", "bright.ARW", "dark.ARW"):
                (folder / name).touch()

            values = {
                "middle.ARW": (1 / 100, 100),
                "bright.ARW": (1 / 25, 100),
                "dark.ARW": (1 / 400, 100),
            }

            def mock_exif(command, **kwargs):
                records = [
                    {
                        "ExposureTime": values[Path(path).name][0],
                        "ISO": values[Path(path).name][1],
                        "FNumber": 4,
                    }
                    for path in command[6:]
                ]
                return SimpleNamespace(stdout=json.dumps(records))

            with patch("prepare_triplet_v4.subprocess.run", side_effect=mock_exif):
                paths, factors, _ = discover_triplet(folder)

            self.assertEqual(
                [path.name for path in paths],
                ["dark.ARW", "middle.ARW", "bright.ARW"],
            )
            np.testing.assert_allclose(factors, [0.25, 1.0, 4.0])

    def test_requires_three_raw_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "one.ARW").touch()

            with self.assertRaisesRegex(ValueError, "exactly 3"):
                discover_triplet(Path(directory))

    def test_calculate_size_preserves_aspect_ratio(self) -> None:
        self.assertEqual(calculate_size(6024, 4024, 768), (768, 512))
        self.assertEqual(calculate_size(6024, 4024, 1024), (1024, 672))

    def test_calculate_size_avoids_float_rounding(self) -> None:
        self.assertEqual(calculate_size(6024, 4024, 1024), (1024, 672))
        self.assertEqual(calculate_size(4024, 6024, 1024), (672, 1024))
        self.assertEqual(calculate_size(6000, 4000, 768), (768, 512))

    def test_rawpy_receives_white_balance_as_list(self) -> None:
        raw_camera = MagicMock()
        raw_camera.__enter__.return_value = raw_camera
        raw_camera.postprocess.return_value = np.full(
            (48, 64, 3), 32768, dtype=np.uint16
        )
        fake_rawpy = SimpleNamespace(
            imread=lambda path: raw_camera,
            ColorSpace=SimpleNamespace(sRGB=1),
        )

        with patch.dict(sys.modules, {"rawpy": fake_rawpy}):
            image = develop_raw(Path("example.ARW"), (2.0, 1.0, 1.5, 1.0))

        self.assertEqual(image.size, (64, 48))
        self.assertEqual(
            raw_camera.postprocess.call_args.kwargs["user_wb"],
            [2.0, 1.0, 1.5, 1.0],
        )

    def test_references_are_loaded_in_fixed_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            input_records = []

            for label, value in zip(("under", "reference", "over"), (20, 90, 200)):
                path = folder / f"{label}.png"
                Image.new("RGB", (64, 48), (value, value, value)).save(path)
                input_records.append({"role": label})

            manifest = {"inputs": input_records, "prepared_dimensions": [64, 48]}
            (folder / "triplet.json").write_text(json.dumps(manifest), encoding="utf-8")

            images, loaded_manifest = load_references(folder)

            self.assertEqual([image.getpixel((0, 0))[0] for image in images], [20, 90, 200])
            self.assertEqual(loaded_manifest["prepared_dimensions"], [64, 48])

    def test_references_reject_mismatched_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for label in ("under", "reference", "over"):
                dimensions = (80, 48) if label == "over" else (64, 48)
                Image.new("RGB", dimensions).save(folder / f"{label}.png")

            manifest = {
                "inputs": [{"role": label} for label in ("under", "reference", "over")],
                "prepared_dimensions": [64, 48],
            }
            (folder / "triplet.json").write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "identical"):
                load_references(folder)

    def test_quality_metrics_identical_images_have_no_change(self) -> None:
        image = Image.new("RGB", (64, 48), (110, 110, 110))
        report = quality_diagnostics(image, image)

        self.assertEqual(report["rgb_change_mae_0_255"], 0.0)
        self.assertEqual(
            report["reference_near_white_fraction"],
            report["output_near_white_fraction"],
        )

    def test_saves_unmodified_generated_output_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = Image.new("RGB", (64, 48), (110, 110, 110))
            edited = Image.new("RGB", (64, 48), (120, 115, 105))

            save_outputs(
                folder,
                source,
                edited,
                {"inputs": [{"role": role} for role in ("under", "reference", "over")]},
                {"steps": 4, "seed": 0},
            )

            with Image.open(folder / "neural_triplet.png") as result:
                self.assertEqual(result.getpixel((0, 0)), (120, 115, 105))

            self.assertTrue((folder / "changes.png").is_file())
            self.assertTrue((folder / "neural_triplet_preview.jpg").is_file())

            report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["inference_parameters"]["seed"], 0)
            self.assertIn("not a validated HDR fusion model", report["method"])


if __name__ == "__main__":
    unittest.main()
