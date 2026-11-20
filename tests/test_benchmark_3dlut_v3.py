import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "poc"))

from benchmark_3dlut_v3 import (
    apply_lut_chunk,
    apply_lut_full_resolution,
    build_classifier,
    infer_adaptive_lut,
    load_pretrained_model,
    run_benchmark,
)


class ConstantClassifier(nn.Module):
    def __init__(self, weights: tuple[float, float, float]):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.weights.reshape(1, 3, 1, 1)


def identity_lut() -> torch.Tensor:
    ramp = torch.linspace(0.0, 1.0, 33)
    blue, green, red = torch.meshgrid(ramp, ramp, ramp, indexing="ij")
    return torch.stack((red, green, blue), dim=0)


class AdaptiveLutTests(unittest.TestCase):
    def test_classifier_has_official_output_shape(self) -> None:
        classifier = build_classifier().eval()
        with torch.inference_mode():
            output = classifier(torch.full((1, 3, 256, 256), 0.5))
        self.assertEqual(tuple(output.shape), (1, 3, 1, 1))
        self.assertIn("model.1.weight", classifier.state_dict())

    def test_identity_lut_keeps_rgb_values(self) -> None:
        values = torch.tensor(
            [[[[0.2, 0.5]], [[0.4, 0.1]], [[0.7, 0.9]]]],
            dtype=torch.float32,
        )
        result = apply_lut_chunk(values, identity_lut())
        torch.testing.assert_close(result, values, atol=1e-5, rtol=1e-5)

    def test_adaptive_lut_combines_pretrained_bases(self) -> None:
        bases = torch.stack((identity_lut(), torch.zeros(3, 33, 33, 33), identity_lut()))
        image = torch.rand(1, 3, 40, 30)
        combined, weights = infer_adaptive_lut(
            ConstantClassifier((0.25, 0.5, 0.75)),
            bases,
            image,
        )
        torch.testing.assert_close(combined, identity_lut(), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(weights, torch.tensor((0.25, 0.5, 0.75)))

    def test_chunked_render_matches_full_render(self) -> None:
        image = np.random.default_rng(51).integers(0, 256, size=(57, 29, 3), dtype=np.uint8)
        full = apply_lut_full_resolution(image, identity_lut(), rows_per_chunk=100)
        tiled = apply_lut_full_resolution(image, identity_lut(), rows_per_chunk=11)
        np.testing.assert_array_equal(full, tiled)
        np.testing.assert_allclose(tiled, image, atol=1)

    def test_full_pipeline_with_synthetic_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = root / "pretrained"
            checkpoints.mkdir()
            classifier = build_classifier()
            torch.save(classifier.state_dict(), checkpoints / "classifier.pth")
            torch.save(
                {str(index): {"LUT": identity_lut()} for index in range(3)},
                checkpoints / "LUTs.pth",
            )
            loaded_classifier, loaded_luts = load_pretrained_model(
                checkpoints, torch.device("cpu")
            )
            self.assertEqual(tuple(loaded_luts.shape), (3, 3, 33, 33, 33))
            self.assertFalse(loaded_classifier.training)

            image_file = root / "source.png"
            Image.new("RGB", (80, 64), (30, 80, 160)).save(image_file)
            output = root / "output"
            run_benchmark(image_file, output, checkpoints, "cpu", 17)
            self.assertTrue((output / "adaptive_lut_enhanced.png").is_file())
            self.assertTrue((output / "model_report.json").is_file())

            with self.assertRaises(FileExistsError):
                run_benchmark(image_file, output, checkpoints, "cpu", 17)


if __name__ == "__main__":
    unittest.main()
