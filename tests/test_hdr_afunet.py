import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "poc"))

from hdr_afunet import prepare_features, reconstruct, tile_origins


class ReferenceModel:
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, 9:12, :, :]


class NeuralHdrTests(unittest.TestCase):
    def test_tile_coverage(self) -> None:
        self.assertEqual(tile_origins(295, 128, 32), [0, 96, 167])
        self.assertEqual(tile_origins(64, 128, 32), [0])

    def test_feature_channels(self) -> None:
        image = np.full((64, 64, 3), 0.5, dtype=np.float32)
        features = prepare_features((image, image, image), (0.25, 1.0, 4.0))
        self.assertEqual(tuple(features.shape), (1, 18, 64, 64))

    def test_tiled_blending(self) -> None:
        generator = np.random.default_rng(7)
        reference = generator.random((189, 207, 3), dtype=np.float32)
        output = reconstruct(
            (reference * 0.3, reference, reference * 0.8),
            (0.25, 1.0, 4.0),
            ReferenceModel(),
            torch.device("cpu"),
            128,
            32,
        )
        np.testing.assert_allclose(output, reference, rtol=1e-5, atol=1e-5)

    def test_tiny_image(self) -> None:
        reference = np.full((24, 36, 3), 0.4, dtype=np.float32)
        output = reconstruct(
            (reference, reference, reference),
            (0.25, 1.0, 4.0),
            ReferenceModel(),
            torch.device("cpu"),
            128,
            32,
        )
        np.testing.assert_allclose(output, reference, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
