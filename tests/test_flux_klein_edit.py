import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "poc"))

from test_flux_klein_edit import image_dimensions, load_input


class FluxKleinPreparationTests(unittest.TestCase):
    def test_landscape_image_is_reduced_for_gpu(self) -> None:
        self.assertEqual(image_dimensions(1504, 1002, 768), (768, 496))

    def test_small_input_is_not_enlarged(self) -> None:
        self.assertEqual(image_dimensions(640, 480, 768), (640, 480))

    def test_invalid_maximum_side_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            image_dimensions(1500, 1000, 750)

    def test_extremely_narrow_image_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "too small"):
            image_dimensions(2000, 200, 768)

    def test_input_is_converted_without_overwriting_original(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jpg"
            Image.new("RGB", (1504, 1002), (80, 120, 180)).save(path)
            original_bytes = path.read_bytes()

            image = load_input(path, 768)

            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (768, 496))
            self.assertEqual(path.read_bytes(), original_bytes)

    def test_missing_image_is_rejected(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_input(Path("/nonexistent/flux-test-input.png"), 768)


if __name__ == "__main__":
    unittest.main()
