import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "poc"))

from prepare_editor_input import calculate_size, prepare_image


class PrepareEditorInputTests(unittest.TestCase):
    def test_resizes_landscape_image_preserving_aspect_ratio(self) -> None:
        self.assertEqual(calculate_size(6000, 4000, 1024), (1024, 683))

    def test_resizes_portrait_image_preserving_aspect_ratio(self) -> None:
        self.assertEqual(calculate_size(4000, 6000, 1024), (683, 1024))

    def test_keeps_small_source_resolution(self) -> None:
        self.assertEqual(calculate_size(640, 480, 1024), (640, 480))

    def test_rejects_undersized_deeplpf_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 32 pixels"):
            calculate_size(4000, 50, 1024)

    def test_creates_new_rgb_png_without_changing_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jpg"
            output = root / "prepared" / "input.png"
            Image.new("RGB", (900, 600), (140, 120, 100)).save(source)
            original_bytes = source.read_bytes()

            dimensions = prepare_image(source, output, 300)

            self.assertEqual(dimensions, (300, 200))
            with Image.open(output) as result:
                self.assertEqual(result.size, (300, 200))
                self.assertEqual(result.mode, "RGB")
            self.assertEqual(source.read_bytes(), original_bytes)

    def test_rejects_overwriting_prepared_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.jpg"
            output = Path(directory) / "output.png"
            Image.new("RGB", (100, 100)).save(source)
            Image.new("RGB", (20, 20)).save(output)

            with self.assertRaises(FileExistsError):
                prepare_image(source, output, 1024)


if __name__ == "__main__":
    unittest.main()
