import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "poc"))

from hdr_pipeline_v2 import (
    calibrate_neural,
    discover_brackets,
    protect_highlights,
    tone_map_local,
)


class HdrPipelineV2Tests(unittest.TestCase):
    def test_discover_sorts_unsorted_names_by_exif(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ("z.CR3", "a.CR3", "m.CR3"):
                (folder / name).touch()

            exposure_times = {"a.CR3": 1 / 40, "m.CR3": 1 / 320, "z.CR3": 1 / 5}

            def mocked_exif(command, **kwargs):
                records = [
                    {
                        "SourceFile": source,
                        "ExposureTime": exposure_times[Path(source).name],
                        "ISO": 100,
                        "FNumber": 4.0,
                    }
                    for source in command[6:]
                ]
                return type("Result", (), {"stdout": json.dumps(records)})()

            with patch("hdr_pipeline_v2.subprocess.run", side_effect=mocked_exif):
                with patch("hdr_diagnostics.subprocess.run", side_effect=mocked_exif):
                    paths, factors, _ = discover_brackets(folder)

            self.assertEqual([path.name for path in paths], ["m.CR3", "a.CR3", "z.CR3"])
            np.testing.assert_allclose(factors, (0.125, 1.0, 8.0))

    def test_discover_rejects_incomplete_brackets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "only.ARW").touch()
            with self.assertRaisesRegex(ValueError, "exactly 3"):
                discover_brackets(Path(directory))

    def test_calibration_matches_unmasked_midtone(self) -> None:
        baseline = np.full((12, 12, 3), 0.4, dtype=np.float32)
        neural = np.full_like(baseline, 0.1)
        mask = np.zeros((12, 12), dtype=np.float32)

        calibrated, factor = calibrate_neural(neural, baseline, mask)

        self.assertAlmostEqual(factor, 4.0)
        np.testing.assert_allclose(calibrated, baseline, atol=1e-6)

    def test_protect_highlights_uses_linear_source(self) -> None:
        neural = np.full((10, 10, 3), 0.2, dtype=np.float32)
        under = np.full_like(neural, 0.1)
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[2:8, 2:8] = 1.0

        result = protect_highlights(neural, under, 0.125, mask)

        self.assertAlmostEqual(float(result[0, 0, 0]), 0.2)
        self.assertAlmostEqual(float(result[5, 5, 0]), 0.8)

    def test_local_tone_mapping_protects_bright_region(self) -> None:
        hdr = np.full((20, 20, 3), 0.2, dtype=np.float32)
        mask = np.zeros((20, 20), dtype=np.float32)
        hdr[0:10, 0:10] = 4.0
        mask[0:10, 0:10] = 1.0

        result = tone_map_local(hdr, mask)

        self.assertEqual(result.dtype, np.uint8)
        self.assertLess(int(result[3, 3, 0]), 240)
        self.assertGreater(int(result[15, 15, 0]), 130)

    def test_tone_mapping_rejects_nonfinite(self) -> None:
        hdr = np.full((10, 10, 3), 0.1, dtype=np.float32)
        hdr[0, 0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "nonfinite"):
            tone_map_local(hdr, np.zeros((10, 10), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
