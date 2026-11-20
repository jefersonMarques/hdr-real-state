import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from hdr_diagnostics import exposure_value


RAW_EXTENSIONS = {
    ".arw", ".cr2", ".cr3", ".dng", ".nef",
    ".orf", ".raf", ".rw2", ".srw",
}
BRACKET_LABELS = ("under", "reference", "over")


def discover_triplet(directory: Path) -> tuple[list[Path], list[float], list[dict]]:
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in RAW_EXTENSIONS
    )

    if len(paths) != 3:
        raise ValueError(f"Expected exactly 3 RAW files, found {len(paths)}")

    command = [
        "exiftool",
        "-json",
        "-n",
        "-ExposureTime",
        "-ISO",
        "-FNumber",
        *(str(path) for path in paths),
    ]
    response = subprocess.run(command, capture_output=True, text=True, check=True)
    records = json.loads(response.stdout)

    if not isinstance(records, list) or len(records) != 3:
        raise ValueError("EXIF metadata must be present for all three RAW files")

    try:
        exposures = [exposure_value(record) for record in records]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Missing or invalid exposure metadata") from error

    ordered = sorted(zip(exposures, paths, records), key=lambda item: item[0])
    source_paths = [item[1] for item in ordered]
    sorted_exposures = [item[0] for item in ordered]
    sorted_metadata = [item[2] for item in ordered]

    if not sorted_exposures[0] < sorted_exposures[1] < sorted_exposures[2]:
        raise ValueError("Three distinct camera exposure levels are required")

    factors = [value / sorted_exposures[1] for value in sorted_exposures]

    if factors[0] > 0.85 or factors[2] < 1.15:
        raise ValueError("Exposure bracket spacing is too small")

    return source_paths, factors, sorted_metadata


def calculate_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")

    if max_side < 256 or max_side % 16 != 0:
        raise ValueError("Maximum side must be at least 256 and divisible by 16")

    scale = min(1.0, max_side / max(width, height))
    result_width = max(16, math.floor(width * scale / 16) * 16)
    result_height = max(16, math.floor(height * scale / 16) * 16)

    if min(result_width, result_height) < 256:
        raise ValueError("The input aspect ratio is too extreme for this experiment")

    return result_width, result_height


def camera_white_balance(reference_path: Path) -> tuple[float, float, float, float]:
    import rawpy

    with rawpy.imread(str(reference_path)) as camera:
        gains = list(camera.camera_whitebalance)
        fallback = list(camera.daylight_whitebalance)

    if len(gains) != 4 or len(fallback) != 4:
        raise ValueError("Camera white balance must have four components")

    gains = [
        float(gain) if np.isfinite(gain) and gain > 0 else float(fallback[index])
        for index, gain in enumerate(gains)
    ]

    if gains[3] <= 0:
        gains[3] = gains[1]

    if not all(np.isfinite(gain) and gain > 0 for gain in gains):
        raise ValueError("Invalid reference RAW white balance coefficients")

    return tuple(gains)


def develop_raw(path: Path, white_balance: tuple[float, ...]) -> Image.Image:
    import rawpy

    with rawpy.imread(str(path)) as camera:
        developed = camera.postprocess(
            user_wb=white_balance,
            use_camera_wb=False,
            no_auto_bright=True,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=16,
            gamma=(2.222, 4.5),
        )

    if developed.ndim != 3 or developed.shape[2] != 3:
        raise ValueError(f"Expected RGB camera data: {path}")

    display_pixels = (developed // 257).astype(np.uint8)
    return Image.fromarray(display_pixels, mode="RGB")


def prepare_triplet(photo_directory: Path, output_directory: Path, max_side: int) -> dict:
    paths, factors, records = discover_triplet(photo_directory)

    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_directory}")

    white_balance = camera_white_balance(paths[1])
    original_dimensions = None
    prepared_dimensions = None
    prepared = []

    for label, path, factor, record in zip(
        BRACKET_LABELS, paths, factors, records, strict=True
    ):
        image = develop_raw(path, white_balance)
        source_size = image.size

        if original_dimensions is not None and source_size != original_dimensions:
            raise ValueError("The RAW images do not have matching developed dimensions")

        original_dimensions = source_size
        target_size = calculate_size(*source_size, max_side)

        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)

        prepared_dimensions = image.size
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = output_directory / f"{label}.png"
        image.save(output_path)
        print(f"{label}: {path.name} | relative exposure: {factor:.3f}", flush=True)

        prepared.append({
            "role": label,
            "raw_path": str(path),
            "prepared_path": str(output_path),
            "relative_exposure": factor,
            "exif": record,
        })

    report = {
        "input_type": "three independent camera RAW files",
        "preprocessing": "camera RAW to sRGB display reference; no HDR merge",
        "original_dimensions": list(original_dimensions),
        "prepared_dimensions": list(prepared_dimensions),
        "white_balance_source": str(paths[1]),
        "shared_camera_white_balance": list(white_balance),
        "inputs": prepared,
        "limitations": [
            "No RAW exposure alignment or moving-object correction.",
            "Model references are 8-bit display images, not sensor-linear RAW tensors.",
            "Downsampled references do not retain original RAW detail or tonal precision.",
            "FLUX multi-reference conditioning is not a physically calibrated HDR fusion network.",
        ],
    }

    (output_directory / "triplet.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare three exposure-bracketed RAW references without HDR merging"
    )
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=768)
    args = parser.parse_args()

    try:
        prepare_triplet(args.photos, args.output_dir, args.max_side)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
