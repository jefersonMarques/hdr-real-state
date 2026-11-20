import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tifffile import imwrite

from hdr_afunet import load_afunet, load_raw, reconstruct
from hdr_diagnostics import (
    accumulate_exposure,
    encode_srgb,
    load_linear_raw,
    read_exposure_factors,
)
from hdr_window_recovery import create_highlight_mask


RAW_EXTENSIONS = {
    ".arw", ".cr2", ".cr3", ".dng", ".nef", ".orf",
    ".raf", ".rw2", ".srw",
}
LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def discover_brackets(directory: Path) -> tuple[tuple[Path, Path, Path], tuple[float, float, float], list[dict]]:
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    paths = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in RAW_EXTENSIONS
    )

    if len(paths) != 3:
        raise ValueError(f"Expected exactly 3 RAW files in {directory}, found {len(paths)}")

    provisional_factors, records = read_exposure_factors((paths[0], paths[1], paths[2]))
    ranked = sorted(zip(provisional_factors, paths, records), key=lambda item: item[0])
    ordered_paths = tuple(item[1] for item in ranked)
    ordered_records = [item[2] for item in ranked]
    factors, _ = read_exposure_factors(ordered_paths)

    if factors[0] > 0.85 or factors[2] < 1.15:
        raise ValueError("Exposure spacing is too small for a reliable HDR bracket")

    return ordered_paths, factors, ordered_records


def luminance(image: np.ndarray) -> np.ndarray:
    return np.sum(image * LUMINANCE_WEIGHTS, axis=2)


def calibrate_neural(
    neural: np.ndarray,
    baseline: np.ndarray,
    highlight_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    if neural.shape != baseline.shape or neural.shape[:2] != highlight_mask.shape:
        raise ValueError("Neural, baseline and highlight mask dimensions must match")

    baseline_luma = luminance(baseline)
    neural_luma = luminance(neural)
    usable = (
        (highlight_mask < 0.05)
        & (baseline_luma > 0.02)
        & (baseline_luma < 1.0)
        & (neural_luma > 0.0001)
    )

    if int(np.count_nonzero(usable)) < 64:
        raise ValueError("Not enough reliable midtones to calibrate the neural output")

    baseline_median = float(np.median(baseline_luma[usable]))
    neural_median = float(np.median(neural_luma[usable]))
    factor = float(np.clip(baseline_median / neural_median, 0.05, 30.0))
    return neural * factor, factor


def protect_highlights(
    hdr: np.ndarray,
    under_linear: np.ndarray,
    under_factor: float,
    mask: np.ndarray,
) -> np.ndarray:
    if hdr.shape != under_linear.shape or hdr.shape[:2] != mask.shape:
        raise ValueError("Incompatible HDR and source image dimensions")
    if under_factor <= 0:
        raise ValueError("Exposure factor must be positive")

    source_radiance = under_linear / under_factor
    result = hdr * (1.0 - mask[..., None])
    result += source_radiance * mask[..., None]
    return np.maximum(result, 0.0)


def tone_map_local(hdr: np.ndarray, highlight_mask: np.ndarray) -> np.ndarray:
    if hdr.ndim != 3 or hdr.shape[2] != 3 or hdr.shape[:2] != highlight_mask.shape:
        raise ValueError("HDR and mask dimensions must match")
    if not np.isfinite(hdr).all():
        raise ValueError("HDR contains nonfinite pixels")

    positive = np.maximum(hdr, 0.0).astype(np.float32)
    scene_luma = luminance(positive)
    interior = scene_luma[(highlight_mask < 0.10) & (scene_luma > 1e-6)]
    windows = scene_luma[(highlight_mask > 0.90) & (scene_luma > 1e-6)]

    if interior.size < 64:
        raise ValueError("Not enough interior pixels to compute tone mapping")

    interior_median = max(float(np.median(interior)), 1e-6)
    interior_scale = 0.42 / (1.0 - 0.42) / interior_median

    if windows.size >= 64:
        window_median = max(float(np.median(windows)), 1e-6)
        window_scale = 0.40 / (1.0 - 0.40) / window_median
        window_scale = min(window_scale, interior_scale)
    else:
        window_scale = interior_scale

    interior_scale = float(np.clip(interior_scale, 0.01, 100.0))
    window_scale = float(np.clip(window_scale, 0.001, 100.0))
    exposure_map = np.exp(
        np.log(interior_scale) * (1.0 - highlight_mask)
        + np.log(window_scale) * highlight_mask
    ).astype(np.float32)

    mapped_luma = scene_luma * exposure_map
    compressed_luma = mapped_luma / (1.0 + mapped_luma)
    compression = compressed_luma / np.maximum(scene_luma, 1e-6)
    linear_preview = positive * compression[..., None]

    return encode_srgb(linear_preview)


def save_preview(pixels: np.ndarray, path: Path) -> None:
    Image.fromarray(pixels).save(path, quality=95)


def generate_baseline(
    paths: tuple[Path, Path, Path],
    factors: tuple[float, float, float],
    output_directory: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    accumulated = None
    weights = None
    under_linear = None
    under_preview = None
    reference_preview = None

    for index, (name, path, factor) in enumerate(
        zip(("under", "reference", "over"), paths, factors, strict=True)
    ):
        print(f"Developing {name}: {path.name}", flush=True)
        image = load_linear_raw(path)

        if accumulated is None:
            accumulated = np.zeros_like(image, dtype=np.float32)
            weights = np.zeros(image.shape[:2], dtype=np.float32)

        if image.shape != accumulated.shape:
            raise ValueError("RAW files have different image dimensions")

        if index in (0, 1):
            preview = encode_srgb(image)
            save_preview(preview, output_directory / f"raw_{name}.jpg")

            if index == 0:
                under_linear = image.copy()
                under_preview = preview
            else:
                reference_preview = preview

        accumulate_exposure(image, factor, accumulated, weights, fallback=index == 0)
        del image

    baseline = accumulated / np.maximum(weights[..., None], 1e-8)
    highlight_mask = create_highlight_mask(
        under_preview.astype(np.float32) / 255.0,
        reference_preview.astype(np.float32) / 255.0,
    )

    del accumulated, weights, under_preview, reference_preview
    return baseline, under_linear, highlight_mask


def generate_neural(
    paths: tuple[Path, Path, Path],
    factors: tuple[float, float, float],
    source: Path,
    checkpoint: Path,
    tile_size: int,
    overlap: int,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running AFUNet on: {device}", flush=True)
    images = tuple(load_raw(path) for path in paths)

    if len({image.shape for image in images}) != 1:
        raise ValueError("RAW dimensions differ")

    model = load_afunet(source, checkpoint, device)
    result = reconstruct(
        images,
        factors,
        model,
        device,
        tile_size,
        overlap,
    )
    del images, model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def run_pipeline(
    photo_directory: Path,
    output_directory: Path,
    afunet_source: Path,
    checkpoint: Path,
    tile_size: int,
    overlap: int,
) -> None:
    paths, factors, metadata = discover_brackets(photo_directory)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {output_directory}")

    output_directory.mkdir(parents=True, exist_ok=True)
    for label, path, factor in zip(("under", "reference", "over"), paths, factors, strict=True):
        print(f"{label}: {path.name} (relative exposure {factor:.4f})", flush=True)

    baseline, under_linear, mask = generate_baseline(paths, factors, output_directory)
    imwrite(output_directory / "conventional_hdr.tiff", baseline, photometric="rgb")
    save_preview(tone_map_local(baseline, mask), output_directory / "conventional_local.jpg")
    Image.fromarray(np.uint8(np.round(mask * 255))).save(output_directory / "highlight_mask.png")

    neural = generate_neural(paths, factors, afunet_source, checkpoint, tile_size, overlap)
    imwrite(output_directory / "neural_original_hdr.tiff", neural, photometric="rgb")
    calibrated, factor = calibrate_neural(neural, baseline, mask)
    protected = protect_highlights(calibrated, under_linear, factors[0], mask)
    save_preview(tone_map_local(calibrated, mask), output_directory / "neural_local.jpg")
    save_preview(tone_map_local(protected, mask), output_directory / "hybrid_preview.jpg")
    imwrite(output_directory / "hybrid_hdr.tiff", protected, photometric="rgb")

    report = {
        "inputs": [str(path) for path in paths],
        "exposure_factors": list(factors),
        "exif": metadata,
        "cuda_used": torch.cuda.is_available(),
        "neural_calibration_factor": factor,
        "highlight_mask_fraction": round(float(np.mean(mask > 0.5)), 4),
        "limitations": [
            "Uncalibrated neural HDR radiance scale",
            "No geometric registration or deghosting",
            "Single-scene experimental highlight mask",
            "Local tone mapping is experimental, not production color management",
        ],
    }
    (output_directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Final preview: {output_directory / 'hybrid_preview.jpg'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Experimental linear HDR and AFUNet hybrid pipeline")
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--afunet-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=32)
    args = parser.parse_args()

    try:
        run_pipeline(
            args.photos,
            args.output_dir,
            args.afunet_source,
            args.checkpoint,
            args.tile_size,
            args.overlap,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
