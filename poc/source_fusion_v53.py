import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tifffile import imwrite

from hdr_diagnostics import LUMINANCE_WEIGHTS
from v5_core import (
    SRGB_PROFILE,
    aligned_chunk,
    encode_srgb_16,
    estimate_translation,
    export_final,
    read_manifest,
)


def smoothstep(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if lower >= upper:
        raise ValueError("Invalid smoothstep bounds")

    scaled = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def source_select_chunk(
    frames: tuple[np.ndarray, np.ndarray, np.ndarray],
    factors: tuple[float, float, float],
    validity: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if (
        len(frames) != 3
        or len(validity) != 3
        or len({frame.shape for frame in frames}) != 1
        or frames[0].ndim != 3
        or frames[0].shape[2] != 3
    ):
        raise ValueError("Exactly three aligned RGB exposure frames are required")

    if not (0 < factors[0] < factors[1] < factors[2]):
        raise ValueError("Exposure factors are not in increasing order")

    if any(mask.shape != frames[0].shape[:2] for mask in validity):
        raise ValueError("Validity masks must match the frames")

    under, reference, over = [
        np.maximum(np.asarray(frame, dtype=np.float32), 0.0)
        for frame in frames
    ]
    under_peak = np.max(under, axis=2)
    reference_peak = np.max(reference, axis=2)
    over_peak = np.max(over, axis=2)

    highlight_source = (
        smoothstep(reference_peak, 0.55, 0.92)
        * smoothstep(under_peak, 0.018, 0.09)
        * (1.0 - smoothstep(under_peak, 0.92, 0.995))
        * validity[0]
    )
    shadow_source = (
        (1.0 - smoothstep(reference_peak, 0.06, 0.24))
        * smoothstep(over_peak, 0.035, 0.12)
        * (1.0 - smoothstep(over_peak, 0.87, 0.99))
        * validity[2]
    )

    under_radiance = under / factors[0]
    reference_radiance = reference / factors[1]
    over_radiance = over / factors[2]

    reference_luma = np.sum(reference_radiance * LUMINANCE_WEIGHTS, axis=2)
    reference_trusted = (reference_peak > 0.10) & (reference_peak < 0.82)

    for selected, candidate in (
        (highlight_source, under_radiance),
        (shadow_source, over_radiance),
    ):
        candidate_luma = np.sum(candidate * LUMINANCE_WEIGHTS, axis=2)
        mismatch = (
            np.abs(candidate_luma - reference_luma)
            / np.maximum(reference_luma, 0.04)
        )
        selected *= np.where(reference_trusted & (mismatch > 0.40), 0.0, 1.0)

    reference_source = np.maximum(0.0, 1.0 - highlight_source - shadow_source)
    reference_source *= validity[1]
    total = highlight_source + reference_source + shadow_source

    selected_hdr = (
        under_radiance * highlight_source[..., None]
        + reference_radiance * reference_source[..., None]
        + over_radiance * shadow_source[..., None]
    ) / np.maximum(total[..., None], 1e-6)

    fallback = np.where(
        validity[1][..., None],
        reference_radiance,
        under_radiance,
    )
    merged = np.where(total[..., None] > 1e-5, selected_hdr, fallback)
    selection = np.argmax(
        np.stack(
            (highlight_source, reference_source, shadow_source),
            axis=2,
        ),
        axis=2,
    ).astype(np.uint8)

    return np.maximum(merged, 0.0).astype(np.float32), selection


def guided_filter(
    guide: np.ndarray,
    source: np.ndarray,
    radius: int,
    regularization: float,
) -> np.ndarray:
    if guide.shape != source.shape or radius < 1 or regularization <= 0:
        raise ValueError("Guided filtering inputs are invalid")

    window = (2 * radius + 1, 2 * radius + 1)
    guide = guide.astype(np.float32)
    source = source.astype(np.float32)

    mean_guide = cv2.boxFilter(guide, -1, window)
    mean_source = cv2.boxFilter(source, -1, window)
    variance = cv2.boxFilter(guide * guide, -1, window)
    variance -= mean_guide * mean_guide
    covariance = cv2.boxFilter(guide * source, -1, window)
    covariance -= mean_guide * mean_source

    gain = covariance / (variance + regularization)
    bias = mean_source - gain * mean_guide
    smooth_gain = cv2.boxFilter(gain, -1, window)
    smooth_bias = cv2.boxFilter(bias, -1, window)
    return smooth_gain * guide + smooth_bias


def detect_exterior_mask(
    under: np.ndarray,
    reference: np.ndarray,
    minimum_fraction: float = 0.0015,
) -> np.ndarray:
    if under.shape != reference.shape or under.ndim != 3 or under.shape[2] != 3:
        raise ValueError("Exterior detection needs equally sized RGB exposure frames")

    reference_luma = np.sum(reference * LUMINANCE_WEIGHTS, axis=2)
    under_luma = np.sum(under * LUMINANCE_WEIGHTS, axis=2)

    bright = (
        (reference_luma > 0.56)
        & (np.max(reference, axis=2) > 0.72)
        & (under_luma > 0.10)
    ).astype(np.uint8)

    closed = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), dtype=np.uint8),
    )
    components, indices, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )
    minimum_pixels = max(32, int(bright.size * minimum_fraction))
    selected = np.zeros(bright.shape, dtype=np.uint8)

    for index in range(1, components):
        if int(stats[index, cv2.CC_STAT_AREA]) < minimum_pixels:
            continue

        selected[indices == index] = 255

    smoothed = guided_filter(
        np.clip(reference_luma, 0.0, 1.0),
        selected.astype(np.float32) / 255.0,
        radius=3,
        regularization=0.004,
    )
    return np.clip(smoothed, 0.0, 1.0).astype(np.float32)


def load_window_mask(
    mask_path: Path,
    dimensions: tuple[int, int],
) -> np.ndarray:
    with Image.open(mask_path) as original:
        mask = original.convert("L")
        if mask.size != (dimensions[1], dimensions[0]):
            raise ValueError("Window mask must have the camera preview dimensions")
        return np.asarray(mask, dtype=np.float32) / 255.0


def scene_tone_parameters(
    hdr: np.ndarray,
    mask: np.ndarray,
    target_mid: float,
    window_target: float,
    base_contrast: float,
    detail_contrast: float,
) -> dict:
    if not (0 < target_mid < 1 and 0 < window_target < 1):
        raise ValueError("Targets must be strictly between zero and one")

    height, width = hdr.shape[:2]
    preview_size = (max(64, round(width * min(1.0, 1024 / max(width, height)))),
                    max(64, round(height * min(1.0, 1024 / max(width, height)))))
    small_hdr = cv2.resize(hdr, preview_size, interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask, preview_size, interpolation=cv2.INTER_AREA)
    small_luma = np.sum(small_hdr * LUMINANCE_WEIGHTS, axis=2)
    valid_interior = (small_mask < 0.25) & (small_luma > 0.002)

    if np.count_nonzero(valid_interior) < 64:
        raise ValueError("Insufficient interior pixels for HDR tone normalization")

    interior_median = max(float(np.median(small_luma[valid_interior])), 1e-5)
    log_luma = np.log(np.maximum(small_luma, 1e-5))
    base = cv2.bilateralFilter(
        log_luma.astype(np.float32),
        d=0,
        sigmaColor=0.52,
        sigmaSpace=12.0,
    )
    mid_logit = math.log(target_mid / (1.0 - target_mid))
    logits = (
        mid_logit
        + base_contrast * (base - math.log(interior_median))
        + detail_contrast * (log_luma - base)
    )

    exterior = small_mask > 0.70
    window_offset = 0.0

    if np.count_nonzero(exterior) >= 64:
        window_logit = math.log(window_target / (1.0 - window_target))
        window_offset = float(
            np.clip(window_logit - np.median(logits[exterior]), -3.0, 0.0)
        )

    return {
        "window_offset": window_offset,
        "interior_median": interior_median,
        "base_preview": base,
        "base_contrast": base_contrast,
        "detail_contrast": detail_contrast,
        "mid_logit": mid_logit,
    }


def tone_map_chunk(
    hdr: np.ndarray,
    base_log: np.ndarray,
    window_mask: np.ndarray,
    parameters: dict,
    shadow_depth: float,
) -> np.ndarray:
    if hdr.shape[:2] != base_log.shape or base_log.shape != window_mask.shape:
        raise ValueError("Tone-map inputs must have the same geometry")

    luma = np.sum(np.maximum(hdr, 0.0) * LUMINANCE_WEIGHTS, axis=2)
    log_luma = np.log(np.maximum(luma, 1e-5))

    logits = (
        parameters["mid_logit"]
        + parameters["base_contrast"] *
        (base_log - math.log(parameters["interior_median"]))
        + parameters["detail_contrast"] * (log_luma - base_log)
        + parameters["window_offset"] * window_mask
    )

    initial_luma = 1.0 / (1.0 + np.exp(-np.clip(logits, -16.0, 16.0)))
    shadow_fraction = np.clip((0.36 - initial_luma) / 0.36, 0.0, 1.0)
    logits -= shadow_depth * shadow_fraction * shadow_fraction
    output_luma = 1.0 / (1.0 + np.exp(-np.clip(logits, -16.0, 16.0)))

    rendered = np.maximum(hdr, 0.0) * (
        output_luma / np.maximum(luma, 1e-6)
    )[..., None]
    rendered /= np.maximum(
        1.0,
        np.max(rendered, axis=2, keepdims=True),
    )
    return np.clip(rendered, 0.0, 1.0).astype(np.float32)


def merge_original_brackets(
    manifest_path: Path,
    destination: Path,
    rows_per_chunk: int = 256,
) -> dict:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Output directory must be empty for a reproducible test")

    frames, exposure_factors, manifest = read_manifest(manifest_path)
    height, width = frames[1].shape[:2]
    shifts = [(0.0, 0.0)] * 3
    confidence = [None, 1.0, None]

    for index in (0, 2):
        dx, dy, score = estimate_translation(frames[1], frames[index])
        shifts[index] = (dx, dy)
        confidence[index] = score

    destination.mkdir(parents=True, exist_ok=True)
    hdr_path = destination / "merged_hdr.npy"
    hdr = np.lib.format.open_memmap(
        hdr_path,
        mode="w+",
        dtype=np.float32,
        shape=(height, width, 3),
    )
    selection = np.lib.format.open_memmap(
        destination / "source_selection.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(height, width),
    )
    counts = np.zeros(3, dtype=np.int64)

    for start in range(0, height, rows_per_chunk):
        end = min(start + rows_per_chunk, height)
        aligned = [
            aligned_chunk(frame, start, end, shift)
            for frame, shift in zip(frames, shifts, strict=True)
        ]
        merged, labels = source_select_chunk(
            tuple(item[0] for item in aligned),
            tuple(exposure_factors),
            tuple(item[1] for item in aligned),
        )
        hdr[start:end] = merged
        selection[start:end] = labels
        counts += np.bincount(labels.ravel(), minlength=3)

    hdr.flush()
    selection.flush()

    report = {
        "method": "direct source-selective radiance fusion of three RAW-developed linear frames",
        "original_raw_files": [record["raw_path"] for record in manifest["inputs"]],
        "source_exposure_factors": exposure_factors,
        "translations_xy": shifts,
        "registration_confidence": confidence,
        "source_selection_fraction": {
            role: float(count / (height * width))
            for role, count in zip(("under", "reference", "over"), counts, strict=True)
        },
        "merged_hdr": str(hdr_path),
        "source_selection": str(destination / "source_selection.npy"),
    }
    (destination / "fusion_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def render_scene(
    manifest_path: Path,
    merged_path: Path,
    output_dir: Path,
    target_mid: float,
    window_target: float,
    base_contrast: float,
    detail_contrast: float,
    shadow_depth: float,
    optional_window_mask: Path | None = None,
    rows_per_chunk: int = 256,
) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Grade output folder must be empty")

    if not (0 <= shadow_depth <= 1.2):
        raise ValueError("Shadow depth must be within [0, 1.2]")

    frames, _, _ = read_manifest(manifest_path)
    hdr = np.load(merged_path, mmap_mode="r", allow_pickle=False)
    height, width = hdr.shape[:2]
    preview_scale = min(1.0, 1024.0 / max(height, width))
    preview_size = (
        max(64, round(width * preview_scale)),
        max(64, round(height * preview_scale)),
    )

    under = cv2.resize(frames[0], preview_size, interpolation=cv2.INTER_AREA)
    reference = cv2.resize(frames[1], preview_size, interpolation=cv2.INTER_AREA)

    if optional_window_mask is None:
        small_mask = detect_exterior_mask(under, reference)
    else:
        small_mask = load_window_mask(
            optional_window_mask,
            (preview_size[1], preview_size[0]),
        )

    full_mask = cv2.resize(
        small_mask, (width, height), interpolation=cv2.INTER_LINEAR
    )
    parameters = scene_tone_parameters(
        hdr, full_mask, target_mid, window_target,
        base_contrast, detail_contrast,
    )
    base_full = cv2.resize(
        parameters["base_preview"],
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / "exterior_mask_preview.png"
    Image.fromarray(
        np.uint8(np.round(small_mask * 255.0))
    ).save(mask_path)

    mapped = np.empty((height, width, 3), dtype=np.uint16)

    for start in range(0, height, rows_per_chunk):
        end = min(start + rows_per_chunk, height)
        rendered = tone_map_chunk(
            np.asarray(hdr[start:end]),
            base_full[start:end],
            full_mask[start:end],
            parameters,
            shadow_depth,
        )
        mapped[start:end] = encode_srgb_16(rendered)

    tiff_path = output_dir / "tone_mapped_16bit.tiff"
    jpeg_path = output_dir / "tone_mapped_preview.jpg"

    imwrite(
        tiff_path,
        mapped,
        compression="deflate",
        photometric="rgb",
        metadata=None,
        extratags=((34675, "B", len(SRGB_PROFILE), SRGB_PROFILE, False),),
    )
    Image.fromarray(
        np.uint8((mapped.astype(np.uint32) * 255 + 32767) // 65535)
    ).save(
        jpeg_path,
        format="JPEG",
        quality=98,
        subsampling=0,
        optimize=False,
        icc_profile=SRGB_PROFILE,
    )

    report = {
        "input": str(merged_path),
        "camera_resolution": [width, height],
        "exterior_mask_fraction": float(np.mean(small_mask > 0.5)),
        "exterior_mask_source": str(optional_window_mask) if optional_window_mask else "auto",
        "exterior_mask_preview": str(mask_path),
        "window_exposure_offset_ev": parameters["window_offset"] / math.log(2),
        "target_mid": target_mid,
        "window_target": window_target,
        "base_contrast": base_contrast,
        "detail_contrast": detail_contrast,
        "shadow_depth": shadow_depth,
        "tiff": str(tiff_path),
        "jpeg": str(jpeg_path),
        "warnings": [
            "Window detection is heuristic, not semantic segmentation.",
            "Auto mask may misclassify bright walls or miss shaded exterior details.",
            "White-clipped source pixels cannot be recovered or hallucinated.",
            "Visual comparison at full source resolution is mandatory.",
        ],
    }
    (output_dir / "tone_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Source-selective three-RAW HDR fusion with edge-aware photographic contrast"
    )
    parser.add_argument("--triplet", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-mid", default=0.47, type=float)
    parser.add_argument("--window-target", default=0.56, type=float)
    parser.add_argument("--base-contrast", default=1.08, type=float)
    parser.add_argument("--detail-contrast", default=1.40, type=float)
    parser.add_argument("--shadow-depth", default=0.30, type=float)
    parser.add_argument("--window-mask", type=Path)
    parser.add_argument("--rows-per-chunk", default=256, type=int)
    args = parser.parse_args()

    try:
        if args.rows_per_chunk < 1:
            raise ValueError("Chunk height must be positive")

        if not (0.1 <= args.base_contrast <= 2.0 and 0.1 <= args.detail_contrast <= 2.0):
            raise ValueError("Contrast factors must be within [0.1, 2.0]")

        merge = merge_original_brackets(
            args.triplet, args.output_dir, args.rows_per_chunk
        )
        render = render_scene(
            args.triplet,
            Path(merge["merged_hdr"]),
            args.output_dir / "grade",
            args.target_mid,
            args.window_target,
            args.base_contrast,
            args.detail_contrast,
            args.shadow_depth,
            args.window_mask,
            args.rows_per_chunk,
        )
        final = export_final(
            Path(merge["merged_hdr"]),
            Path(render["tiff"]),
            args.output_dir / "final",
        )
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Final JPEG: {final['jpeg']}")
    print(f"Final 16-bit TIFF: {final['tiff']}")
    print(f"Window segmentation mask: {render['exterior_mask_preview']}")
    print(f"Exterior coverage: {render['exterior_mask_fraction']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
