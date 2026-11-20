import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from inspect_raw_sensor_v5 import channel_samples
from v5_core import (
    aligned_chunk,
    estimate_translation,
    fusion_chunk,
    read_manifest,
)


def determine_clip_code(
    sampled_values: np.ndarray,
    declared_white: float,
    global_white: float,
) -> float:
    above_white = sampled_values[sampled_values >= declared_white]

    if above_white.size < 20:
        raise ValueError("Too few upper sensor samples to validate the clipping plateau")

    codes, counts = np.unique(above_white, return_counts=True)
    dominant_index = int(np.argmax(counts))
    dominant_code = float(codes[dominant_index])
    dominance = float(counts[dominant_index] / above_white.size)

    if (
        dominance < 0.25
        or dominant_code < declared_white
        or dominant_code < global_white * 0.97
        or dominant_code > global_white * 1.02
    ):
        raise ValueError(
            "Camera clipping plateau is not sufficiently established; "
            "do not use raw-aware confidence without camera calibration"
        )

    return dominant_code


def calculate_sensor_confidence(
    values: np.ndarray,
    black_level: float,
    clip_code: float,
) -> np.ndarray:
    if not 0 <= black_level < clip_code:
        raise ValueError("Invalid RAW black level or clipping plateau")

    signal_range = clip_code - black_level
    normalized_signal = (
        values.astype(np.float32) - black_level
    ) / signal_range

    shadow_confidence = np.clip(
        (normalized_signal - 0.003) / 0.015,
        0.0,
        1.0,
    )
    highlight_confidence = np.clip(
        (1.0 - normalized_signal) / 0.02,
        0.0,
        1.0,
    )
    return shadow_confidence * highlight_confidence


def build_sensor_confidence(
    raw_path: Path,
    expected_dimensions: tuple[int, int],
    sampling_stride: int = 8,
) -> tuple[np.ndarray, dict]:
    import rawpy

    with rawpy.imread(str(raw_path)) as camera:
        mosaic = camera.raw_image_visible
        color_map = camera.raw_colors_visible
        pattern = camera.raw_pattern
        orientation = getattr(camera.sizes, "flip", 0)

        if (
            mosaic is None
            or color_map is None
            or pattern is None
            or tuple(pattern.shape) != (2, 2)
            or mosaic.shape != expected_dimensions
            or color_map.shape != expected_dimensions
            or mosaic.shape[0] % 2 != 0
            or mosaic.shape[1] % 2 != 0
            or orientation != 0
        ):
            raise ValueError(
                "V5.2 requires an unrotated Bayer RAW mosaic matching "
                "the developed RGB geometry. Do not reproject a CFA mask blindly."
            )

        black_levels = list(camera.black_level_per_channel)
        camera_white_levels = camera.camera_white_level_per_channel
        declared_levels = (
            list(camera_white_levels)
            if camera_white_levels is not None
            else [camera.white_level] * 4
        )

        if len(black_levels) < 4 or len(declared_levels) < 4:
            raise ValueError("Incomplete camera black/white calibration metadata")

        clip_levels = {}
        for row in range(2):
            for column in range(2):
                channel = int(color_map[row, column])

                if channel < 0 or channel > 3:
                    raise ValueError("Unsupported Bayer channel index")

                if channel in clip_levels:
                    continue

                sampled = channel_samples(
                    mosaic, color_map, channel, sampling_stride, 2
                )
                clip_levels[channel] = determine_clip_code(
                    sampled,
                    float(declared_levels[channel]),
                    float(camera.white_level),
                )

        block_confidence = np.ones(
            (mosaic.shape[0] // 2, mosaic.shape[1] // 2),
            dtype=np.float32,
        )

        for row in range(2):
            for column in range(2):
                channel = int(color_map[row, column])
                plane = mosaic[row::2, column::2]
                color_plane = color_map[row::2, column::2]

                if not np.all(color_plane == channel):
                    raise ValueError("RAW CFA channel phase is not spatially consistent")

                weight = calculate_sensor_confidence(
                    plane,
                    float(black_levels[channel]),
                    clip_levels[channel],
                )
                np.minimum(block_confidence, weight, out=block_confidence)

        # A máscara é suavizada na grade Bayer antes de ser projetada no RGB revelado.
        block_confidence = cv2.GaussianBlur(
            block_confidence, (3, 3), sigmaX=0.7
        )
        confidence = cv2.resize(
            block_confidence,
            (mosaic.shape[1], mosaic.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        report = {
            "raw_path": str(raw_path),
            "sensor_dimensions": list(mosaic.shape),
            "orientation": orientation,
            "clip_codes_by_cfa_index": {
                str(index): float(value)
                for index, value in sorted(clip_levels.items())
            },
            "declared_white_levels": [
                float(level) for level in declared_levels[:4]
            ],
            "global_white_level": float(camera.white_level),
            "near_zero_confidence_fraction": float(np.mean(confidence < 0.05)),
            "near_full_confidence_fraction": float(np.mean(confidence > 0.95)),
        }

    return confidence.astype(np.float32), report


def merge_raw_guided(
    triplet_path: Path,
    output_dir: Path,
    alignment: str = "translation",
    rows_per_chunk: int = 256,
) -> dict:
    if alignment not in ("translation", "none") or rows_per_chunk < 1:
        raise ValueError("Invalid alignment or rows-per-chunk option")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {output_dir}")

    images, factors, manifest = read_manifest(triplet_path)
    height, width = images[1].shape[:2]
    records = manifest["inputs"]

    masks = []
    sensor_reports = []

    output_dir.mkdir(parents=True, exist_ok=True)

    for index, record in enumerate(records):
        raw_path = Path(record["raw_path"])

        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)

        print(f"RAW confidence: {raw_path.name}", flush=True)
        mask, sensor_report = build_sensor_confidence(
            raw_path, (height, width)
        )
        mask_path = output_dir / f"{record['role']}_sensor_confidence.npy"
        np.save(mask_path, mask, allow_pickle=False)
        masks.append(np.load(mask_path, mmap_mode="r", allow_pickle=False))
        sensor_report["confidence_path"] = str(mask_path)
        sensor_reports.append(sensor_report)
        del mask

    shifts = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    scores = [None, 1.0, None]

    if alignment == "translation":
        for index in (0, 2):
            dx, dy, score = estimate_translation(images[1], images[index])
            shifts[index] = (dx, dy)
            scores[index] = score

    merged_path = output_dir / "merged_hdr.npy"
    merged = np.lib.format.open_memmap(
        merged_path,
        mode="w+",
        dtype=np.float32,
        shape=(height, width, 3),
    )

    low_confidence_pixels = 0

    for start in range(0, height, rows_per_chunk):
        end = min(start + rows_per_chunk, height)
        aligned_rgb = [
            aligned_chunk(image, start, end, shift)
            for image, shift in zip(images, shifts, strict=True)
        ]
        aligned_masks = [
            aligned_chunk(mask, start, end, shift)
            for mask, shift in zip(masks, shifts, strict=True)
        ]
        confidences = [
            np.clip(result[0], 0.0, 1.0) * result[1]
            for result in aligned_masks
        ]
        low_confidence_pixels += int(
            np.count_nonzero(np.maximum.reduce(confidences) < 0.05)
        )
        merged[start:end] = fusion_chunk(
            tuple(item[0] for item in aligned_rgb),
            tuple(factors),
            tuple(item[1] for item in aligned_rgb),
            tuple(confidences),
        )

    merged.flush()
    report = {
        "version": "v5.2-experimental",
        "output": str(merged_path),
        "source_manifest": str(triplet_path),
        "dimensions": [height, width],
        "source_alignment": alignment,
        "translations_xy": shifts,
        "registration_confidences": scores,
        "exposure_factors": factors,
        "low_sensor_confidence_across_all_exposures_fraction": (
            low_confidence_pixels / (height * width)
        ),
        "sensor_sources": sensor_reports,
        "limitations": [
            "This is sensor-guided fusion of LibRaw-developed linear RGB, not native Bayer fusion.",
            "Sensor confidence is a heuristic based on black level and repeated high-code plateaus.",
            "The RGB development stage may already have clipped some highlight color information.",
            "Registration corrects translations only; motion and fine leaves require separate review.",
            "When every exposure is unreliable, no real detail can be recovered.",
            "No trained neural image editor is used in this V5.2 experiment.",
        ],
    }
    (output_dir / "merge_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"V5.2 merged HDR: {merged_path}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experimental sensor-guided HDR fusion from three camera RAWs"
    )
    parser.add_argument("--triplet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--alignment", choices=("translation", "none"), default="translation"
    )
    parser.add_argument("--rows-per-chunk", type=int, default=256)
    args = parser.parse_args()

    try:
        merge_raw_guided(
            args.triplet,
            args.output_dir,
            args.alignment,
            args.rows_per_chunk,
        )
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
