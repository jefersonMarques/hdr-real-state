import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from prepare_triplet_v4 import camera_white_balance, discover_triplet


def sensor_channel_statistics(
    values: np.ndarray,
    black_level: float,
    white_level: float,
) -> dict:
    if not 0 <= black_level < white_level:
        raise ValueError("Sensor black and white levels are invalid")

    if values.size == 0:
        raise ValueError("Cannot analyze an empty sensor color channel")

    normalized = (values.astype(np.float32) - black_level) / (
        white_level - black_level
    )
    above_white = values[values >= white_level]
    sampled_max = int(np.max(values))

    dominant_high_code = None
    dominant_high_fraction = 0.0

    if above_white.size > 0:
        unique_codes, code_counts = np.unique(above_white, return_counts=True)
        dominant_index = int(np.argmax(code_counts))
        dominant_high_code = int(unique_codes[dominant_index])
        dominant_high_fraction = float(code_counts[dominant_index] / values.size)

    return {
        "sample_count": int(normalized.size),
        "black_level": float(black_level),
        "white_level": float(white_level),
        "signal_fraction_p01": float(np.percentile(normalized, 1)),
        "signal_fraction_p50": float(np.percentile(normalized, 50)),
        "signal_fraction_p99": float(np.percentile(normalized, 99)),
        "near_black_fraction": float(np.mean(normalized <= 0.005)),
        "near_white_fraction": float(np.mean(normalized >= 0.995)),
        "sensor_saturated_fraction": float(np.mean(normalized >= 1.0)),
        "above_declared_white_fraction": float(np.mean(values >= white_level)),
        "strictly_above_declared_white_fraction": float(np.mean(values > white_level)),
        "sampled_raw_p99": float(np.percentile(values, 99)),
        "sampled_raw_max": sampled_max,
        "sampled_max_code_fraction": float(np.mean(values == sampled_max)),
        "dominant_high_code": dominant_high_code,
        "dominant_high_code_sample_fraction": dominant_high_fraction,
    }


def validate_sensor_levels(
    black_levels: list,
    camera_white_levels: list | None,
    fallback_white_level: int,
) -> tuple[list[float], list[float]]:
    if len(black_levels) < 4:
        raise ValueError("Four sensor black level entries are required")

    if camera_white_levels is None or len(camera_white_levels) < 4:
        whites = [float(fallback_white_level)] * 4
    else:
        whites = [
            float(level) if level is not None and float(level) > 0
            else float(fallback_white_level)
            for level in camera_white_levels[:4]
        ]

    blacks = [float(level) for level in black_levels[:4]]

    if not all(
        math.isfinite(black) and math.isfinite(white) and 0 <= black < white
        for black, white in zip(blacks, whites, strict=True)
    ):
        raise ValueError("Invalid sensor level metadata")

    return blacks, whites


def channel_samples(
    mosaic: np.ndarray,
    color_map: np.ndarray,
    channel: int,
    sample_stride: int,
    cfa_period: int,
) -> np.ndarray:
    if mosaic.shape != color_map.shape:
        raise ValueError("Sensor mosaic and CFA color map do not match")

    if sample_stride < 1 or cfa_period < 1:
        raise ValueError("Sensor sampling parameters must be positive")

    stride = max(cfa_period, sample_stride)
    stride = math.ceil(stride / cfa_period) * cfa_period
    collected = []

    for row in range(cfa_period):
        for column in range(cfa_period):
            sampled_colors = color_map[row::stride, column::stride]
            selected = sampled_colors == channel

            if not np.any(selected):
                continue

            sampled_values = mosaic[row::stride, column::stride]
            collected.append(np.asarray(sampled_values[selected]))

    if not collected:
        raise ValueError(f"No sensor samples for color channel {channel}")

    return np.concatenate(collected)


def inspect_raw_sensor(path: Path, sample_stride: int = 8) -> dict:
    import rawpy

    with rawpy.imread(str(path)) as raw:
        mosaic = raw.raw_image_visible
        color_map = raw.raw_colors_visible

        if mosaic is None or color_map is None or mosaic.ndim != 2:
            raise ValueError("Camera RAW does not expose a supported CFA mosaic")

        blacks, whites = validate_sensor_levels(
            list(raw.black_level_per_channel),
            raw.camera_white_level_per_channel,
            raw.white_level,
        )
        pattern = raw.raw_pattern
        period = int(max(pattern.shape)) if pattern is not None else 2

        indices = sorted(int(value) for value in np.unique(color_map))

        if any(index < 0 or index > 3 for index in indices):
            raise ValueError("Unsupported RAW sensor color index")

        statistics = {}

        for index in indices:
            values = channel_samples(
                mosaic,
                color_map,
                index,
                sample_stride,
                period,
            )
            statistics[f"channel_{index}"] = sensor_channel_statistics(
                values, blacks[index], whites[index]
            )

        return {
            "file": str(path),
            "visible_mosaic_dimensions": list(mosaic.shape),
            "cfa_pattern_size": list(pattern.shape) if pattern is not None else None,
            "cfa_color_description": str(raw.color_desc),
            "black_levels": blacks,
            "white_levels": whites,
            "libraw_global_white_level": int(raw.white_level),
            "sampling_stride": sample_stride,
            "channels": statistics,
        }


def inspect_triplet(photos: Path, output_dir: Path, sample_stride: int) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {output_dir}")

    paths, exposure_factors, exif_records = discover_triplet(photos)
    camera_wb = camera_white_balance(paths[1])
    results = []

    for role, path, factor, exif in zip(
        ("under", "reference", "over"),
        paths,
        exposure_factors,
        exif_records,
        strict=True,
    ):
        print(f"Inspecting sensor: {role} - {path.name}", flush=True)
        inspected = inspect_raw_sensor(path, sample_stride)
        results.append({
            **inspected,
            "role": role,
            "relative_exposure": float(factor),
            "exif": exif,
        })

    if len({tuple(record["visible_mosaic_dimensions"]) for record in results}) != 1:
        raise ValueError("RAW sensor dimensions differ across exposures")

    report = {
        "source": "RawPy sensor mosaic before demosaicing or RGB development",
        "shared_reference_camera_wb": list(camera_wb),
        "inputs": results,
        "interpretation": [
            "Near-white and near-black fractions are measured relative to the camera-declared white and black levels.",
            "The legacy sensor_saturated_fraction denotes samples at or above declared white, not verified physical sensor saturation.",
            "The dominant high code and sampled maximum can reveal a plateau beyond the declared camera white level; they do not establish the true physical clip threshold.",
            "These global percentages are not yet spatial clipping masks or evidence of HDR fidelity.",
            "Clipped highlights in the middle RAW may still be recoverable from the dark exposure.",
            "Sensor saturation and ISO noise require spatial and camera-aware treatment before fusion.",
            "RAW files and full sensor pixels are not copied into this report.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "raw_sensor_report.json"
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Sensor report: {destination}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect camera-native sensor black levels and clipping in three RAW exposures"
    )
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-stride", type=int, default=8)
    args = parser.parse_args()

    try:
        inspect_triplet(args.photos, args.output_dir, args.sample_stride)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
