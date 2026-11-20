import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark_editor_v3 import (
    create_comparison,
    image_statistics,
    rgb_mae,
)


def load_evaluation_pixels(path: Path, max_side: int = 1200) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)

    with Image.open(path) as source:
        rgb = source.convert("RGB")
        scale = min(1.0, max_side / max(rgb.size))
        dimensions = (
            max(1, round(rgb.width * scale)),
            max(1, round(rgb.height * scale)),
        )
        if dimensions != rgb.size:
            rgb = rgb.resize(dimensions, Image.Resampling.LANCZOS)
        return np.asarray(rgb)


def evaluate_v5(
    before: Path,
    after: Path,
    reference: Path | None,
    output_dir: Path,
) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    original = load_evaluation_pixels(before)
    final = load_evaluation_pixels(after)
    target_size = (final.shape[1], final.shape[0])

    with Image.fromarray(original) as source:
        comparable = np.asarray(
            source.resize(target_size, Image.Resampling.LANCZOS)
        )

    panels = [("Source reference exposure", original), ("V5 HDR final", final)]
    report = {
        "before": str(before),
        "after": str(after),
        "source_statistics": image_statistics(original),
        "final_statistics": image_statistics(final),
        "difference_from_source_0_255": rgb_mae(comparable, final),
        "pixel_difference_caveat": (
            "Unregistered resampling and photographic grading can dominate pixel MAE."
        ),
    }

    if reference is not None:
        professional = load_evaluation_pixels(reference)
        with Image.fromarray(professional) as target:
            comparable_target = np.asarray(
                target.resize(target_size, Image.Resampling.LANCZOS)
            )
        panels.append(("Professional reference", professional))
        report["reference"] = str(reference)
        report["professional_statistics"] = image_statistics(professional)
        report["difference_from_professional_0_255"] = rgb_mae(
            comparable_target, final
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    create_comparison(panels, output_dir / "comparison.jpg")
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare source, V5 real HDR export and a professional reference"
    )
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        evaluate_v5(args.before, args.after, args.reference, args.output_dir)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Comparison: {args.output_dir / 'comparison.jpg'}")
    print(f"Report: {args.output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
