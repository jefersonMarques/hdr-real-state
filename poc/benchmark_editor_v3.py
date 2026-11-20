import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def load_display_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)

    with Image.open(path) as source:
        image = source.convert("RGB")
        return np.asarray(image, dtype=np.uint8)


def resize_to_match(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return image

    source = Image.fromarray(image)
    resized = source.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8)


def rgb_mae(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("Images must share dimensions before comparison")

    error = np.abs(left.astype(np.float32) - right.astype(np.float32))
    return round(float(np.mean(error)), 4)


def image_statistics(image: np.ndarray) -> dict[str, float]:
    pixels = image.astype(np.float32) / 255.0
    brightness = np.sum(pixels * LUMINANCE_WEIGHTS, axis=2)
    return {
        "brightness_median": round(float(np.median(brightness)), 4),
        "dark_fraction": round(float(np.mean(brightness < 0.25)), 4),
        "bright_fraction": round(float(np.mean(brightness > 0.95)), 4),
        "rgb_mean": [round(float(value), 4) for value in np.mean(pixels, axis=(0, 1))],
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source:
        while part := source.read(1024 * 1024):
            digest.update(part)

    return digest.hexdigest()


def create_comparison(panels: list[tuple[str, np.ndarray]], output_path: Path) -> None:
    panel_width = 500
    panel_height = 360
    label_height = 34
    canvas = Image.new(
        "RGB",
        (panel_width * len(panels), panel_height + label_height),
        (245, 245, 245),
    )
    drawer = ImageDraw.Draw(canvas)

    for index, (title, pixels) in enumerate(panels):
        with Image.fromarray(pixels) as original:
            preview = ImageOps.contain(
                original,
                (panel_width, panel_height),
                method=Image.Resampling.LANCZOS,
            )

        x = panel_width * index + (panel_width - preview.width) // 2
        y = label_height + (panel_height - preview.height) // 2
        canvas.paste(preview, (x, y))
        drawer.text((panel_width * index + 12, 10), title, fill=(25, 25, 25))

    canvas.save(output_path, quality=95)


def build_report(
    before: np.ndarray,
    after: np.ndarray,
    reference: np.ndarray | None,
    files: dict[str, Path | None],
) -> dict:
    if before.shape != after.shape:
        raise ValueError("Neural editor must preserve original image dimensions")

    report = {
        "files": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for name, path in files.items()
            if path is not None
        },
        "dimensions": list(before.shape[:2]),
        "baseline": image_statistics(before),
        "editor": image_statistics(after),
        "editor_change_mae_0_255": rgb_mae(before, after),
        "warnings": [
            "Pixel MAE is descriptive, not a validated perceptual or editing-quality score.",
            "No geometry registration, color-managed Delta E or semantic region analysis.",
        ],
    }

    if reference is not None:
        aligned = resize_to_match(reference, before.shape[0], before.shape[1])
        report["reference"] = image_statistics(aligned)
        report["reference_comparison"] = {
            "baseline_mae_0_255": rgb_mae(before, aligned),
            "editor_mae_0_255": rgb_mae(after, aligned),
            "reference_resized": reference.shape[:2] != before.shape[:2],
        }

    return report


def run_comparison(
    before_path: Path,
    after_path: Path,
    reference_path: Path | None,
    output_directory: Path,
) -> None:
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_directory}")

    before = load_display_image(before_path)
    after = load_display_image(after_path)
    reference = load_display_image(reference_path) if reference_path is not None else None

    report = build_report(
        before,
        after,
        reference,
        {
            "baseline": before_path,
            "editor": after_path,
            "professional": reference_path,
        },
    )

    panels = [("Conventional input", before), ("AI photo editor", after)]

    if reference is not None:
        panels.append(("Photoshop reference", reference))

    output_directory.mkdir(parents=True, exist_ok=True)
    create_comparison(panels, output_directory / "comparison.jpg")
    (output_directory / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"Comparison: {output_directory / 'comparison.jpg'}")
    print(f"Report: {output_directory / 'report.json'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a neural photographic editor against a fixed HDR baseline"
    )
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        run_comparison(args.before, args.after, args.reference, args.output_dir)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
