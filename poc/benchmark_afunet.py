import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageDraw, ImageOps
from tifffile import imread, imwrite

from hdr_afunet import load_afunet, load_raw, reconstruct
from hdr_diagnostics import brightness_metrics, tone_map
from hdr_pipeline_v2 import discover_brackets


CHECKPOINT_NAMES = ("Kalantari", "Hu", "Tel")


def verify_existing_baseline(report_path: Path, paths: tuple[Path, Path, Path]) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    recorded_names = [Path(value).name for value in report["inputs"]]
    current_names = [path.name for path in paths]

    if recorded_names != current_names:
        raise ValueError(
            "The conventional baseline was generated from different RAW filenames. "
            f"Recorded: {recorded_names}; current: {current_names}"
        )


def resize_float_image(image: np.ndarray, max_side: int) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image")

    if max_side <= 0 or max(image.shape[:2]) <= max_side:
        return image

    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    new_height = max(1, round(height * scale))
    new_width = max(1, round(width * scale))
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
    resized = functional.interpolate(
        tensor.unsqueeze(0),
        size=(new_height, new_width),
        mode="area",
    )
    return resized.squeeze(0).permute(1, 2, 0).numpy()


def mean_absolute_error(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("Images must have matching dimensions")

    difference = np.abs(left.astype(np.float32) - right.astype(np.float32))
    return round(float(np.mean(difference)), 4)


def reference_preview(path: Path, dimensions: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        if rgb.size != dimensions:
            rgb = rgb.resize(dimensions, Image.Resampling.LANCZOS)

        return np.asarray(rgb, dtype=np.uint8)


def analyze_preview(
    preview: np.ndarray,
    baseline: np.ndarray,
    professional: np.ndarray | None,
) -> dict:
    result = {
        "brightness": brightness_metrics(preview),
        "mean_absolute_difference_from_baseline_0_255": mean_absolute_error(
            preview, baseline
        ),
    }

    if professional is not None:
        result["approximate_rgb_mae_from_professional_0_255"] = (
            mean_absolute_error(preview, professional)
        )

    return result


def create_contact_sheet(
    images: list[tuple[str, Path]],
    output: Path,
) -> None:
    panel_width = 440
    panel_height = 320
    heading_height = 36
    canvas = Image.new(
        "RGB",
        (panel_width * len(images), panel_height + heading_height),
        (245, 245, 245),
    )
    drawing = ImageDraw.Draw(canvas)

    for index, (title, path) in enumerate(images):
        with Image.open(path) as source:
            thumbnail = ImageOps.contain(
                source.convert("RGB"),
                (panel_width, panel_height),
            )

        x = panel_width * index + (panel_width - thumbnail.width) // 2
        y = heading_height + (panel_height - thumbnail.height) // 2
        canvas.paste(thumbnail, (x, y))
        drawing.text((panel_width * index + 10, 10), title, fill=(20, 20, 20))

    canvas.save(output, quality=95)


def checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def run_checkpoint(
    name: str,
    checkpoint: Path,
    source: Path,
    images: tuple[np.ndarray, np.ndarray, np.ndarray],
    factors: tuple[float, float, float],
    device: torch.device,
    tile_size: int,
    overlap: int,
    output: Path,
) -> tuple[np.ndarray, float, dict]:
    print(f"AFUNet checkpoint: {name} | device: {device}", flush=True)
    model = load_afunet(source, checkpoint, device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    started = time.perf_counter()
    predicted_hdr = reconstruct(
        images,
        factors,
        model,
        device,
        tile_size,
        overlap,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - started
    imwrite(output / f"{name.lower()}_hdr.tiff", predicted_hdr, photometric="rgb")
    preview = tone_map(predicted_hdr)
    Image.fromarray(preview).save(output / f"{name.lower()}_preview.png")

    metadata = {
        "seconds": round(elapsed, 3),
        "checkpoint_sha256": checkpoint_hash(checkpoint),
        "raw_hdr_median": round(float(np.median(predicted_hdr)), 6),
        "raw_hdr_p99": round(float(np.percentile(predicted_hdr, 99)), 6),
    }

    del model, predicted_hdr

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return preview, elapsed, metadata


def run_benchmark(args: argparse.Namespace) -> None:
    paths, factors, exif = discover_brackets(args.photos)
    verify_existing_baseline(args.baseline_report, paths)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    checkpoints = [
        (name, args.checkpoint_dir / f"{name}.pth")
        for name in CHECKPOINT_NAMES
    ]

    for _, path in checkpoints:
        if not path.is_file():
            raise FileNotFoundError(path)

    if not args.baseline_hdr.is_file():
        raise FileNotFoundError(args.baseline_hdr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "Exposure order: "
        + ", ".join(path.name for path in paths)
        + f"; relative factors: {factors}",
        flush=True,
    )

    images = tuple(
        resize_float_image(load_raw(path), args.max_side)
        for path in paths
    )
    baseline_hdr = np.asarray(imread(args.baseline_hdr), dtype=np.float32)

    if images[0].shape[:2] != baseline_hdr.shape[:2]:
        raise ValueError("Saved baseline TIFF does not match the RAW dimensions")

    baseline_hdr = resize_float_image(baseline_hdr, args.max_side)
    baseline = tone_map(baseline_hdr)
    baseline_path = args.output_dir / "conventional_preview.png"
    Image.fromarray(baseline).save(baseline_path)

    professional = None
    if args.professional_jpeg is not None:
        professional = reference_preview(
            args.professional_jpeg,
            (baseline.shape[1], baseline.shape[0]),
        )

    report = {
        "inputs": [str(path) for path in paths],
        "relative_exposures": list(factors),
        "exif": exif,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "image_dimensions": list(baseline.shape[:2]),
        "max_side": args.max_side,
        "professional_reference": (
            str(args.professional_jpeg)
            if args.professional_jpeg is not None
            else None
        ),
        "metrics_note": (
            "Professional JPEG MAE is descriptive only. The reference may "
            "differ in crop, alignment, geometry, sharpening and grading. "
            "Each HDR is independently normalized by the same global tone-map "
            "function. A lower error alone does not prove a better HDR."
        ),
        "results": {},
    }

    report["results"]["conventional"] = analyze_preview(
        baseline,
        baseline,
        professional,
    )
    panels = [("Conventional (same tone map)", baseline_path)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name, checkpoint in checkpoints:
        preview, _, metadata = run_checkpoint(
            name,
            checkpoint,
            args.afunet_source,
            images,
            factors,
            device,
            args.tile_size,
            args.overlap,
            args.output_dir,
        )
        report["results"][name.lower()] = {
            **metadata,
            **analyze_preview(preview, baseline, professional),
        }
        panels.append((f"AFUNet {name}", args.output_dir / f"{name.lower()}_preview.png"))

    if args.professional_jpeg is not None:
        panels.append(("Photoshop reference", args.professional_jpeg))

    create_contact_sheet(panels, args.output_dir / "comparison.jpg")
    (args.output_dir / "benchmark.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"Saved: {args.output_dir / 'comparison.jpg'}", flush=True)
    print(f"Report: {args.output_dir / 'benchmark.json'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Independent comparison of three AFUNet checkpoints"
    )
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--afunet-source", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--baseline-hdr", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--professional-jpeg", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=32)
    args = parser.parse_args()

    try:
        run_benchmark(args)
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
