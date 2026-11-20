import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tifffile import imread, imwrite


LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def exposure_value(metadata: dict) -> float:
    shutter = float(metadata["ExposureTime"])
    sensitivity = float(metadata["ISO"])
    aperture = float(metadata["FNumber"])

    if shutter <= 0 or sensitivity <= 0 or aperture <= 0:
        raise ValueError("Invalid camera exposure metadata")

    return shutter * sensitivity / (aperture * aperture)


def read_exposure_factors(paths: tuple[Path, Path, Path]) -> tuple[tuple[float, ...], list[dict]]:
    command = [
        "exiftool", "-json", "-n", "-ExposureTime", "-ISO", "-FNumber",
        *(str(path) for path in paths),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    records = json.loads(result.stdout)

    if len(records) != 3:
        raise ValueError("Expected EXIF information for three RAW files")

    try:
        exposure_values = [exposure_value(record) for record in records]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Missing or invalid shutter, ISO or aperture metadata") from error

    reference_value = exposure_values[1]
    factors = tuple(value / reference_value for value in exposure_values)

    if not factors[0] < factors[1] < factors[2]:
        raise ValueError("Exposure order must be under, reference, over")

    return factors, records


def load_linear_raw(path: Path) -> np.ndarray:
    import rawpy

    if not path.is_file():
        raise FileNotFoundError(path)

    with rawpy.imread(str(path)) as source:
        image = source.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=16,
            gamma=(1.0, 1.0),
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a three-channel RGB RAW: {path}")

    return image.astype(np.float32) / 65535.0


def accumulate_exposure(
    image: np.ndarray,
    factor: float,
    accumulated: np.ndarray,
    accumulated_weights: np.ndarray,
    fallback: bool = False,
) -> None:
    if factor <= 0 or image.shape != accumulated.shape:
        raise ValueError("Invalid exposure factor or inconsistent image dimensions")

    maximum = np.max(image, axis=2)
    shadow_confidence = np.clip(maximum / 0.08, 0.0, 1.0)
    highlight_confidence = np.clip((0.985 - maximum) / 0.20, 0.0, 1.0)
    weights = shadow_confidence * highlight_confidence

    if fallback:
        weights += 0.00001

    accumulated += (image / factor) * weights[..., None]
    accumulated_weights += weights


def merge_linear_brackets(
    images: tuple[np.ndarray, np.ndarray, np.ndarray],
    factors: tuple[float, float, float],
) -> np.ndarray:
    if len({image.shape for image in images}) != 1:
        raise ValueError("Bracket dimensions do not match")

    accumulated = np.zeros_like(images[0], dtype=np.float32)
    weights = np.zeros(images[0].shape[:2], dtype=np.float32)

    for index, (image, factor) in enumerate(zip(images, factors, strict=True)):
        accumulate_exposure(image, factor, accumulated, weights, fallback=index == 0)

    return accumulated / np.maximum(weights[..., None], 1e-8)


def encode_srgb(linear: np.ndarray) -> np.ndarray:
    clipped = np.clip(linear, 0.0, 1.0)
    encoded = np.where(
        clipped <= 0.0031308,
        clipped * 12.92,
        1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
    )
    return np.uint8(np.round(encoded * 255.0))


def tone_map(hdr: np.ndarray) -> np.ndarray:
    if hdr.ndim != 3 or hdr.shape[2] != 3 or not np.isfinite(hdr).all():
        raise ValueError("HDR image must be finite RGB data")

    positive = np.maximum(hdr.astype(np.float32), 0.0)
    luminance = np.sum(positive * LUMINANCE_WEIGHTS, axis=2)
    samples = luminance[luminance > 1e-6]

    if samples.size == 0:
        return np.zeros_like(positive, dtype=np.uint8)

    middle = float(np.percentile(samples, 55.0))
    exposure = float(np.clip(0.55 / max(middle, 1e-6), 0.01, 64.0))
    scaled_luminance = exposure * luminance
    compressed_luminance = scaled_luminance / (1.0 + scaled_luminance)
    compressed_rgb = positive * (compressed_luminance / np.maximum(luminance, 1e-6))[..., None]
    return encode_srgb(compressed_rgb)


def save_jpeg(pixels: np.ndarray, path: Path) -> None:
    Image.fromarray(pixels, mode="RGB").save(path, quality=94)


def brightness_metrics(pixels: np.ndarray) -> dict[str, float]:
    encoded = pixels.astype(np.float32) / 255.0
    brightness = np.sum(encoded * LUMINANCE_WEIGHTS, axis=2)
    return {
        "brightness_p05": round(float(np.percentile(brightness, 5)), 4),
        "brightness_p50": round(float(np.percentile(brightness, 50)), 4),
        "brightness_p95": round(float(np.percentile(brightness, 95)), 4),
        "dark_pixel_fraction": round(float(np.mean(brightness < 0.25)), 4),
        "near_white_pixel_fraction": round(float(np.mean(brightness > 0.98)), 4),
    }


def make_comparison(panels: list[tuple[str, Path]], output_path: Path) -> None:
    panel_width = 480
    panel_height = 360
    label_height = 32
    canvas = Image.new(
        "RGB",
        (panel_width * len(panels), panel_height + label_height),
        (245, 245, 245),
    )
    drawing = ImageDraw.Draw(canvas)

    for index, (label, path) in enumerate(panels):
        with Image.open(path) as original:
            thumbnail = ImageOps.contain(original.convert("RGB"), (panel_width, panel_height))
        x = panel_width * index + (panel_width - thumbnail.width) // 2
        y = label_height + (panel_height - thumbnail.height) // 2
        canvas.paste(thumbnail, (x, y))
        drawing.text((panel_width * index + 12, 10), label, fill=(20, 20, 20))

    canvas.save(output_path, quality=93)


def create_diagnostics(
    paths: tuple[Path, Path, Path],
    output_directory: Path,
    neural_hdr: Path | None,
    neural_preview: Path | None,
    final_reference: Path | None,
) -> None:
    factors, metadata = read_exposure_factors(paths)
    output_directory.mkdir(parents=True, exist_ok=True)
    panels = []
    metrics = {"exposure_factors": list(factors), "input_exif": metadata, "images": {}}
    accumulated = None
    accumulated_weights = None

    for index, (name, path, factor) in enumerate(zip(("under", "reference", "over"), paths, factors, strict=True)):
        image = load_linear_raw(path)

        if accumulated is None:
            accumulated = np.zeros_like(image, dtype=np.float32)
            accumulated_weights = np.zeros(image.shape[:2], dtype=np.float32)

        preview_path = output_directory / f"{name}.jpg"
        preview_pixels = encode_srgb(image)
        save_jpeg(preview_pixels, preview_path)
        metrics["images"][name] = brightness_metrics(preview_pixels)
        panels.append((f"RAW {name}", preview_path))
        del preview_pixels

        accumulate_exposure(
            image, factor, accumulated, accumulated_weights, fallback=index == 0,
        )
        del image

    baseline = accumulated / np.maximum(accumulated_weights[..., None], 1e-8)
    baseline_path = output_directory / "conventional_hdr.tiff"
    imwrite(baseline_path, baseline.astype(np.float32), photometric="rgb")
    baseline_preview = output_directory / "conventional_preview.jpg"
    baseline_pixels = tone_map(baseline)
    save_jpeg(baseline_pixels, baseline_preview)
    metrics["images"]["conventional"] = brightness_metrics(baseline_pixels)
    panels.append(("Conventional HDR", baseline_preview))
    del baseline, baseline_pixels, accumulated, accumulated_weights

    if neural_hdr is not None:
        neural = np.asarray(imread(neural_hdr), dtype=np.float32)
        neural_pixels = tone_map(neural)
        mapped_path = output_directory / "neural_remapped.jpg"
        save_jpeg(neural_pixels, mapped_path)
        metrics["images"]["neural_remapped"] = brightness_metrics(neural_pixels)
        panels.append(("Neural (same tone map)", mapped_path))
        del neural, neural_pixels

    if neural_preview is not None:
        panels.append(("Neural (original preview)", neural_preview))
        with Image.open(neural_preview) as original:
            metrics["images"]["neural_original"] = brightness_metrics(np.asarray(original.convert("RGB")))

    if final_reference is not None:
        panels.append(("Professional reference", final_reference))
        with Image.open(final_reference) as original:
            metrics["images"]["professional_reference"] = brightness_metrics(np.asarray(original.convert("RGB")))

    make_comparison(panels, output_directory / "comparison.jpg")
    (output_directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"Comparison: {output_directory / 'comparison.jpg'}")
    print(f"Metrics: {output_directory / 'metrics.json'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="RAW exposure and neural/conventional HDR comparison")
    parser.add_argument("--under", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--over", type=Path, required=True)
    parser.add_argument("--neural-hdr", type=Path)
    parser.add_argument("--neural-preview", type=Path)
    parser.add_argument("--final-reference", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        create_diagnostics(
            (args.under, args.reference, args.over),
            args.output_dir,
            args.neural_hdr,
            args.neural_preview,
            args.final_reference,
        )
    except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
