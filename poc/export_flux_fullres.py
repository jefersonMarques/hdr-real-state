import argparse
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps
from tifffile import imread, imwrite


LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def load_srgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)

    with Image.open(path) as source:
        upright = ImageOps.exif_transpose(source).convert("RGB")
        profile_data = source.info.get("icc_profile")

        if profile_data:
            input_profile = ImageCms.ImageCmsProfile(io.BytesIO(profile_data))
            output_profile = ImageCms.ImageCmsProfile(io.BytesIO(SRGB_PROFILE))
            return ImageCms.profileToProfile(
                upright,
                input_profile,
                output_profile,
                outputMode="RGB",
            )

        return upright.copy()


def decode_srgb(pixels: np.ndarray) -> np.ndarray:
    values = pixels.astype(np.float32) / 255.0
    return np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )


def encode_srgb(linear: np.ndarray) -> np.ndarray:
    values = np.clip(linear, 0.0, 1.0)
    return np.where(
        values <= 0.0031308,
        12.92 * values,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    )



def tone_map_linear_hdr(hdr: np.ndarray, highlight_mask: np.ndarray) -> np.ndarray:
    if hdr.ndim != 3 or hdr.shape[2] != 3 or hdr.shape[:2] != highlight_mask.shape:
        raise ValueError("HDR and highlight mask dimensions must match")

    if not np.isfinite(hdr).all():
        raise ValueError("HDR image contains nonfinite values")

    positive = np.maximum(hdr.astype(np.float32), 0.0)
    scene_luma = np.sum(positive * LUMINANCE_WEIGHTS, axis=2)
    interior = scene_luma[(highlight_mask < 0.10) & (scene_luma > 1e-6)]
    windows = scene_luma[(highlight_mask > 0.90) & (scene_luma > 1e-6)]

    if interior.size < 64:
        raise ValueError("Not enough reliable pixels in the HDR interior")

    interior_median = max(float(np.median(interior)), 1e-6)
    interior_scale = (0.42 / (1.0 - 0.42)) / interior_median

    if windows.size >= 64:
        window_median = max(float(np.median(windows)), 1e-6)
        window_scale = min(
            (0.40 / (1.0 - 0.40)) / window_median,
            interior_scale,
        )
    else:
        window_scale = interior_scale

    interior_scale = float(np.clip(interior_scale, 0.01, 100.0))
    window_scale = float(np.clip(window_scale, 0.001, 100.0))
    exposure_map = np.exp(
        np.log(interior_scale) * (1.0 - highlight_mask)
        + np.log(window_scale) * highlight_mask
    ).astype(np.float32)
    scaled_luma = scene_luma * exposure_map
    mapped_luma = scaled_luma / (1.0 + scaled_luma)
    return positive * (mapped_luma / np.maximum(scene_luma, 1e-6))[..., None]


def box_blur(values: np.ndarray, radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("Blur radius must be nonnegative")

    if radius == 0:
        return values.copy()

    window = radius * 2 + 1
    padded = np.pad(
        values,
        ((radius, radius), (radius, radius), (0, 0)),
        mode="edge",
    )
    summed = np.pad(
        padded.cumsum(axis=0, dtype=np.float64).cumsum(axis=1, dtype=np.float64),
        ((1, 0), (1, 0), (0, 0)),
        mode="constant",
    )
    blurred = (
        summed[window:, window:]
        - summed[:-window, window:]
        - summed[window:, :-window]
        + summed[:-window, :-window]
    ) / (window * window)
    return blurred.astype(np.float32)


def calculate_adjustments(
    source: np.ndarray,
    edited: np.ndarray,
    max_exposure_ev: float,
    max_color_log: float,
    blur_radius: int,
) -> np.ndarray:
    if source.shape != edited.shape or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("The FLUX input and edited image must have matching RGB dimensions")

    if max_exposure_ev <= 0 or max_color_log < 0:
        raise ValueError("Adjustment limits must be positive")

    source_linear = decode_srgb(source)
    edited_linear = decode_srgb(edited)
    source_luma = np.sum(source_linear * LUMINANCE_WEIGHTS, axis=2)
    edited_luma = np.sum(edited_linear * LUMINANCE_WEIGHTS, axis=2)

    exposure_log = np.log((edited_luma + 0.02) / (source_luma + 0.02))
    exposure_log = np.clip(
        exposure_log,
        -max_exposure_ev * math.log(2.0),
        max_exposure_ev * math.log(2.0),
    )

    channel_log = np.log((edited_linear + 0.03) / (source_linear + 0.03))
    color_log = np.clip(
        channel_log - exposure_log[..., None],
        -max_color_log,
        max_color_log,
    )
    adjustments = np.concatenate((exposure_log[..., None], color_log), axis=2)
    return box_blur(box_blur(adjustments, blur_radius), blur_radius)


def upscale_adjustments(
    adjustments: np.ndarray,
    target_size: tuple[int, int],
) -> list[np.ndarray]:
    result = []

    for channel in range(adjustments.shape[2]):
        channel_pixels = np.ascontiguousarray(adjustments[..., channel], dtype=np.float32)
        with Image.fromarray(channel_pixels, mode="F") as image:
            scaled = image.resize(target_size, Image.Resampling.BICUBIC)
            result.append(np.asarray(scaled, dtype=np.float32))

    return result


def apply_adjustments(
    source: np.ndarray,
    adjustments: list[np.ndarray],
    strength: float,
    rows_per_chunk: int,
    linear_source: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        raise ValueError("Expected an 8-bit RGB source image")

    if len(adjustments) != 4 or any(item.shape != source.shape[:2] for item in adjustments):
        raise ValueError("Adjustment fields must have the source image dimensions")

    if not 0.0 <= strength <= 1.0 or rows_per_chunk <= 0:
        raise ValueError("Invalid strength or chunk height")

    if linear_source is not None:
        if linear_source.shape != source.shape or not np.isfinite(linear_source).all():
            raise ValueError("Linear HDR render must be finite and match source geometry")

    height, width, _ = source.shape
    output_16bit = np.empty((height, width, 3), dtype=np.uint16)
    output_8bit = np.empty_like(source)

    for start in range(0, height, rows_per_chunk):
        end = min(start + rows_per_chunk, height)
        linear = (
            linear_source[start:end]
            if linear_source is not None
            else decode_srgb(source[start:end])
        )
        exposure = adjustments[0][start:end]
        color = np.stack(
            (adjustments[1][start:end], adjustments[2][start:end], adjustments[3][start:end]),
            axis=2,
        )
        exponent = strength * (exposure[..., None] + color)
        adjusted_linear = linear * np.exp(exponent)
        encoded = encode_srgb(adjusted_linear)

        output_16bit[start:end] = np.uint16(np.round(encoded * 65535.0))
        output_8bit[start:end] = np.uint8(np.round(encoded * 255.0))

    return output_8bit, output_16bit


def export_photo(
    full_resolution_source: Path,
    flux_source: Path,
    flux_edit: Path,
    output_directory: Path,
    strength: float,
    max_exposure_ev: float,
    max_color_log: float,
    blur_radius: int,
    rows_per_chunk: int,
    linear_hdr: Path | None = None,
    highlight_mask: Path | None = None,
) -> None:
    if (linear_hdr is None) != (highlight_mask is None):
        raise ValueError("Linear HDR and its highlight mask must be supplied together")

    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {output_directory}")

    full_image = load_srgb(full_resolution_source)
    low_source = load_srgb(flux_source)
    low_edited = load_srgb(flux_edit)

    if low_source.size != low_edited.size:
        raise ValueError("The original FLUX input and generated output have different sizes")

    if full_image.width < low_source.width or full_image.height < low_source.height:
        raise ValueError("The supposed full-resolution source is smaller than the FLUX input")

    adjustments = calculate_adjustments(
        np.asarray(low_source),
        np.asarray(low_edited),
        max_exposure_ev,
        max_color_log,
        blur_radius,
    )
    resized = upscale_adjustments(adjustments, full_image.size)
    linear_source = None

    if linear_hdr is not None and highlight_mask is not None:
        high_precision_hdr = np.asarray(imread(linear_hdr), dtype=np.float32)

        with Image.open(highlight_mask) as source_mask:
            mask = np.asarray(source_mask.convert("L"), dtype=np.float32) / 255.0

        linear_source = tone_map_linear_hdr(high_precision_hdr, mask)

        if linear_source.shape[:2] != (full_image.height, full_image.width):
            raise ValueError("HDR geometry differs from the original full-resolution image")

    pixels_8bit, pixels_16bit = apply_adjustments(
        np.asarray(full_image),
        resized,
        strength,
        rows_per_chunk,
        linear_source=linear_source,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    jpeg_path = output_directory / "photo_fullres.jpg"
    tiff_path = output_directory / "photo_fullres_16bit.tiff"

    Image.fromarray(pixels_8bit, mode="RGB").save(
        jpeg_path,
        format="JPEG",
        quality=98,
        subsampling=0,
        optimize=False,
        icc_profile=SRGB_PROFILE,
    )
    imwrite(
        tiff_path,
        pixels_16bit,
        photometric="rgb",
        compression="deflate",
        metadata=None,
        extratags=((34675, "B", len(SRGB_PROFILE), SRGB_PROFILE, False),),
    )

    report = {
        "source": str(full_resolution_source),
        "flux_input": str(flux_source),
        "flux_output": str(flux_edit),
        "source_dimensions": list(full_image.size),
        "flux_dimensions": list(low_source.size),
        "jpeg": {"path": str(jpeg_path), "quality": 98, "chroma_subsampling": "4:4:4"},
        "tiff": {"path": str(tiff_path), "bits_per_channel": 16, "compression": "lossless_deflate"},
        "strength": strength,
        "linear_hdr_source": str(linear_hdr) if linear_hdr is not None else None,
        "highlight_mask": str(highlight_mask) if highlight_mask is not None else None,
        "tiff_uses_high_precision_hdr": linear_source is not None,
        "maximum_exposure_adjustment_ev": max_exposure_ev,
        "maximum_color_log_shift": max_color_log,
        "blur_radius": blur_radius,
        "warnings": [
            "This is an original-resolution source-preserving render guided by FLUX, not native full-resolution generative inference.",
            "The TIFF stores 16-bit computed output. Without --linear-hdr it is based on 8-bit RGB and cannot restore lost RAW precision.",
            "The learned style is approximated by smooth bounded light/color adjustments; exact FLUX pixel content is not reproduced.",
            "Fine details already clipped in the source cannot be reconstructed by this export.",
            "No verified alignment, structural edit detection or professional photo-quality approval has been performed.",
        ],
    }
    (output_directory / "export_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"JPEG: {jpeg_path} ({full_image.width}x{full_image.height})", flush=True)
    print(f"TIFF: {tiff_path} ({full_image.width}x{full_image.height})", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export native-source-resolution photos guided by a low-resolution FLUX edit"
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--flux-input", required=True, type=Path)
    parser.add_argument("--flux-edit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--max-exposure-ev", type=float, default=0.65)
    parser.add_argument("--max-color-log", type=float, default=0.12)
    parser.add_argument("--blur-radius", type=int, default=12)
    parser.add_argument("--rows-per-chunk", type=int, default=256)
    parser.add_argument("--linear-hdr", type=Path)
    parser.add_argument("--highlight-mask", type=Path)
    args = parser.parse_args()

    try:
        export_photo(
            args.source,
            args.flux_input,
            args.flux_edit,
            args.output_dir,
            args.strength,
            args.max_exposure_ev,
            args.max_color_log,
            args.blur_radius,
            args.rows_per_chunk,
            linear_hdr=args.linear_hdr,
            highlight_mask=args.highlight_mask,
        )
    except (OSError, ValueError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
