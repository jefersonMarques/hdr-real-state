import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms
from tifffile import imread, imwrite

from hdr_diagnostics import LUMINANCE_WEIGHTS, encode_srgb
from prepare_triplet_v4 import camera_white_balance, discover_triplet


BRACKET_NAMES = ("under", "reference", "over")
SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def load_linear_raw(path: Path, white_balance: tuple[float, ...]) -> np.ndarray:
    import rawpy

    with rawpy.imread(str(path)) as raw:
        developed = raw.postprocess(
            user_wb=list(white_balance),
            use_camera_wb=False,
            no_auto_bright=True,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=16,
            gamma=(1.0, 1.0),
        )

    if developed.ndim != 3 or developed.shape[2] != 3:
        raise ValueError(f"Invalid developed RAW image: {path}")

    return developed.astype(np.float32) / 65535.0


def prepare_linear_triplet(photos: Path, output_dir: Path) -> dict:
    paths, factors, metadata = discover_triplet(photos)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    white_balance = camera_white_balance(paths[1])
    output_dir.mkdir(parents=True, exist_ok=True)
    dimensions = None
    records = []

    for label, path, factor, exif in zip(
        BRACKET_NAMES, paths, factors, metadata, strict=True
    ):
        image = load_linear_raw(path, white_balance)

        if dimensions is None:
            dimensions = image.shape
        elif image.shape != dimensions:
            raise ValueError("All bracket RAWs must develop to identical image geometry")

        destination = output_dir / f"{label}_linear.npy"
        np.save(destination, image, allow_pickle=False)
        records.append({
            "role": label,
            "raw_path": str(path.resolve()),
            "linear_path": str(destination.resolve()),
            "relative_exposure": float(factor),
            "exif": exif,
        })
        del image

    manifest = {
        "input_type": "three camera RAW exposures",
        "color_space": "linear sRGB (LibRaw developed, shared camera white balance)",
        "encoding": "float32",
        "dimensions": list(dimensions),
        "shared_white_balance": list(white_balance),
        "inputs": records,
        "limitations": [
            "LibRaw demosaicing and color transforms are not identical to camera sensor radiance.",
            "Camera response and ISO calibration may introduce radiometric uncertainty.",
            "Registration, deghosting and scene consistency are performed in the merge stage.",
        ],
    }
    (output_dir / "triplet_v5.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def read_manifest(path: Path) -> tuple[list[np.ndarray], list[float], dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = manifest.get("inputs")

    if not isinstance(records, list) or len(records) != 3:
        raise ValueError("Exactly three RAW exposure records are required")

    if [record.get("role") for record in records] != list(BRACKET_NAMES):
        raise ValueError("Incorrect bracket order")

    images = [
        np.load(Path(record["linear_path"]), mmap_mode="r", allow_pickle=False)
        for record in records
    ]
    factors = [float(record["relative_exposure"]) for record in records]

    if len({image.shape for image in images}) != 1:
        raise ValueError("The three developed RAWs have different dimensions")

    if images[0].ndim != 3 or images[0].shape[-1] != 3:
        raise ValueError("Expected three-channel linear RGB data")

    if not (0 < factors[0] < factors[1] < factors[2]):
        raise ValueError("Invalid camera exposure factors")

    return images, factors, manifest


def make_registration_image(image: np.ndarray, max_side: int = 960) -> tuple[np.ndarray, float]:
    import cv2

    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    target = (max(32, round(width * scale)), max(32, round(height * scale)))
    preview = cv2.resize(np.asarray(image), target, interpolation=cv2.INTER_AREA)
    luminance = np.sum(preview * LUMINANCE_WEIGHTS, axis=2)
    compressed = np.log1p(25.0 * np.maximum(luminance, 0.0)).astype(np.float32)
    edges = cv2.Laplacian(compressed, cv2.CV_32F, ksize=3)
    return edges, scale


def estimate_translation(
    reference: np.ndarray,
    source: np.ndarray,
    min_response: float = 0.04,
) -> tuple[float, float, float]:
    import cv2

    reference_edges, scale = make_registration_image(reference)
    source_edges, other_scale = make_registration_image(source)

    if reference_edges.shape != source_edges.shape or abs(scale - other_scale) > 1e-6:
        raise ValueError("Cannot align differently sized camera frames")

    window = cv2.createHanningWindow(
        (reference_edges.shape[1], reference_edges.shape[0]), cv2.CV_32F
    )
    translation, confidence = cv2.phaseCorrelate(
        reference_edges, source_edges, window
    )
    dx = float(translation[0]) / scale
    dy = float(translation[1]) / scale

    if not all(math.isfinite(item) for item in (dx, dy, confidence)):
        raise ValueError("Nonfinite camera alignment result")

    if confidence < min_response:
        raise ValueError(
            f"Unreliable bracket registration (confidence={confidence:.3f}). "
            "Check for handheld movement or try --alignment none only if frames are registered."
        )

    if max(abs(dx), abs(dy)) > 0.07 * max(reference.shape[:2]):
        raise ValueError("Camera movement exceeds translation alignment limit")

    return dx, dy, float(confidence)


def aligned_chunk(
    image: np.ndarray,
    start: int,
    end: int,
    translation: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    height, width = image.shape[:2]
    dx, dy = translation

    if dx == 0.0 and dy == 0.0:
        return np.asarray(image[start:end]), np.ones((end - start, width), dtype=bool)

    x = np.arange(width, dtype=np.float32)[None, :] + np.float32(dx)
    y = np.arange(start, end, dtype=np.float32)[:, None] + np.float32(dy)
    map_x = np.broadcast_to(x, (end - start, width)).copy()
    map_y = np.broadcast_to(y, (end - start, width)).copy()
    valid = (
        (map_x >= 0.0)
        & (map_x <= width - 1)
        & (map_y >= 0.0)
        & (map_y <= height - 1)
    )
    pixels = cv2.remap(
        image, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    return pixels, valid


def fusion_chunk(
    inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
    factors: tuple[float, float, float],
    validity: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    if len({item.shape for item in inputs}) != 1:
        raise ValueError("Cannot merge images with inconsistent dimensions")

    if not (0 < factors[0] < factors[1] < factors[2]):
        raise ValueError("Exposure factors must increase from dark to bright")

    if validity is None:
        validity = tuple(
            np.ones(inputs[0].shape[:2], dtype=bool) for _ in inputs
        )

    normalized_reference = inputs[1].astype(np.float32) / factors[1]
    reference_max = np.max(inputs[1], axis=2)
    reference_luma = np.sum(normalized_reference * LUMINANCE_WEIGHTS, axis=2)
    reference_reliable = (reference_max > 0.1) & (reference_max < 0.9)

    total = np.zeros_like(normalized_reference, dtype=np.float32)
    total_weight = np.zeros(reference_max.shape, dtype=np.float32)

    for index, (pixels, factor, valid) in enumerate(
        zip(inputs, factors, validity, strict=True)
    ):
        pixels = np.asarray(pixels, dtype=np.float32)
        maximum = np.max(pixels, axis=2)
        shadows = np.clip((maximum - 0.008) / 0.07, 0.0, 1.0)
        highlights = np.clip((0.985 - maximum) / 0.08, 0.0, 1.0)
        weight = shadows * highlights * valid

        candidate = pixels / factor

        if index != 1:
            candidate_luma = np.sum(candidate * LUMINANCE_WEIGHTS, axis=2)
            relative_difference = np.abs(candidate_luma - reference_luma)
            relative_difference /= np.maximum(reference_luma, 0.04)
            mismatched = reference_reliable & (relative_difference > 0.4)
            weight *= np.where(mismatched, 0.05, 1.0)

        total += candidate * weight[..., None]
        total_weight += weight

    fallback = inputs[0].astype(np.float32) / factors[0]
    result = np.where(
        total_weight[..., None] > 1e-6,
        total / np.maximum(total_weight[..., None], 1e-6),
        fallback,
    )
    return np.maximum(result, 0.0)


def merge_linear_triplet(
    manifest_path: Path,
    output_dir: Path,
    alignment: str = "translation",
    rows_per_chunk: int = 256,
) -> dict:
    if rows_per_chunk < 1:
        raise ValueError("Chunk height must be positive")

    if alignment not in {"translation", "none"}:
        raise ValueError("Alignment must be translation or none")

    inputs, factors, manifest = read_manifest(manifest_path)
    merged_path = output_dir / "merged_hdr.npy"

    if merged_path.exists():
        raise FileExistsError(merged_path)

    shifts = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    confidences = [None, 1.0, None]

    if alignment == "translation":
        for index in (0, 2):
            dx, dy, score = estimate_translation(inputs[1], inputs[index])
            shifts[index] = (dx, dy)
            confidences[index] = score

    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = inputs[1].shape[:2]
    merged = np.lib.format.open_memmap(
        merged_path, mode="w+", dtype=np.float32, shape=(height, width, 3)
    )

    for start in range(0, height, rows_per_chunk):
        end = min(start + rows_per_chunk, height)
        aligned = tuple(
            aligned_chunk(image, start, end, shift)
            for image, shift in zip(inputs, shifts, strict=True)
        )
        merged[start:end] = fusion_chunk(
            tuple(item[0] for item in aligned),
            tuple(factors),
            tuple(item[1] for item in aligned),
        )

    merged.flush()
    report = {
        "source_manifest": str(manifest_path),
        "output": str(merged_path),
        "dimensions": [height, width],
        "alignment": alignment,
        "translations_xy": shifts,
        "registration_confidences": confidences,
        "relative_exposures": factors,
        "limitations": [
            "Translation-only registration cannot correct perspective or foliage motion.",
            "Motion rejection is based on exposure-normalized disagreement, not optical flow.",
            "All source frames must represent the same photographed scene.",
        ],
    }
    (output_dir / "merge_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def tone_map_linear(
    hdr: np.ndarray,
    target_mid: float = 0.42,
    local_strength: float = 0.25,
) -> np.ndarray:
    import cv2

    if hdr.ndim != 3 or hdr.shape[2] != 3 or not 0 < target_mid < 1:
        raise ValueError("Expected HDR RGB image and valid target midtone")

    if not 0 <= local_strength <= 0.75:
        raise ValueError("Local strength must be within [0, 0.75]")

    if not np.isfinite(hdr).all():
        raise ValueError("HDR image contains nonfinite pixels")

    pixels = np.maximum(np.asarray(hdr, dtype=np.float32), 0.0)
    luma = np.sum(pixels * LUMINANCE_WEIGHTS, axis=2)
    samples = luma[luma > 1e-7]

    if samples.size == 0:
        return np.zeros_like(pixels, dtype=np.float32)

    median = max(float(np.median(samples)), 1e-6)
    global_scale = (target_mid / (1.0 - target_mid)) / median

    height, width = luma.shape
    resize_factor = min(1.0, 960.0 / max(height, width))
    size = (max(32, round(width * resize_factor)), max(32, round(height * resize_factor)))
    small = cv2.resize(np.log1p(luma), size, interpolation=cv2.INTER_AREA)
    edge_preserved = cv2.bilateralFilter(
        small.astype(np.float32), d=0, sigmaColor=0.30, sigmaSpace=22.0
    )
    base = np.expm1(
        cv2.resize(edge_preserved, (width, height), interpolation=cv2.INTER_LINEAR)
    )
    adjustment = np.clip(
        np.power(median / np.maximum(base, 1e-5), local_strength),
        0.60, 2.40,
    )

    mapped_luma = luma * np.float32(global_scale) * adjustment
    compressed_luma = mapped_luma / (1.0 + mapped_luma)
    scale = compressed_luma / np.maximum(luma, 1e-8)
    return np.clip(pixels * scale[..., None], 0.0, 1.0).astype(np.float32)


def encode_srgb_16(linear: np.ndarray) -> np.ndarray:
    clipped = np.clip(linear, 0.0, 1.0)
    encoded = np.where(
        clipped <= 0.0031308,
        clipped * 12.92,
        1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
    )
    return np.uint16(np.round(encoded * 65535.0))


def tone_map_file(hdr_path: Path, output_dir: Path) -> dict:
    preview_path = output_dir / "tone_mapped_preview.jpg"
    tiff_path = output_dir / "tone_mapped_16bit.tiff"

    if preview_path.exists() or tiff_path.exists():
        raise FileExistsError("A tone-mapping output already exists")

    hdr = np.load(hdr_path, mmap_mode="r", allow_pickle=False)
    mapped = tone_map_linear(hdr)
    image_16 = encode_srgb_16(mapped)
    preview = (image_16.astype(np.uint32) * 255 + 32767) // 65535

    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview.astype(np.uint8)).save(
        preview_path, "JPEG", quality=96, subsampling=0, optimize=False,
        icc_profile=SRGB_PROFILE,
    )
    imwrite(
        tiff_path,
        image_16,
        photometric="rgb",
        compression="deflate",
        metadata=None,
        extratags=((34675, "B", len(SRGB_PROFILE), SRGB_PROFILE, False),),
    )

    report = {
        "source": str(hdr_path),
        "preview": str(preview_path),
        "tiff": str(tiff_path),
        "dimensions": list(image_16.shape[:2]),
        "bits_per_channel": 16,
        "color_space": "sRGB",
        "limitations": [
            "Local tone mapping requires review for halos on window boundaries.",
            "Highlight regions are not semantically segmented as windows.",
        ],
    }
    (output_dir / "tone_map_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def export_final(hdr_path: Path, tone_mapped_path: Path, output_dir: Path) -> dict:
    import shutil

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Final export directory is not empty: {output_dir}")

    raw_hdr = np.load(hdr_path, mmap_mode="r", allow_pickle=False)
    mapped = imread(tone_mapped_path)

    if mapped.dtype != np.uint16 or mapped.ndim != 3 or mapped.shape[2] != 3:
        raise ValueError("Final TIFF must contain 16-bit RGB pixels")

    if raw_hdr.shape != mapped.shape:
        raise ValueError("HDR and final TIFF geometries do not match")

    output_dir.mkdir(parents=True, exist_ok=True)
    jpeg_path = output_dir / "photo_final.jpg"
    tiff_path = output_dir / "photo_final_16bit.tiff"

    preview = np.uint8((mapped.astype(np.uint32) * 255 + 32767) // 65535)
    Image.fromarray(preview).save(
        jpeg_path, "JPEG", quality=98, subsampling=0,
        optimize=False, icc_profile=SRGB_PROFILE,
    )
    shutil.copyfile(tone_mapped_path, tiff_path)
    report = {
        "source_hdr": str(hdr_path),
        "source_tiff": str(tone_mapped_path),
        "jpeg": str(jpeg_path),
        "tiff": str(tiff_path),
        "dimensions": list(mapped.shape[:2]),
        "jpeg_quality": 98,
        "jpeg_chroma_subsampling": "4:4:4",
        "tiff_bits_per_channel": 16,
        "note": "Camera-resolution output; photographic quality not yet approved.",
    }
    (output_dir / "export_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
