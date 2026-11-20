import argparse
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0


def smoothstep(values: np.ndarray, start: float, end: float) -> np.ndarray:
    if start >= end:
        raise ValueError("Invalid transition limits")

    normalized = np.clip((values - start) / (end - start), 0.0, 1.0)
    return normalized * normalized * (3.0 - 2.0 * normalized)


def luminance(image: np.ndarray) -> np.ndarray:
    return np.sum(image * LUMINANCE_WEIGHTS, axis=2)


def keep_large_components(binary: np.ndarray, minimum_fraction: float) -> np.ndarray:
    height, width = binary.shape
    visited = np.zeros(binary.shape, dtype=bool)
    selected = np.zeros(binary.shape, dtype=bool)
    minimum_size = max(12, int(height * width * minimum_fraction))
    neighbors = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

    for origin_y, origin_x in zip(*np.nonzero(binary & ~visited)):
        if visited[origin_y, origin_x]:
            continue

        visited[origin_y, origin_x] = True
        queue = deque([(int(origin_y), int(origin_x))])
        component = []

        while queue:
            y, x = queue.popleft()
            component.append((y, x))

            for offset_y, offset_x in neighbors:
                neighbor_y = y + offset_y
                neighbor_x = x + offset_x
                if not (0 <= neighbor_y < height and 0 <= neighbor_x < width):
                    continue
                if visited[neighbor_y, neighbor_x] or not binary[neighbor_y, neighbor_x]:
                    continue

                visited[neighbor_y, neighbor_x] = True
                queue.append((neighbor_y, neighbor_x))

        if len(component) >= minimum_size:
            for y, x in component:
                selected[y, x] = True

    return selected


def create_highlight_mask(under: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if under.shape != reference.shape or under.ndim != 3 or under.shape[2] != 3:
        raise ValueError("Source previews must be equally sized RGB images")

    raw_mask = smoothstep(luminance(under), 0.18, 0.35)
    raw_mask *= smoothstep(luminance(reference), 0.60, 0.88)
    height, width = raw_mask.shape
    coarse_width = max(1, width // 4)
    coarse_height = max(1, height // 4)
    coarse = Image.fromarray(np.uint8(np.round(raw_mask * 255.0)))
    coarse = coarse.resize((coarse_width, coarse_height), Image.Resampling.BOX)
    clusters = keep_large_components(np.asarray(coarse) >= 140, minimum_fraction=0.0015)
    mask = Image.fromarray(np.uint8(clusters) * 255, mode="L")

    if min(mask.size) >= 9:
        mask = mask.filter(ImageFilter.MaxFilter(7))
        mask = mask.filter(ImageFilter.MinFilter(7))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=1.0))

    mask = mask.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(mask, dtype=np.float32) / 255.0


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    return np.where(
        image <= 0.04045,
        image / 12.92,
        np.power((image + 0.055) / 1.055, 2.4),
    )


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    clipped = np.clip(image, 0.0, 1.0)
    return np.where(
        clipped <= 0.0031308,
        clipped * 12.92,
        1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
    )


def recover_highlights(
    base: np.ndarray,
    under: np.ndarray,
    mask: np.ndarray,
    exposure_stops: float,
) -> np.ndarray:
    if base.shape != under.shape or base.shape[:2] != mask.shape:
        raise ValueError("Source and mask dimensions must match")
    if not -3.0 <= exposure_stops <= 3.0:
        raise ValueError("Exposure adjustment must remain within three stops")

    base_linear = srgb_to_linear(base)
    highlight_linear = srgb_to_linear(under) * (2.0 ** exposure_stops)
    composite = base_linear * (1.0 - mask[..., None])
    composite += highlight_linear * mask[..., None]
    return linear_to_srgb(composite)


def save_rgb(image: np.ndarray, path: Path) -> None:
    encoded = np.uint8(np.round(np.clip(image, 0.0, 1.0) * 255.0))
    Image.fromarray(encoded, mode="RGB").save(path, quality=95)


def create_comparison(items: list[tuple[str, Path]], path: Path) -> None:
    panel_width = 480
    panel_height = 330
    header_height = 28
    canvas = Image.new("RGB", (len(items) * panel_width, panel_height + header_height), "white")
    painter = ImageDraw.Draw(canvas)

    for index, (label, image_path) in enumerate(items):
        with Image.open(image_path) as source:
            preview = ImageOps.contain(source.convert("RGB"), (panel_width, panel_height))
        x = index * panel_width + (panel_width - preview.width) // 2
        y = header_height + (panel_height - preview.height) // 2
        canvas.paste(preview, (x, y))
        painter.text((index * panel_width + 10, 8), label, fill="black")

    canvas.save(path, quality=94)


def main() -> int:
    parser = argparse.ArgumentParser(description="Experimental highlight recovery from real RAW brackets")
    parser.add_argument("--under-preview", required=True, type=Path)
    parser.add_argument("--reference-preview", required=True, type=Path)
    parser.add_argument("--conventional-preview", required=True, type=Path)
    parser.add_argument("--neural-preview", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--highlight-ev", type=float, default=0.35)
    args = parser.parse_args()

    try:
        under = read_rgb(args.under_preview)
        reference = read_rgb(args.reference_preview)
        conventional = read_rgb(args.conventional_preview)
        neural = read_rgb(args.neural_preview)
        mask = create_highlight_mask(under, reference)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        mask_path = args.output_dir / "highlight_mask.jpg"
        Image.fromarray(np.uint8(np.round(mask * 255))).save(mask_path, quality=95)
        comparisons = []

        for label, image in (("Conventional", conventional), ("Neural", neural)):
            original = args.output_dir / f"{label.lower()}_before.jpg"
            recovered = args.output_dir / f"{label.lower()}_window_recovered.jpg"
            save_rgb(image, original)
            save_rgb(recover_highlights(image, under, mask, args.highlight_ev), recovered)
            comparisons.extend(((f"{label} before", original), (f"{label} window protected", recovered)))

        create_comparison(comparisons, args.output_dir / "window_comparison.jpg")
        print(f"Highlighted area: {np.mean(mask > 0.5):.2%}")
        print(f"Comparison: {args.output_dir / 'window_comparison.jpg'}")
    except (OSError, ValueError) as error:
        parser.exit(status=1, message=f"Error: {error}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
