import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps


def calculate_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("Input image dimensions must be positive")

    if max_side < 64:
        raise ValueError("Maximum side must be at least 64 pixels")

    scale = min(1.0, max_side / max(width, height))
    output_width = max(1, round(width * scale))
    output_height = max(1, round(height * scale))

    if min(output_width, output_height) < 32:
        raise ValueError("DeepLPF needs both image dimensions to be at least 32 pixels")

    return output_width, output_height


def prepare_image(source_path: Path, output_path: Path, max_side: int) -> tuple[int, int]:
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    if output_path.exists():
        raise FileExistsError(f"Prepared image already exists: {output_path}")

    with Image.open(source_path) as source:
        normalized = ImageOps.exif_transpose(source)
        rgb = normalized.convert("RGB")
        output_size = calculate_size(*rgb.size, max_side)

        if output_size != rgb.size:
            rgb = rgb.resize(output_size, Image.Resampling.LANCZOS)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(output_path, format="PNG")

    return output_size


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a bounded-resolution RGB image for DeepLPF inference"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=1024)
    args = parser.parse_args()

    try:
        width, height = prepare_image(args.source, args.output, args.max_side)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Prepared image: {args.output} ({width}x{height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
