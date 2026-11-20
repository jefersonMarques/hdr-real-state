import argparse
import sys
from pathlib import Path

from v5_core import merge_linear_triplet


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fuse three aligned linear RAW exposures into a float32 HDR image"
    )
    parser.add_argument("--triplet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--alignment", choices=("translation", "none"), default="translation"
    )
    parser.add_argument("--rows-per-chunk", type=int, default=256)
    args = parser.parse_args()

    try:
        report = merge_linear_triplet(
            args.triplet, args.output_dir, args.alignment, args.rows_per_chunk
        )
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Merged HDR: {report['output']}")
    print(f"Translations: {report['translations_xy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
