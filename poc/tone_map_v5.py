import argparse
import sys
from pathlib import Path

from v5_core import tone_map_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a floating-point HDR image as JPEG and 16-bit TIFF"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = tone_map_file(args.source, args.output_dir)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"HDR preview: {report['preview']}")
    print(f"HDR TIFF: {report['tiff']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
