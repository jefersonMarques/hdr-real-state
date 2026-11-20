import argparse
import sys
from pathlib import Path

from v5_core import export_final


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a camera-resolution JPEG and genuine 16-bit RGB TIFF"
    )
    parser.add_argument("--hdr", type=Path, required=True)
    parser.add_argument("--tone-mapped", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = export_final(args.hdr, args.tone_mapped, args.output_dir)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Final JPEG: {report['jpeg']}")
    print(f"Final TIFF: {report['tiff']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
