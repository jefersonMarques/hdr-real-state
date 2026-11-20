import argparse
import sys
from pathlib import Path

from v5_core import prepare_linear_triplet


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Decode and prepare three exposure-bracketed RAWs as linear RGB"
    )
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = prepare_linear_triplet(args.photos, args.output_dir)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Prepared: {args.output_dir / 'triplet_v5.json'}")
    for record in report["inputs"]:
        print(f"{record['role']}: {record['raw_path']} ({record['relative_exposure']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
