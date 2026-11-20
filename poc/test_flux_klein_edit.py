import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageOps


MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_PROMPT = (
    "Edit this existing real estate photograph to improve natural lighting "
    "and photographic color balance. Make the white ceiling neutral, "
    "maintain realistic wood and foliage colors, and keep window views "
    "recognizable. Preserve the exact room layout, objects, furniture, "
    "doors, frames, curtains, exterior scenery and fine tree leaf details. "
    "Do not add, remove, replace or redesign any elements. "
    "Produce a realistic photograph, not an illustration."
)


def image_dimensions(width: int, height: int, max_side: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("Input image dimensions must be positive")

    if max_side < 256 or max_side % 16 != 0:
        raise ValueError("Maximum side must be at least 256 and divisible by 16")

    scale = min(1.0, max_side / max(width, height))
    output_width = max(16, math.floor(width * scale / 16) * 16)
    output_height = max(16, math.floor(height * scale / 16) * 16)

    if min(output_width, output_height) < 256:
        raise ValueError("Resized image is too small for this evaluation")

    return output_width, output_height


def load_input(path: Path, max_side: int) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)

    with Image.open(path) as source:
        upright = ImageOps.exif_transpose(source)
        image = upright.convert("RGB")

    target = image_dimensions(image.width, image.height, max_side)

    if image.size != target:
        image = image.resize(target, Image.Resampling.LANCZOS)

    return image


def run_editor(
    input_path: Path,
    output_directory: Path,
    prompt: str,
    max_side: int,
    seed: int,
    steps: int,
    offload: str,
) -> None:
    import torch
    from diffusers import Flux2KleinPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the active Python environment")

    if not prompt.strip():
        raise ValueError("The editing prompt cannot be empty")

    if steps < 1:
        raise ValueError("Inference steps must be positive")

    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_directory}")

    image = load_input(input_path, max_side)
    print(f"Input: {input_path} | Processing size: {image.width}x{image.height}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)} | Offload: {offload}", flush=True)

    pipeline = Flux2KleinPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )

    if offload == "sequential":
        pipeline.enable_sequential_cpu_offload()
    else:
        pipeline.enable_model_cpu_offload()

    generator = torch.Generator(device="cuda").manual_seed(seed)
    print("Starting FLUX.2 image editing inference", flush=True)

    with torch.inference_mode():
        result = pipeline(
            image=image,
            prompt=prompt,
            height=image.height,
            width=image.width,
            guidance_scale=1.0,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]

    output_directory.mkdir(parents=True, exist_ok=True)
    output_image = output_directory / "flux_edit.png"
    result.save(output_image)

    report = {
        "model_id": MODEL_ID,
        "input_path": str(input_path),
        "output_path": str(output_image),
        "resolution": [image.width, image.height],
        "seed": seed,
        "steps": steps,
        "offload": offload,
        "prompt": prompt,
        "limitations": [
            "Generative editing can alter property geometry and fine exterior details.",
            "Output is a lower-resolution evaluation image, not a verified faithful property photograph.",
            "No architectural or semantic fidelity checks have been implemented.",
        ],
    }

    (output_directory / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved: {output_image}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experimental local FLUX.2 Klein real-estate photo editing"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/v2-new-photos/conventional_local.jpg"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/flux-klein-test-01"),
    )
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offload", choices=("sequential", "model"), default="sequential")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    try:
        run_editor(
            input_path=args.input,
            output_directory=args.output_dir,
            prompt=args.prompt,
            max_side=args.max_side,
            seed=args.seed,
            steps=args.steps,
            offload=args.offload,
        )
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
