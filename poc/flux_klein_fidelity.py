import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageOps


MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


def load_input(path: Path, max_side: int) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)

    if max_side < 256 or max_side % 16 != 0:
        raise ValueError("Maximum side must be at least 256 and divisible by 16")

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    scale = min(1.0, max_side / max(image.size))
    width = max(16, math.floor(image.width * scale / 16) * 16)
    height = max(16, math.floor(image.height * scale / 16) * 16)

    if min(width, height) < 256:
        raise ValueError("Prepared image is too small for this evaluation")

    if (width, height) != image.size:
        image = image.resize((width, height), Image.Resampling.LANCZOS)

    return image


DEFAULT_PROMPT = (
    "Conservatively retouch image 1 as an existing real estate photograph. "
    "Only balance exposure, reduce unwanted color casts, and gently improve "
    "tonal clarity. Keep the lighting direction and real shadow patterns "
    "unchanged. Preserve the existing white ceiling, exact wood colors, "
    "green foliage, detailed leaves, glass reflections, window frames, "
    "furniture, walls and all architecture. Do not invent new highlights, "
    "sunlight, shadows, reflections, objects or textures. Do not relocate "
    "or redesign anything. Avoid oversaturation and dramatic contrast. "
    "Make a subtle natural photographic correction, not a new scene."
)

STYLE_REFERENCE_INSTRUCTION = (
    " Image 2 is a separately edited version of the same room. "
    "Use it only as a loose guide for neutral whites and restrained color. "
    "Do not copy any objects, lighting patterns, shadows, or geometry from image 2. "
    "Image 1 remains the sole source of scene content."
)


def validate_strengths(strengths: list[float], steps: int) -> list[float]:
    if steps < 1:
        raise ValueError("Inference steps must be positive")

    if not strengths or len(strengths) > 3:
        raise ValueError("Supply between one and three strengths")

    result = []

    for strength in strengths:
        if not 0.0 < strength <= 1.0:
            raise ValueError("Every strength must be within (0, 1]")

        if steps * strength < 1.0:
            raise ValueError("Strength is too low for the selected inference step count")

        if any(round(value * 100) == round(strength * 100) for value in result):
            raise ValueError("Strength values must produce distinct output filenames")

        result.append(strength)

    return result


def build_prompt(base_prompt: str, has_style_reference: bool) -> str:
    if not base_prompt.strip():
        raise ValueError("The editing prompt cannot be empty")

    return base_prompt + STYLE_REFERENCE_INSTRUCTION if has_style_reference else base_prompt


def prepare_images(
    source_path: Path,
    max_side: int,
    style_reference: Path | None,
) -> tuple[Image.Image, Image.Image, Image.Image | None]:
    image = load_input(source_path, max_side)
    mask = Image.new("L", image.size, 255)
    style = None

    if style_reference is not None:
        style = load_input(style_reference, min(max_side, 512))

    return image, mask, style


def compare_images(source: Image.Image, edited: Image.Image) -> dict[str, float]:
    if source.size != edited.size:
        raise ValueError("FLUX output dimensions differ from the source")

    before = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    after = np.asarray(edited.convert("RGB"), dtype=np.float32) / 255.0
    weights = np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)
    luminance_before = before @ weights
    luminance_after = after @ weights

    return {
        "rgb_change_mae_0_255": round(float(np.mean(np.abs(after - before)) * 255.0), 4),
        "source_brightness_median": round(float(np.median(luminance_before)), 4),
        "edited_brightness_median": round(float(np.median(luminance_after)), 4),
        "source_near_white_fraction": round(float(np.mean(luminance_before > 0.95)), 4),
        "edited_near_white_fraction": round(float(np.mean(luminance_after > 0.95)), 4),
    }


def save_variation(
    source: Image.Image,
    edited: Image.Image,
    output_directory: Path,
    strength: float,
) -> dict:
    if source.size != edited.size:
        raise ValueError("FLUX must preserve the expected image dimensions")

    suffix = f"{round(strength * 100):03d}"
    edit_path = output_directory / f"flux_strength_{suffix}.png"
    diff_path = output_directory / f"changes_strength_{suffix}.png"

    edited.save(edit_path)
    differences = ImageChops.difference(source.convert("RGB"), edited.convert("RGB"))
    ImageEnhance.Contrast(differences).enhance(2.0).save(diff_path)

    return {
        "strength": strength,
        "edited_file": str(edit_path),
        "difference_file": str(diff_path),
        "comparison_metrics": compare_images(source, edited),
    }


def run_experiment(args: argparse.Namespace) -> None:
    import torch

    try:
        from diffusers import Flux2KleinInpaintPipeline
    except ImportError as error:
        raise RuntimeError(
            "Flux2KleinInpaintPipeline is unavailable. Update Diffusers in the isolated .venv-flux."
        ) from error

    strengths = validate_strengths(args.strengths, args.steps)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {args.output_dir}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the active virtual environment")

    image, mask, style = prepare_images(args.input, args.max_side, args.style_reference)
    prompt = build_prompt(args.prompt, style is not None)

    print(f"FLUX fidelity test on {image.size}, GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Strengths: {strengths}; reference: {args.style_reference}", flush=True)

    pipeline = Flux2KleinInpaintPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )

    if args.offload == "sequential":
        pipeline.enable_sequential_cpu_offload()
    else:
        pipeline.enable_model_cpu_offload()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared_input = args.output_dir / "flux_input.png"
    image.save(prepared_input)
    mask.save(args.output_dir / "edit_mask.png")

    results = []

    for strength in strengths:
        print(f"Running image-conditioned editing at strength={strength:.2f}", flush=True)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)

        with torch.inference_mode():
            generated = pipeline(
                prompt=prompt,
                image=image,
                mask_image=mask,
                image_reference=style,
                strength=strength,
                height=image.height,
                width=image.width,
                num_inference_steps=args.steps,
                guidance_scale=1.0,
                generator=generator,
            ).images[0]

        results.append(save_variation(image, generated, args.output_dir, strength))
        del generated
        torch.cuda.empty_cache()

    report = {
        "model": MODEL_ID,
        "pipeline": "Flux2KleinInpaintPipeline",
        "source": str(args.input),
        "prepared_source": str(prepared_input),
        "style_reference": str(args.style_reference) if args.style_reference else None,
        "resolution": list(image.size),
        "steps": args.steps,
        "seed": args.seed,
        "offload": args.offload,
        "prompt": prompt,
        "strengths": results,
        "warnings": [
            "Full-frame inpainting with a white mask is an experimental image-to-image workflow.",
            "Lower strength tends to limit edits but does not guarantee geometry or texture fidelity.",
            "The secondary image may cause copying or hallucination despite style-only instructions.",
            "RGB differences and brightness are descriptive, not tests of object preservation.",
        ],
    }

    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"Results saved to {args.output_dir}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate source-preserving FLUX.2 Klein editing at controlled strengths"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--style-reference", type=Path)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.5, 0.75])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offload", choices=("sequential", "model"), default="sequential")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    try:
        run_experiment(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
