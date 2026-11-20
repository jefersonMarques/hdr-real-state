import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageEnhance


MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
REFERENCE_LABELS = ("under", "reference", "over")
DEFAULT_PROMPT = (
    "Create one professionally retouched real estate photograph using the "
    "THREE different exposure captures of the EXACT SAME scene. "
    "Image 1 is the dark exposure: use its real window and exterior details, "
    "including individual tree leaves, branches, reflections and sky. "
    "Image 2 is the middle exposure and is the authority for ALL geometry, "
    "camera viewpoint, furniture, material colors and physical structure. "
    "Image 3 is the bright exposure: use its real interior shadow detail. "
    "Combine information from these three exposures into a single balanced "
    "natural photographic result. Maintain neutral white walls and ceiling, "
    "realistic wood tones and restrained natural greenery. "
    "Keep existing light direction and physically plausible shadow patterns. "
    "Do not add, move, erase or redesign objects. Do not invent highlights, "
    "sunbeams, reflections, textures, exterior scenery or fine leaf details. "
    "Do not create a collage, grid or split screen. Output ONE final image "
    "with the same framing and perspective as Image 2."
)


def load_references(directory: Path) -> tuple[list[Image.Image], dict]:
    manifest_path = directory / "triplet.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("inputs")

    if not isinstance(records, list) or len(records) != 3:
        raise ValueError("A manifest containing exactly three exposures is required")

    roles = [record.get("role") for record in records]

    if roles != list(REFERENCE_LABELS):
        raise ValueError("The exposure order must be under, reference, over")

    references = []

    for label in REFERENCE_LABELS:
        path = directory / f"{label}.png"

        if not path.is_file():
            raise FileNotFoundError(path)

        with Image.open(path) as source:
            references.append(source.convert("RGB").copy())

    dimensions = [image.size for image in references]

    if len(set(dimensions)) != 1:
        raise ValueError("All three references must have identical image dimensions")

    if tuple(manifest.get("prepared_dimensions", [])) != dimensions[0]:
        raise ValueError("The manifest dimensions do not match the reference images")

    return references, manifest


def quality_diagnostics(reference: Image.Image, generated: Image.Image) -> dict:
    if reference.size != generated.size:
        raise ValueError("Output dimensions differ from the reference exposure")

    original = np.asarray(reference, dtype=np.float32)
    output = np.asarray(generated, dtype=np.float32)
    luma_weights = np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)

    reference_luma = (original / 255.0) @ luma_weights
    output_luma = (output / 255.0) @ luma_weights

    return {
        "rgb_change_mae_0_255": round(float(np.mean(np.abs(output - original))), 4),
        "reference_near_white_fraction": round(float(np.mean(reference_luma > 0.95)), 4),
        "output_near_white_fraction": round(float(np.mean(output_luma > 0.95)), 4),
        "reference_brightness_median": round(float(np.median(reference_luma)), 4),
        "output_brightness_median": round(float(np.median(output_luma)), 4),
    }


def save_outputs(
    directory: Path,
    reference: Image.Image,
    generated: Image.Image,
    manifest: dict,
    parameters: dict,
) -> None:
    if reference.size != generated.size:
        raise ValueError("FLUX output has unexpected geometry")

    directory.mkdir(parents=True, exist_ok=True)
    output_png = directory / "neural_triplet.png"
    output_jpeg = directory / "neural_triplet_preview.jpg"
    generated.save(output_png)
    generated.save(output_jpeg, format="JPEG", quality=98, subsampling=0, optimize=False)

    difference = ImageChops.difference(reference, generated)
    ImageEnhance.Contrast(difference).enhance(2.0).save(directory / "changes.png")

    report = {
        "model": MODEL_ID,
        "method": "FLUX multi-reference conditioning; not a validated HDR fusion model",
        "bracket_manifest": manifest,
        "inference_parameters": parameters,
        "output": str(output_png),
        "metrics_against_middle_exposure": quality_diagnostics(reference, generated),
        "warnings": [
            "Model may synthesize non-existent lighting, structures, leaves or scenery.",
            "No alignment, deghosting or pixel-level physical HDR consistency.",
            "The preview is limited to neural inference resolution, not RAW resolution.",
            "JPEG at quality 98 does not recreate missing source detail.",
            "Professional quality and commercial photo fidelity have not been established.",
        ],
    }
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_inference(
    source_directory: Path,
    output_directory: Path,
    steps: int,
    seed: int,
    offload: str,
    prompt: str,
) -> None:
    if steps < 1 or not prompt.strip():
        raise ValueError("A valid prompt and positive inference step count are required")

    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_directory}")

    import torch
    from diffusers import Flux2KleinPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the active FLUX environment")

    references, manifest = load_references(source_directory)
    width, height = references[1].size
    print(f"Three RAW-derived reference exposures: {width}x{height}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)} | offload: {offload}", flush=True)

    pipeline = Flux2KleinPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )

    if offload == "sequential":
        pipeline.enable_sequential_cpu_offload()
    else:
        pipeline.enable_model_cpu_offload()

    generator = torch.Generator(device="cuda").manual_seed(seed)

    with torch.inference_mode():
        result = pipeline(
            image=references,
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=1.0,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]

    parameters = {
        "steps": steps,
        "seed": seed,
        "offload": offload,
        "prompt": prompt,
        "dimensions": [width, height],
        "reference_order": list(REFERENCE_LABELS),
    }
    save_outputs(output_directory, references[1], result, manifest, parameters)
    print(f"Neural triplet result: {output_directory / 'neural_triplet.png'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experimental FLUX neural image generation conditioned on three camera exposures"
    )
    parser.add_argument("--triplet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offload", choices=("sequential", "model"), default="sequential")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    try:
        run_inference(
            args.triplet_dir,
            args.output_dir,
            args.steps,
            args.seed,
            args.offload,
            args.prompt,
        )
    except (OSError, ValueError, RuntimeError, ImportError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
