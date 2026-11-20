import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional


LUT_SIZE = 33
LUT_CHANNELS = 3


class ImageAdaptiveLutClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode="bilinear"),
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *self._block(16, 32, normalize=True),
            *self._block(32, 64, normalize=True),
            *self._block(64, 128, normalize=True),
            *self._block(128, 128, normalize=False),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, LUT_CHANNELS, kernel_size=8),
        )

    @staticmethod
    def _block(input_channels: int, output_channels: int, normalize: bool) -> list[nn.Module]:
        layers: list[nn.Module] = [
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2),
        ]

        if normalize:
            layers.append(nn.InstanceNorm2d(output_channels, affine=True))

        return layers

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)


def build_classifier() -> nn.Module:
    return ImageAdaptiveLutClassifier()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_checkpoint_state(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint is not a mapping: {path}")

    return checkpoint


def load_pretrained_model(checkpoint_directory: Path, device: torch.device) -> tuple[nn.Module, torch.Tensor]:
    classifier = build_classifier()
    classifier.load_state_dict(
        load_checkpoint_state(checkpoint_directory / "classifier.pth"),
        strict=True,
    )
    classifier = classifier.to(device).eval()

    saved_luts = load_checkpoint_state(checkpoint_directory / "LUTs.pth")
    bases = []

    for index in range(3):
        identifier = str(index)

        if identifier not in saved_luts:
            raise ValueError(f"Missing pretrained LUT: {identifier}")

        state = saved_luts[identifier]

        if not isinstance(state, dict) or "LUT" not in state:
            raise ValueError(f"Invalid LUT state: {identifier}")

        lut = state["LUT"]

        if not isinstance(lut, torch.Tensor) or tuple(lut.shape) != (
            LUT_CHANNELS,
            LUT_SIZE,
            LUT_SIZE,
            LUT_SIZE,
        ):
            raise ValueError(f"Unexpected pretrained LUT dimensions: {identifier}")

        if not torch.isfinite(lut).all():
            raise ValueError(f"Nonfinite pretrained LUT: {identifier}")

        bases.append(lut.float())

    return classifier, torch.stack(bases, dim=0).to(device)


def infer_adaptive_lut(
    classifier: nn.Module,
    bases: torch.Tensor,
    image: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("Expected one RGB image in NCHW format")

    if tuple(bases.shape) != (3, 3, LUT_SIZE, LUT_SIZE, LUT_SIZE):
        raise ValueError("Expected three pretrained 33-cube RGB LUTs")

    weights = classifier(image).flatten()

    if weights.shape != (3,) or not torch.isfinite(weights).all():
        raise ValueError("Classifier must predict three finite LUT weights")

    combined_lut = torch.einsum("i,icbgr->cbgr", weights, bases)
    return combined_lut, weights


def apply_lut_chunk(image: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("Expected one RGB image in NCHW format")

    if tuple(lut.shape) != (3, LUT_SIZE, LUT_SIZE, LUT_SIZE):
        raise ValueError("Expected an RGB 33-cube LUT")

    # grid_sample usa x=R, y=G e z=B para o volume indexado como [B, G, R].
    coordinates = image.clamp(0.0, 1.0).permute(0, 2, 3, 1)
    coordinates = coordinates.mul(2.0).sub(1.0).unsqueeze(1)
    output = functional.grid_sample(
        lut.unsqueeze(0),
        coordinates,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return output.squeeze(2)


def apply_lut_full_resolution(
    image: np.ndarray,
    lut: torch.Tensor,
    rows_per_chunk: int = 256,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image")

    if image.dtype != np.uint8:
        raise ValueError("Expected 8-bit display RGB pixels")

    if rows_per_chunk <= 0:
        raise ValueError("Chunk height must be positive")

    height, width, _ = image.shape
    result = np.empty_like(image)

    with torch.inference_mode():
        for first_row in range(0, height, rows_per_chunk):
            last_row = min(first_row + rows_per_chunk, height)
            piece = np.ascontiguousarray(image[first_row:last_row])
            tensor = torch.from_numpy(piece).to(
                device=lut.device,
                dtype=torch.float32,
            )
            tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
            adjusted = apply_lut_chunk(tensor, lut)
            adjusted = torch.clamp(adjusted, 0.0, 1.0) * 255.0
            encoded = adjusted.round().byte().squeeze(0).permute(1, 2, 0)
            result[first_row:last_row] = encoded.cpu().numpy()

    return result


def run_benchmark(
    input_path: Path,
    output_directory: Path,
    checkpoint_directory: Path,
    device_name: str,
    rows_per_chunk: int,
) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_directory}")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    device = torch.device(device_name)

    with Image.open(input_path) as source:
        image = np.array(ImageOps.exif_transpose(source).convert("RGB"), dtype=np.uint8, copy=True)

    classifier, bases = load_pretrained_model(checkpoint_directory, device)

    with torch.inference_mode():
        classifier_input = torch.from_numpy(np.ascontiguousarray(image)).to(
            device=device,
            dtype=torch.float32,
        )
        classifier_input = classifier_input.permute(2, 0, 1).unsqueeze(0) / 255.0
        combined_lut, weights = infer_adaptive_lut(
            classifier,
            bases,
            classifier_input,
        )

    output_image = apply_lut_full_resolution(image, combined_lut, rows_per_chunk)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_file = output_directory / "adaptive_lut_enhanced.png"
    Image.fromarray(output_image).save(output_file)

    report = {
        "model": "HuiZeng/Image-Adaptive-3DLUT",
        "weights_source": "pretrained_models/sRGB (paired FiveK)",
        "checkpoint_directory": str(checkpoint_directory),
        "classifier_sha256": hash_file(checkpoint_directory / "classifier.pth"),
        "luts_sha256": hash_file(checkpoint_directory / "LUTs.pth"),
        "input_path": str(input_path),
        "input_sha256": hash_file(input_path),
        "device": str(device),
        "dimensions": list(image.shape[:2]),
        "lut_weights": [float(value) for value in weights.cpu().tolist()],
        "output": str(output_file),
        "warnings": [
            "Global learned 3D LUT, not semantic local retouching",
            "Pretrained on FiveK, not specialized for real-estate imagery",
            "Checkpoint redistribution and dataset rights require separate commercial review",
        ],
    }
    (output_directory / "model_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"Neural editor output: {output_file}", flush=True)
    print(f"Model report: {output_directory / 'model_report.json'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark neural image-adaptive 3D LUT without legacy CUDA extensions"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--rows-per-chunk", type=int, default=256)
    args = parser.parse_args()

    try:
        run_benchmark(
            args.input,
            args.output_dir,
            args.checkpoint_dir,
            args.device,
            args.rows_per_chunk,
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
