import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def load_raw(path: Path) -> np.ndarray:
    import rawpy

    if not path.is_file():
        raise FileNotFoundError(path)

    with rawpy.imread(str(path)) as raw_image:
        image = raw_image.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=16,
            gamma=(2.222, 4.5),
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image: {path}")

    return image.astype(np.float32) / 65535.0


def load_afunet(source_directory: Path, checkpoint_path: Path, device: torch.device):
    if not (source_directory / "models" / "AFUNet.py").is_file():
        raise FileNotFoundError("AFUNet source checkout not found")

    if not checkpoint_path.is_file():
        raise FileNotFoundError("AFUNet checkpoint not found")

    sys.path.insert(0, str(source_directory.resolve()))

    try:
        from models.AFUNet import AFUNet
    finally:
        sys.path.remove(str(source_directory.resolve()))

    model = AFUNet(
        img_size=(40, 40),
        in_chans=18,
        window_size=8,
        img_range=1.0,
        drop_path_rate=0.1,
        depths=[5, 5, 5, 5],
        embed_dim=72,
        num_heads=[4, 4, 4, 4],
        mlp_ratio=2,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    weights = {name.removeprefix("module."): value for name, value in state.items()}
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model.to(device)


def tile_origins(length: int, size: int, overlap: int) -> list[int]:
    if length <= 0 or size <= 0 or not 0 <= overlap < size:
        raise ValueError("Invalid tile dimensions")

    if length <= size:
        return [0]

    starts = list(range(0, length - size + 1, size - overlap))
    final = length - size

    if starts[-1] != final:
        starts.append(final)

    return starts


def prepare_features(images: tuple[np.ndarray, ...], exposure_times: tuple[float, ...]) -> torch.Tensor:
    features = []

    for image, exposure_time in zip(images, exposure_times, strict=True):
        ldr = np.clip(image, 0.0, 1.0)
        hdr_estimate = np.power(ldr, 2.2) / exposure_time
        combined = np.concatenate((hdr_estimate, ldr), axis=2)
        features.append(np.moveaxis(combined, 2, 0))

    stacked = np.concatenate(features, axis=0)
    return torch.from_numpy(np.ascontiguousarray(stacked)).unsqueeze(0)


def reconstruct(
    images: tuple[np.ndarray, np.ndarray, np.ndarray],
    exposure_times: tuple[float, float, float],
    model,
    device: torch.device,
    tile_size: int,
    overlap: int,
) -> np.ndarray:
    if len({image.shape for image in images}) != 1:
        raise ValueError("Brackets must have identical dimensions")

    if tile_size % 32 != 0:
        raise ValueError("Tile size must be a multiple of 32")

    height, width, channels = images[0].shape

    if channels != 3:
        raise ValueError("Brackets must be RGB images")

    accumulated = np.zeros((height, width, 3), np.float32)
    accumulated_weights = np.zeros((height, width, 1), np.float32)

    for y in tile_origins(height, tile_size, overlap):
        for x in tile_origins(width, tile_size, overlap):
            bottom = min(y + tile_size, height)
            right = min(x + tile_size, width)
            valid_height = bottom - y
            valid_width = right - x
            patches = []

            for image in images:
                patch = image[y:bottom, x:right]
                padded = np.pad(
                    patch,
                    ((0, tile_size - valid_height), (0, tile_size - valid_width), (0, 0)),
                    mode="edge",
                )
                patches.append(padded)

            features = prepare_features(tuple(patches), exposure_times).to(device)

            with torch.inference_mode():
                output = model(features)

            if output.shape != (1, 3, tile_size, tile_size):
                raise ValueError("Unexpected AFUNet output dimensions")

            prediction = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            prediction = prediction[:valid_height, :valid_width]
            vertical = np.hanning(valid_height) if valid_height > 1 else np.ones(1)
            horizontal = np.hanning(valid_width) if valid_width > 1 else np.ones(1)
            weight = (0.1 + 0.9 * np.outer(vertical, horizontal))[..., None]
            accumulated[y:bottom, x:right] += prediction * weight
            accumulated_weights[y:bottom, x:right] += weight

    result = accumulated / np.maximum(accumulated_weights, 1e-8)

    if not np.isfinite(result).all():
        raise ValueError("Neural reconstruction produced non-finite pixels")

    return np.maximum(result, 0.0)


def save_images(result: np.ndarray, output: Path) -> None:
    from PIL import Image
    from tifffile import imwrite

    hdr_path = output.with_suffix(".tiff")
    preview_path = output.with_name(output.name + "_preview.jpg")

    if hdr_path.exists() or preview_path.exists():
        raise FileExistsError("Output files already exist")

    output.parent.mkdir(parents=True, exist_ok=True)
    imwrite(hdr_path, result.astype(np.float32), photometric="rgb")
    white_point = max(float(np.percentile(result, 99.5)), 1e-6)
    normalized = result / white_point
    mapped = normalized / (0.65 + normalized)
    preview = np.uint8(np.round(np.power(np.clip(mapped, 0.0, 1.0), 1.0 / 2.2) * 255))
    Image.fromarray(preview).save(preview_path, quality=94)
    print(hdr_path)
    print(preview_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Experimental three-RAW neural HDR evaluation")
    parser.add_argument("--under", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--over", type=Path, required=True)
    parser.add_argument("--afunet-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ev-step", type=float, default=2.0)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=32)
    args = parser.parse_args()

    if not 0.0 < args.ev_step <= 6.0:
        parser.error("EV step must be between zero and six")

    if len({args.under, args.reference, args.over}) != 3:
        parser.error("Three distinct exposures are required")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = (load_raw(args.under), load_raw(args.reference), load_raw(args.over))
        model = load_afunet(args.afunet_source, args.checkpoint, device)
        result = reconstruct(
            images,
            (2.0 ** (-args.ev_step), 1.0, 2.0 ** args.ev_step),
            model,
            device,
            args.tile_size,
            args.overlap,
        )
        save_images(result, args.output)
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
