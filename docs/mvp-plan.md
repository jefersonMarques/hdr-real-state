# MVP plan

## Product targets

- Windows desktop application with an offline installer.
- Three camera RAW inputs for each interior shot, usually -2 EV, 0 EV, +2 EV.
- Neural HDR reconstruction is mandatory for the production path.
- Reliable white balance and local exposure correction for mixed interior lighting.
- Preserve genuine window details, color fidelity and architectural geometry.
- No mandatory cloud APIs or image uploads.
- Acceleration on compatible AMD/NVIDIA/Intel cards; CPU fallback.

## Stage 0: neural proof of concept (committed)

The script `poc/hdr_afunet.py` is an **experimental neural model evaluation tool**, not an editor or AutoHDR-equivalent application. It expects AFUNet's original source and checkpoint in local directories. No pretrained weights or third-party source are included.

On a computer with Python and PyTorch installed:

1. Install the dependencies using `python -m pip install -r requirements.txt`.
2. Obtain AFUNet source and checkpoint from its official GitHub repository after checking their terms.
3. Execute with your own three RAWs. Example:

```powershell
python poc/hdr_afunet.py --under "C:\photos\room_-2.CR3" --reference "C:\photos\room_0.CR3" --over "C:\photos\room_+2.CR3" --afunet-source "C:\projects\AFUNet" --checkpoint "C:\projects\AFUNet\pretrain_model\Kalantari.pth" --output "C:\photos\output\room"
```

Run synthetic unit tests with `python -m unittest discover -s tests -v`.

## Important limitations of stage 0

- RAW processing uses camera white balance via rawpy; **there is no neural mixed-light white-balance model yet**.
- Bracket exposures are assumed to be evenly spaced, default 2 EV. Exact EXIF values are not yet read.
- A simple preview tone map is used. The output is not a production real-estate edit.
- Neural inference is tiled, but complete RAW images still occupy system RAM.
- No production alignment calibration, photographic quality benchmark, batch mode, GUI, ONNX conversion or installer.
- The AFUNet checkpoint completed local inference on Sony ARW brackets with an NVIDIA RTX 4060, but photographic quality failed the first evaluation.
- GPU acceleration is only enabled where PyTorch reports CUDA availability, including supported ROCm environments; otherwise the experiment uses CPU.
- Real full-resolution output and installer compatibility have not been verified.

## Stage 0.1: reproducible quality diagnosis (committed)

The first real Sony ARW/RTX 4060 AFUNet experiment finished and produced a float32 HDR image, but its original JPEG preview was strongly underexposed with minimal visible outdoor detail. This is a failed **image quality** gate, not a failed CUDA or file-processing test.

The diagnostic path is described in [quality-validation.md](quality-validation.md). It uses actual EXIF shutter speed, ISO and aperture to compute bracket exposure ratios, produces a linear camera-RAW exposure-weighted conventional HDR, re-renders the existing AFUNet TIFF with the same preview curve and creates a side-by-side comparison with image statistics.

There is no additional neural inference. The diagnostic script has synthetic unit tests, but its real-RAW baseline output still needs validation on the local WSL images. This baseline is not an alternative final product: neural processing remains mandatory for the intended application.

**Next decision:** inspect whether window detail exists in the underexposed RAW and conventional merge; assess the new neural re-map separately from its original JPEG; then fix model input calibration/alignment or develop production tone mapping based on the evidence.

## Stage 1: model and quality validation

Test 30–50 real interiors with three RAW files each. Compare AFUNet and HDR-Transformer to professionally edited outputs. Validate deghosting, mixed white balance, window detail and wall neutrality.

## Stage 2: offline desktop product

Choose and license a suitable model, implement the Qt GUI, automatic bracket grouping, color management, neural white-balance enhancement, ONNX acceleration, batch exports and a tested Windows installer with bundled weights.

## Current gate: neural model qualification

The real-estate HDR v2 results have not established a neural quality advantage. Further tuning of the hybrid preview is paused pending a model-only comparison of the three upstream AFUNet checkpoints (Kalantari, Hu, Tel) versus the conventional HDR, with the same output tone mapper. The benchmark tool, instructions, known caveats and decision criteria are in [neural-qualification.md](neural-qualification.md). Only after this comparison should another model architecture, camera-to-LDR input calibration or specialized fine-tuning be considered. No pretrained-model superiority is claimed before local results are submitted.
