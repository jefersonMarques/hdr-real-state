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
- The real AFUNet checkpoint has not yet been tested against user-supplied three-camera RAWs.
- GPU acceleration is only enabled where PyTorch reports CUDA availability, including supported ROCm environments; otherwise the experiment uses CPU.
- Real full-resolution output and installer compatibility have not been verified.

## Stage 1: model and quality validation

Test 30–50 real interiors with three RAW files each. Compare AFUNet and HDR-Transformer to professionally edited outputs. Validate deghosting, mixed white balance, window detail and wall neutrality.

## Stage 2: offline desktop product

Choose and license a suitable model, implement the Qt GUI, automatic bracket grouping, color management, neural white-balance enhancement, ONNX acceleration, batch exports and a tested Windows installer with bundled weights.
