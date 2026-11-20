# HDR pipeline v2: EXIF-driven three-RAW experiment

This is an **experimental** high-precision RAW/neural hybrid, not a production-ready HDR real-estate editor. It adds linear-light RAW source protection before tone mapping. It does **not** replace or alter the original AFUNet PoC or the JPEG-level window experiment.

## Prerequisites

- Linux/WSL with Python 3.12 and an existing environment from `requirements.txt`
- ExifTool installed and available on PATH
- The original AFUNet checkout and Kalantari checkpoint already present at `$HOME/AFUNet`
- A working CUDA/PyTorch configuration (the tested WSL setup uses PyTorch 2.11.0 + CUDA 12.8, RTX 4060)
- Exactly three supported camera RAW files in `photos/`, representing the **same composition** with distinct bracket exposures

The script handles .ARW, .CR2, .CR3, .DNG, .NEF, .ORF, .RAF, .RW2 and .SRW extensions (case-insensitive). Other JPEG/TIFF files are ignored. If four or more camera RAW files are in the folder, the script rejects the input to prevent accidental mixing. The three RAW names do not need to be renamed.

## Run

```bash
cd /opt/hdr-real-state
git pull --ff-only
source .venv/bin/activate

python -m py_compile poc/hdr_pipeline_v2.py
python -m unittest discover -s tests -v

time python poc/hdr_pipeline_v2.py \
  --photos photos \
  --afunet-source "$HOME/AFUNet" \
  --checkpoint "$HOME/AFUNet/pretrain_model/Kalantari.pth" \
  --output-dir output/v2-new-photos \
  --tile-size 128 \
  --overlap 32

explorer.exe "$(wslpath -w /opt/hdr-real-state/output/v2-new-photos)"
```

Use a **new** output folder name for every run; the script refuses to overwrite files in a nonempty destination. The process automatically orders the exposures by EXIF `shutter * ISO / aperture²`, so the filenames can be arbitrary. It prints each chosen RAW and the relative factors. It rejects unusually small exposure ranges.

## Output files

- `raw_under.jpg`, `raw_reference.jpg`: unadjusted previews of the RAW exposures
- `conventional_hdr.tiff`: conventional exposure-weighted linear RAW baseline
- `conventional_local.jpg`: locally tone-mapped conventional baseline
- `neural_original_hdr.tiff`: AFUNet float32 output before new calibration/blending
- `neural_local.jpg`: neural image calibrated to baseline midtone scale with local tone mapping
- `hybrid_hdr.tiff`: calibrated neural HDR combined **in float32 linear-light working data** with radiance estimated from the dark RAW where the mask applies
- `hybrid_preview.jpg`: the experimental combined final JPEG preview
- `highlight_mask.png`: the selection to inspect for leakage/false detections
- `report.json`: EXIF values, exposure ratios, CUDA availability, neural normalization factor, highlight coverage and remaining limitations

The original underexposed RAW pixels are not composited into the output JPEG. Their relative radiance contribution is combined before the final tone mapping.

## Essential limitations

- No geometric alignment/deghosting is implemented. The camera must be stationary, and misalignment may produce color fringes or doubled window frames.
- The AFUNet's benchmark training input differs from this PoC's RAW development. A **scalar midtone normalization does not ensure correct camera color or true HDR linear-radiance calibration**. The hybrid TIFF is a diagnostic scene-like working space, not guaranteed physically calibrated scene radiance.
- The highlight mask is based on reference-versus-underexposed brightness differences and area filtering. It may misidentify ceiling lights, sunlit walls, mirrors or non-window objects. Inspect `highlight_mask.png` every time.
- The local tone mapping is a prototype, not a tuned professional real-estate color workflow. It may be too flat, harsh or produce halos.
- The program does not use the already-produced TIFF from previous photos; every new group launches a new AFUNet inference.
- Real camera RAW end-to-end operation of **v2** is pending the first local WSL run. Synthetic unit tests do not establish photographic image quality.

## Quality gate

Compare `conventional_local.jpg`, `neural_local.jpg` and `hybrid_preview.jpg`. Check the exterior window view, neutral walls, shadow detail, color balance, haloing and double edges at full resolution. Send those three files together with `highlight_mask.png` and `report.json` for review. If v2 fails, keep the raw files and outputs; future fixes must be based on this evidence rather than overwriting the initial models and previews.
