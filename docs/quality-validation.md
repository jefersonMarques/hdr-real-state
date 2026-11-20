# HDR quality validation

## Scope and evidence

The first user-run AFUNet evaluation on Sony ARW brackets completed on an NVIDIA RTX 4060 under WSL with PyTorch CUDA 12.8. The resulting JPEG preview of a bedroom was heavily underexposed, while the window was nearly white. This **does not prove** that the neural HDR TIFF lost window detail: the original preview uses a separate, simplistic tone mapper.

AFUNet upstream expects three prepared LDR images and relative exposure factors. The existing PoC develops RAW images via rawpy, then derives eighteen model channels. The RAW developing settings, model's training distribution, source alignment, exposure calibration and preview mapping remain unvalidated.

## Diagnostic workflow

The new tool deliberately **does not change** `poc/hdr_afunet.py`. It:

1. Reads shutter, ISO and aperture values using ExifTool and computes effective exposure factors relative to the middle image: `shutter_time * ISO / aperture^2`.
2. Develops the three RAW files to 16-bit **linear RGB** using camera white balance and disables automatic brightness.
3. Produces an exposure-weighted, unclipped conventional HDR reference. This is a simple stationary-camera baseline, not a professional exposure-fusion editor.
4. Applies **the same tone-mapping function** independently to the conventional HDR and the already-generated AFUNet floating-point TIFF.
5. Produces source previews, a side-by-side comparison and descriptive luminosity statistics. An optional professionally edited JPEG can be added as another panel.

The algorithm uses a luminance-based auto-exposure target and a global highlight-compressing tone curve; the resulting previews are for diagnosis, not production editing.

## Run on WSL with the existing images

```bash
cd /opt/hdr-real-state
git pull --ff-only
source .venv/bin/activate
python -m unittest discover -s tests -v

python poc/hdr_diagnostics.py \
  --under photos/DSC02055.ARW \
  --reference photos/DSC02054.ARW \
  --over photos/DSC02056.ARW \
  --neural-hdr output/teste2.tiff \
  --neural-preview output/teste2_preview.jpg \
  --output-dir output/review-test2
```

If a professional reference image is available, add `--final-reference photos/final.jpg`, adjusting the file name.

ExifTool is required on PATH; on Ubuntu install it via `sudo apt install libimage-exiftool-perl`. The Python packages are already covered by `requirements.txt`. No neural inference or PyTorch GPU run is required for this review.

## Output

```text
output/review-test2/
  under.jpg
  reference.jpg
  over.jpg
  conventional_hdr.tiff
  conventional_preview.jpg
  neural_remapped.jpg
  comparison.jpg
  metrics.json
```

The optional neural original-preview and professional-reference pictures are read from their original paths and added to the comparison sheet; they are not copied into the output directory.

In `metrics.json`, `brightness_p05/p50/p95` are image-wide percentiles of encoded RGB brightness (not calibrated scene luminance), while `dark_pixel_fraction` and `near_white_pixel_fraction` indicate how much of the preview lies near either extreme. **These metrics do not measure reconstruction accuracy**, detail, ghosting, or similarity to a professional edit.

For a fair qualitative judgement, inspect corresponding regions of the window, white wall, ceiling and carpet at 100% magnification, not only the downsampled contact sheet. Color, alignment, tonal character and contrast still require assessment.

## Caveats and next decision gate

- **No registration or deghosting** is performed by this conventional baseline. Camera motion can create double edges, and people/objects can appear ghosted.
- Mixed interior lighting, camera-color calibration and local neutral-wall balance have not been corrected.
- The two TIFF images may have different output scale and color-transfer characteristics. A common preview tone mapper equalizes part of the visualization but **does not make the underlying neural and conventional radiance values directly comparable**.
- No quantitative PSNR/SSIM is calculated against a manually edited JPEG because the reference may be geometrically altered, differently cropped and color graded.
- Real ARW end-to-end operation of this new baseline must be confirmed on the user's WSL; automated tests use synthetic arrays.
- Model training data uses LDR inputs. A wrong RAW-to-LDR mapping, camera response, exposure factor or missing alignment may cause a genuine neural reconstruction problem; a better JPEG curve alone cannot fix it.

**Decision gate:** if the conventional baseline reveals window detail but the neural remap cannot, prioritize the neural input contract and source alignment. If both TIFFs retain detail but the original neural JPEG is dark, prioritize tone mapping/exposure control. If neither retains window detail, inspect the original underexposed RAW and the demosaicing/highlight handling first.

## Stage 0.2: highlight recovery experiment

The supplied images confirm that the dark RAW preview contains true outdoor scene detail (roof, tree, fence and sky). The conventional merged HDR TIFF retains large relative radiance values in the window, which the global tone mapper compresses toward white. The neural re-mapped JPEG is also nearly white around the window. **The neural float TIFF was not provided for direct examination**, so neural reconstruction loss versus preview loss remains undecided.

A separate diagnostic post-processor at `poc/hdr_window_recovery.py` tests an exposure-aware highlight mask. It compares the bright reference preview with the underexposed RAW preview, filters out small disconnected highlights (such as ceiling lights) and locally preserves actual pixels from the underexposed source. It performs linear-light compositing with a small adjustable exposure gain.

Run **after** `hdr_diagnostics.py` has generated the five preview files:

```bash
python poc/hdr_window_recovery.py \
  --under-preview output/review-test2/under.jpg \
  --reference-preview output/review-test2/reference.jpg \
  --conventional-preview output/review-test2/conventional_preview.jpg \
  --neural-preview output/review-test2/neural_remapped.jpg \
  --output-dir output/review-test2/window-test
```

Files:
- `window_comparison.jpg`: conventional and neural previews before and after preserving source highlights
- `conventional_window_recovered.jpg` and `neural_window_recovered.jpg`: separate full-resolution previews
- `highlight_mask.jpg`: actual mask used for compositing, important for identifying false selections or halos

The mask thresholding was tuned only on **one** uploaded real-estate bracket and **must not** be considered robust for every camera, sky, window shape or interior. No camera-motion registration/deghosting is present. When exposure misalignment exists, window edges can double. The preview compositing is loss of dynamic range (JPEG-domain) and does **not** replace color-managed HDR radiance reconstruction; it only tests whether a source-guided highlight path restores real details.

No model weights, AFUNet implementation or prior neural outputs are changed. Confirm improvements at 100% resolution and inspect the mask before proceeding to calibrated RAW-domain highlight protection and neural model-input correction.
