# V5 — source-faithful three-RAW HDR baseline

Status: **implemented as a separate experimental pipeline on `main`; synthetic WSL camera tests and image-quality approval are still pending.** This version uses no neural inference. The user's request to make AI learn from the three original RAWs remains a separate model-research goal, and this classical baseline does not claim to satisfy it.

## Why V5 exists

The prior V4 FLUX multi-reference experiment produced a recognizable edited photograph but remained too dark and invented spatial illumination. Giving FLUX three reference pictures is not equivalent to calibrated sensor-linear HDR fusion. V5 makes a stronger **source-faithful photographic baseline** so that future exposure-aware neural fusion and editing can be measured against a physically plausible output. It does not feed a pre-existing HDR JPEG into a model.

## Processing

```text
photos/{three camera RAWs}
  -> read EXIF exposure = shutter * ISO / aperture^2
  -> order under / reference / over
  -> shared camera WB from reference RAW
  -> LibRaw linear-ish sRGB development, gamma=(1,1), 16-bit
  -> float32 under/reference/over .npy (original camera resolution)
  -> preview-based translation registration by phase correlation
  -> source weight confidence (shadow and highlight clipping)
  -> exposure-normalized radiance merge + reference-based mismatch suppression
  -> merged_hdr.npy (float32)
  -> locally adapted luminance tone map with edge-aware guidance
  -> tone_mapped_preview.jpg + tone_mapped_16bit.tiff
  -> photo_final.jpg + photo_final_16bit.tiff + report
```

The term **linear RGB** means LibRaw-developed, linear-tone RGB in sRGB primaries, not an unprocessed Bayer mosaic and not precisely calibrated scene radiance. Exposure ratios rely on EXIF and assume stable camera behavior; different ISO sensitivity and camera response may require calibration. Genuine photographic source authenticity still needs visual checks.

The registration algorithm handles **translations**, not lens distortion, perspective changes, rotating cameras or moving leaves. Moving-object suppression is a simple discrepancy test against the reference frame. A failed registration aborts rather than silently blending badly aligned frames. The `--alignment none` escape hatch must only be used for confidently aligned tripod captures.

The local tone map operates on an HDR float32 merge and balances nearby luminance with an edge-aware estimate. It does not semantically segment windows or reproduce Photoshop's complete retouching work, and may create halos. No generative/model-synthesized room content is introduced by V5.

## Files and commands

The V5 code resides in:
- `poc/v5_core.py`: full implementation, with source preservation and chunked HDR merge
- `poc/prepare_triplet_linear_v5.py`: camera RAW input and EXIF/WB preprocessing
- `poc/merge_hdr_v5.py`: aligned float32 radiance fusion
- `poc/tone_map_v5.py`: source-based local tone mapping to JPEG/TIFF
- `poc/export_v5.py`: final full-resolution JPEG and TIFF export
- `poc/benchmark_v5.py`: side-by-side and descriptive metrics
- `tests/test_v5_pipeline.py`: synthetic regression tests

All stages save into new output locations and refuse to overwrite their expected outputs. Existing photos and v2-v4 outputs remain intact.

### 1. Checkout and environment

```bash
cd /opt/hdr-real-state
git pull --ff-only
source .venv/bin/activate

python -c "import cv2, rawpy, numpy, tifffile, PIL; print('V5 dependencies ready')"
python -m unittest discover -s tests -p 'test_v5_pipeline.py' -v
```

If `cv2` is absent, install it into **only** `.venv`:

```bash
python -m pip install --no-deps "opencv-python-headless==4.11.0.86"
```

This project uses the same version during V5 PoC validation. Do not reinstall CUDA or overwrite the separate `.venv-flux` environment.

### 2. Develop original RAWs into linear RGB

```bash
python poc/prepare_triplet_linear_v5.py \
  --photos photos \
  --output-dir output/v5-01
```

This stage requires **exactly three RAW files**, `exiftool`, a shared camera white balance and consistent dimensions.

Expected `output/v5-01`:
- `under_linear.npy`, `reference_linear.npy`, `over_linear.npy`: original-resolution float32 RGB arrays
- `triplet_v5.json`: originals, EXIF, exposure ratios and shared WB

### 3. Actual RAW-derived HDR radiance merge

```bash
python poc/merge_hdr_v5.py \
  --triplet output/v5-01/triplet_v5.json \
  --output-dir output/v5-01 \
  --alignment translation \
  --rows-per-chunk 256
```

Produces:
- `merged_hdr.npy`: original-size float32 exposure-normalized HDR
- `merge_report.json`: measured frame translations, confidence and limitations

If registration reports poor confidence, inspect the bracket geometry and do not ignore the error. Only use `--alignment none` when there is good evidence that the original camera was stationary and the images match pixel-to-pixel.

### 4. Render a final preview and 16-bit TIFF

```bash
python poc/tone_map_v5.py \
  --source output/v5-01/merged_hdr.npy \
  --output-dir output/v5-01
```

This generates:
- `tone_mapped_preview.jpg`: preview quality 96, full source dimensions
- `tone_mapped_16bit.tiff`: actual 16-bit-per-channel RGB from float HDR, lossless Deflate, sRGB ICC
- `tone_map_report.json`

### 5. Export deliverable files

```bash
python poc/export_v5.py \
  --hdr output/v5-01/merged_hdr.npy \
  --tone-mapped output/v5-01/tone_mapped_16bit.tiff \
  --output-dir output/v5-01/final
```

Produces `photo_final.jpg` (JPEG quality 98, chroma 4:4:4, embedded sRGB profile), `photo_final_16bit.tiff` (lossless 16-bit with ICC) and `export_report.json`.

The TIFF contains true 16-bit tonal values derived from the float32 HDR. It does **not** restore data clipped in every RAW or guarantee lossless original camera color handling.

### 6. Side-by-side comparison with professional retouch

```bash
python poc/benchmark_v5.py \
  --before output/v4-triplet-input-02/reference.png \
  --after output/v5-01/final/photo_final.jpg \
  --reference photos/DSC01858.jpg \
  --output-dir output/v5-01/review

explorer.exe "$(wslpath -w /opt/hdr-real-state/output/v5-01/review)"
```

The benchmark resizes images for approximate RGB metrics, **without registration**. It cannot prove photographic fidelity. It produces `comparison.jpg` and `report.json`.

## Expected visual quality gate

Check the final image at 100% zoom, not only a contact sheet:
1. Is the exterior foliage genuinely present with the same leaf silhouettes as the darkest RAW?
2. Is the interior bright, but without artificial shafts of sunlight or geometrically modified shadows?
3. Are ceiling and walls neutral? Are wood and exterior greens accurate, without unnatural saturation?
4. Are window-frame, curtains and furniture edges undoubled? If not, inspect `merge_report.json` for alignment.
5. Are transitions between window and interior free of strong halos?
6. Are JPEG and TIFF the camera's native pixel dimensions? Are large smooth areas free of banding?

The V5 baseline should beat the earlier conventional HDR in visually judged photography; **this has not yet been demonstrated**. Do not approve a model or a product based only on passing synthetic tests. If the baseline does not meet the objective, analyze color transforms, exposure calibration, registration and local tone mapping using the original RAWs, with photographic evidence.

## Next AI stage

Once there is a reliable baseline, train or evaluate a model that actually sees **three registered, exposure-calibrated source frames and shutter/ISO factors**, then predicts *how to fuse/expose/edit* them. The physically derived V5 HDR is a competing baseline and an optional training feature, **not** a JPEG that the model must fix. User-approved source/Photoshop paired datasets remain necessary for professional real-estate retouch quality.

Development remains exclusively on `main` with no branches, PRs or Actions.
