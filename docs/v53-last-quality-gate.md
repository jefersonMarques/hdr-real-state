# V5.3 — final source-preserving photography quality trial

Status: **experimental implementation on `main`; source-file checks are complete, but actual Sony ARW processing and full photographic quality verification are pending on the user's WSL.**

## Decision: one focused trial, not more version churn

The last kitchen benchmark (professional editor versus V5.2) had inadequate exterior clarity, washed indoor contrast and poor integration around window edges. The V5.2 native-sensor confidence masks were valuable diagnostics but changed the V5.1 output by only about **0.44 RGB levels / 255** in the prior user-supplied comparison.

A repeat of the same merge with different midtone constants is unlikely to meet the photographic target. V5.3 is a **different, independently testable and reversible source-composite experiment**.

**This experiment is NOT trained neural AI.** It makes a deterministic, source-faithful composition from the three RAW-derived linear frames. This is intentional: native real leaf geometry and structural details should come from the camera, not a generative model. It does not guarantee professional equivalence.

## What changes and why

```text
Original under.ARW (-1 EV) -> original 16-bit linear RAW development
Original reference.ARW (0 EV) -> original 16-bit linear RAW development
Original over.ARW (+1 EV) -> original 16-bit linear RAW development
                       |
                       v
Same three independent developed frames in triplet_v5.json
                       |
                       v
Translation registration (same established V5 algorithm)
                       |
                       v
Explicit per-pixel selection from real exposures:
  bright/reference clipped -> prioritize valid under-exposed detail
  neutral midtones -> prioritize middle exposure geometry
  weak shadows -> prioritize valid over-exposed signal
  detected structural disagreement -> prefer reliable middle exposure
                       |
                       v
Full-resolution float32 camera-derived merged_hdr.npy
                       |
                       v
Detect bright exterior from reference + under source evidence
  -> connected area validation + guided-filter edge-aware mask
  -> mask overlay JPEG for REQUIRED visual inspection
                       |
                       v
Log-luminance local tone map:
  -> separate base versus high-frequency detail
  -> controlled window-target brightness without broad glow
  -> moderate shadow toe and local contrast
  -> no scene synthesis or made-up lighting
                       |
                       v
Native-resolution JPEG quality 98, 4:4:4, sRGB ICC
Native-resolution TIFF RGB 16-bit with sRGB ICC and lossless Deflate
```

This uses the **existing independently prepared three camera RAW frames**, not either old HDR output. In other words, the V5 manifest's `under_linear.npy`, `reference_linear.npy`, and `over_linear.npy` must correspond to the three original ARWs, not to a previously composited or gamma-encoded image. New training or model inference is not performed.

## New files

- `poc/source_fusion_v53.py`: complete, opt-in source-selective merge and tone rendering
- `tests/test_source_fusion_v53.py`: nine synthetic tests for real-exposure selection, sharp window mask, no gross spill, source provenance, native JPEG/TIFF export
- `docs/v53-last-quality-gate.md`: this guide

Earlier V5/V5.1/V5.2 code and results are untouched. The new algorithm uses only one fresh output folder. No branches, PRs or Actions.

## Run with the existing kitchen RAWs

```bash
cd /opt/hdr-real-state
git pull --ff-only
source .venv/bin/activate

python -m unittest discover -s tests -p 'test_source_fusion_v53.py' -v
```

**Do not proceed if these nine tests fail.** Share the failure details. The tests are synthetic and do not imply a camera-photo quality guarantee.

After passing:

```bash
python poc/source_fusion_v53.py \
  --triplet output/v5-new-photos-01/triplet_v5.json \
  --output-dir output/v53-final-trial-01 \
  --target-mid 0.47 \
  --window-target 0.56 \
  --base-contrast 1.08 \
  --detail-contrast 1.40 \
  --shadow-depth 0.30
```

The outputs are:

| File | Purpose |
|---|---|
| `final/photo_final.jpg` | Full original camera resolution, photographic JPEG |
| `final/photo_final_16bit.tiff` | 16-bit-per-channel photographic TIFF |
| `grade/exterior_mask_overlay.jpg` | **First file to inspect:** red overlay shows exactly where the exterior mask was applied |
| `grade/exterior_mask_preview.png` | Grayscale mask (white = full exterior adjustment) |
| `grade/tone_report.json` | Automated mask coverage, correction amount and parameters |
| `source_selection.npy` | Per-pixel source role: 0=under, 1=reference, 2=over |
| `fusion_report.json` | Alignment shifts and proportions of pixels selected from each RAW |
| `merged_hdr.npy` | Independently constructed HDR from all three original camera brackets |

The output may be several hundred megabytes. Check disk and RAM availability before trying 24 MP. The output folder must not already exist with files.

## Compare fairly against V5.2

```bash
python poc/benchmark_v5.py \
  --before output/v52-raw-guided-01/final/photo_final.jpg \
  --after output/v53-final-trial-01/final/photo_final.jpg \
  --output-dir output/v53-final-trial-01/review

explorer.exe "$(wslpath -w /opt/hdr-real-state/output/v53-final-trial-01)"
```

The previous professional kitchen photograph was provided visually in the conversation, **not verified as a file on the WSL filesystem**. Do not use the older bedroom `photos/DSC01858.jpg` as a target. If an actual professional retouch of the kitchen is saved locally, pass its verified file path to `--reference` in the benchmark.

## Non-negotiable acceptance check

Review three actual outputs: `grade/exterior_mask_overlay.jpg`, `review/comparison.jpg` and `final/photo_final_16bit.tiff`.

- The red overlay must cover only window/exterior or legitimately bright openings. It must not cover the white ceiling, the marble countertop or interior walls. If it does, **reject the automatic mask result**, even if the contact sheet looks bright.
- Outdoor skyline, vegetation and window edges should be as sharp and as faithful to the under-exposed original RAW as interior cabinets and furniture are to the middle RAW. Do not interpret invented detail or extra sharpening halos as recovery.
- No washed image, no glowing transition over furniture/curtains, no big artificial patches of sunlight.
- Shadows must retain depth without hiding real materials. Whites stay neutral and wood tones realistic.
- No doubled window frames or ghosted branches.
- The color-managed 16-bit TIFF must be checked at 100% zoom in Photoshop, not assessed only from the JPEG.

The algorithm may still misidentify bright interior walls as exterior, and the pixel-level source selection may still have seams in moving foliage. The current translation registration cannot correct moving subjects, parallax or large geometric shifts. The camera RAW development may have clipped some sRGB channels. **None of these can be honestly declared solved without real-image comparisons.**

**Stop condition:** If this independently changed, source-based pipeline does not yield a material visual improvement over V5.2 against the professional kitchen image, do not keep iterating arbitrary tuning or promising a product-ready neural solution. Record the failed quality gate and reassess the project scope/data/model strategy.

## Optional manually approved exterior mask

There is an optional `--window-mask` argument for a manually painted **8-bit grayscale mask** exactly matching the script's prepared reference preview dimensions (longest side 1024, maintained aspect ratio). White indicates outdoor/detail-preservation adjustment, black indicates interior. It is intended for diagnosing automatic segmentation, not as a substitute for repeatable automated processing. A nonempty new output directory is required for reruns.

The shape is validated; no unsafe resize/crop assumption is made. This option is not necessary for the initial test.
