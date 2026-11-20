# V5.2 — opt-in RAW-sensor confidence HDR experiment

Status: **implemented on `main`, real camera run and photo-quality approval pending.** This path preserves V5/V5.1 outputs. It does **not** replace the user's existing HDR merge or count as learned AI.

## Motivation: the actual Sony RAW evidence

The user supplied a real audit of `DSC01132.ARW` (-1 EV), `DSC01134.ARW` (0 EV) and `DSC01133.ARW` (+1 EV), all ISO 160, with 6024 × 4024 visible Bayer samples.

The RAW metadata reports:
- Sensor black level: **512** for all four CFA indices
- Camera-declared channel white: **15360**
- LibRaw global white: **16380**
- Repeated upper code in every exposure/channel: **16372**
- Rare observed maximum: **16596**

The dominant upper value 16372 makes a camera-specific clipping plateau **plausible**. Since samples regularly exceed 15360, using 15360 as an unconditional hard clip would discard potentially useful sensor data. Conversely, an occasional maximum 16596 does **not** establish useful tonal information beyond the high-code plateau.

The camera clipping behavior is inferred from a **sampled distribution**, not experimentally calibrated with a step-wedge, independent camera response function or full scene comparison. Do not generalize this threshold to other camera models/ISOs.

## What V5.2 actually does

`poc/merge_hdr_raw_v52.py` consumes the existing original-RAW V5 triplet manifest and the three original Sony ARW files.

1. Read each RAW's untouched Bayer mosaic, CFA map, global/channel white levels and black levels.
2. For each Bayer color phase, sample the high values and require evidence of a repeatable near-global-white dominant code.
3. Generate a per-pixel confidence estimate from the sensor's black level and inferred high-code plateau.
   - Confidence is low very close to black where noise can dominate.
   - Confidence remains high at a declared white of 15360 if the plateau is demonstrably higher.
   - Confidence falls toward zero close to the observed high-code plateau (~16372 for these ARWs).
4. Aggregate worst-case confidence across the 2 × 2 Bayer block, then spatially smooth the confidence map.
5. Use the **same V5 translation alignment** for both demosaiced linear RGB and the native-sensor-derived confidence maps.
6. Multiply the existing exposure-reliability weights by the new native-sensor confidence weights. Keep the rest of V5's fusion, tone mapping and export unchanged.
7. Save a separate float32 HDR and full-resolution confidence maps for review.

**Important distinction:** the color pixels still come from the three original `rawpy`-developed *linear RGB* frames. We are using native RAW mosaics to determine how trustworthy each exposure is, **not merging 2 × 2 Bayer samples at subpixel translations**. This avoids mixing red/green/blue CFA phases improperly. The existing LibRaw development may have already clipped or altered some color information, so this experiment cannot promise restored sensor highlights.

The confidence formula and spatial smoothing are initial physical heuristics, **not a calibrated sensor noise model**. The same optical alignment limits (leaf motion and perspective) remain. If the high-code plateau is weak/ambiguous or the RAW orientation/geometry does not agree, the experiment aborts rather than silently applying arbitrary thresholds.

### Isolation

- Existing `poc/merge_hdr_v5.py` and V5 output folders are preserved.
- `v5_core.fusion_chunk` accepts an **optional** `sensor_confidence` parameter. When omitted, all existing default behavior is unchanged.
- The new merger refuses a nonempty output folder.
- All code goes directly to `main`. No branches, PRs or GitHub Actions.
- No generated RAWs, masks or model checkpoints are committed.

## Run on the user's existing working WSL environment

### 1. Tests

```bash
cd /opt/hdr-real-state
git pull --ff-only
source .venv/bin/activate

python -m unittest discover -s tests -p 'test_v5_pipeline.py' -v
python -m unittest discover -s tests -p 'test_raw_sensor_v5.py' -v
python -m unittest discover -s tests -p 'test_raw_guided_v52.py' -v
```

Expected: 13 existing HDR tests, seven sensor-audit tests and seven new guided-fusion tests. These new tests have been committed but **not yet executed on the user's WSL**. If a test fails, stop and share the traceback before running real RAW fusion.

### 2. New opt-in HDR fusion, same original camera RAWs

Use the **existing V5 camera-RAW manifest**, because it points to all three original camera ARWs and their individually decoded linear RGB frames. Do not regenerate it or reuse any JPEG HDR as an input.

```bash
python poc/merge_hdr_raw_v52.py \
  --triplet output/v5-new-photos-01/triplet_v5.json \
  --output-dir output/v52-raw-guided-01 \
  --alignment translation \
  --rows-per-chunk 256
```

Expected:
- `output/v52-raw-guided-01/merged_hdr.npy`
- `under_sensor_confidence.npy`, `reference_sensor_confidence.npy`, `over_sensor_confidence.npy`
- `merge_report.json`: candidate CFA clip codes, confidence statistics, registration and low-confidence fraction

These generated mask files are float32 at 6024 × 4024 and may occupy several hundred megabytes combined. Allow sufficient WSL disk space and system RAM. If RAW geometry or clipping plateau validation fails, **do not disable the checks** until the source is examined.

### 3. Render using the same tone-mapping settings as V5.1

```bash
python poc/tone_map_v5.py \
  --source output/v52-raw-guided-01/merged_hdr.npy \
  --output-dir output/v52-raw-guided-01/grade-balanced \
  --target-mid 0.37 \
  --local-strength 0.15 \
  --shadow-contrast 0.40

python poc/export_v5.py \
  --hdr output/v52-raw-guided-01/merged_hdr.npy \
  --tone-mapped output/v52-raw-guided-01/grade-balanced/tone_mapped_16bit.tiff \
  --output-dir output/v52-raw-guided-01/final
```

This produces original-resolution `photo_final.jpg`, `photo_final_16bit.tiff` and `export_report.json`. The tone map is deliberately identical to V5.1; that isolates the **sensor-confidence fusion** difference.

### 4. Compare exactly the same scene with V5.1

```bash
python poc/benchmark_v5.py \
  --before output/v5-new-photos-01/final-balanced/photo_final.jpg \
  --after output/v52-raw-guided-01/final/photo_final.jpg \
  --output-dir output/v52-raw-guided-01/review

explorer.exe "$(wslpath -w /opt/hdr-real-state/output/v52-raw-guided-01/review)"
```

No older `photos/DSC01858.jpg` Photoshop edit is used as a target for this new scene. If a matching professional edit is available, add it separately to the benchmark.

## Acceptance gate

Inspect the V5.1 and V5.2 files at **100% zoom** around:
- Window foliage and branches: same physical leaf edges, fewer fused bright/smeared spots
- Window frames and glass reflections: no halos or color shifts
- Dark interior: improved usable detail without stronger noise or patchy masks
- Whites/wood: reduced color cast without introducing new lighting
- Sensor confidence maps: expected selection between dark/middle/bright exposures, no large arbitrary holes

**Reject** this V5.2 opt-in path if it worsens details, colors, or noise. Its mere completion, code-level tests or reduced RGB error are not proof of photographic improvement.

## Next research

A commercially reliable sensor-aware HDR stage will eventually require raw camera color matrix/calibration, sensor noise models by ISO and channel, precision-preserving scene-linear demosaicing, more robust optical registration and deghosting. A separate specialized AI model should learn photographic retouching from licensed bracket/edit pairs instead of creating leaf/scenery geometry.
