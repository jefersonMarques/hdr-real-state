# V5 RAW sensor audit — before an exposure-aware V5.2 fusion change

## Why it matters

The current V5 begins with three real RAWs and decodes them through LibRaw at `output_bps=16`, `no_auto_bright=True`, and `gamma=(1,1)`. Those pixels are **demosaiced and camera-color-converted linear RGB** — not original Bayer samples or unbounded, calibrated scene radiance.

The fusion currently scores reliability using the developed RGB maximum near 0 and 1. That can misjudge exposure suitability because white balance, LibRaw scaling and camera color conversion may clip channels **after** the sensor capture. A true RAW-aware HDR algorithm needs camera-native black/white limits, CFA filter pattern, saturation masks, exposure/ISO calibration and noise confidence.

The source's 16-bit container does not prove a 16-bit camera ADC. The sensor's actual meaningful headroom is established from the native white and black levels, not JPEG/TIFF output bit depth.

## New read-only sensor inspection

The implementation `poc/inspect_raw_sensor_v5.py` examines each camera RAW's visible sensor mosaic **before** demosaicing or RGB conversion. It uses:

- `raw.raw_image_visible`: actual camera sample counts
- `raw.raw_colors_visible` and `raw.raw_pattern`: CFA color index and periodicity
- `raw.black_level_per_channel`: per-channel sensor black levels
- `raw.camera_white_level_per_channel` when present, else `raw.white_level`
- EXIF-derived bracket order and shutter/ISO/F-number ratios, with a common camera white balance

The program reports sample fractions close to the real black and white thresholds for each CFA index. It accounts for all phase positions in Bayer or periodic X-Trans CFAs rather than sampling only one color. The sampling is intentionally a statistical diagnostic: it does **not** yet generate spatial masks or modify V5's existing HDR result.

## Run without rebuilding the V5 output

```bash
cd /opt/hdr-real-state
git pull --ff-only
source .venv/bin/activate

python -m unittest discover -s tests -p 'test_v5_pipeline.py' -v
python -m unittest discover -s tests -p 'test_raw_sensor_v5.py' -v

python poc/inspect_raw_sensor_v5.py \
  --photos photos \
  --output-dir output/v5-raw-audit-01 \
  --sample-stride 8

cat output/v5-raw-audit-01/raw_sensor_report.json
```

Expected: 13 legacy V5 tests plus five new sensor-audit tests. They remain synthetic unit tests; do not assert real sensor behavior until the user runs the audit on their Sony RAWs.

`raw_sensor_report.json` includes each original filename, EXIF exposure ratio, color channel index, native black/white boundaries and normalized signal percentiles, near-black percentage and saturation percentage. The RAW files are neither copied nor modified. The report is local only.

For the current bracket:
- `DSC01132.ARW` = -1 EV = factor 0.5
- `DSC01134.ARW` = 0 EV = factor 1
- `DSC01133.ARW` = +1 EV = factor 2

Only two EV separate the darkest and brightest brackets, so strongly sunlit windows may still be difficult even with proper fusion. Check actual sensor clipping first rather than assume the source has recoverable detail.

## Real Sony ARW audit: 2026-11-20

The user ran all five original RAW inspection tests successfully and generated a report for:

| RAW | Exposure | Declared black | Declared white | Green at/above white | Red at/above white |
|---|---:|---:|---:|---:|---:|
| `DSC01132.ARW` | -1 EV | 512 | 15360 | 3.83% | 0.66% |
| `DSC01134.ARW` | 0 EV | 512 | 15360 | 6.20% | 2.69% |
| `DSC01133.ARW` | +1 EV | 512 | 15360 | 7.31% | 4.96% |

The 99th percentile of multiple Bayer channels was normalized to `1.068157315`, which corresponds to a raw code of approximately `16372` using the reported black/white levels. This is **above the camera-declared white of 15360**. The report's historical `sensor_saturated_fraction` thus measures samples *at/above the reported camera white threshold*, **not** proven physical photo-site clipping. Many high samples may be part of a plateau, but a percentile cannot by itself prove a plateau or usable remaining headroom.

The user also confirms `4024 x 6024` visible CFA geometry, 2x2 Bayer pattern, same base ISO 160 for the bracket, and common camera white balance `[3068, 1024, 1704, 1024]`.

### Supplemental non-destructive white-level/plateau check

The independent script now includes, per CFA color:

- `libraw_global_white_level` at the image level, **separate** from the four `white_levels` values.
- `sampled_raw_p99` and `sampled_raw_max` of actual mosaic counts.
- `sampled_max_code_fraction`.
- `dominant_high_code` and its fraction of all sampled pixels; a recurring high code may signal the sensor's output plateau.
- `above_declared_white_fraction` and `strictly_above_declared_white_fraction`, without claiming all such pixels are physically saturated.
- Historical `sensor_saturated_fraction` retained for report compatibility, but **explicitly reinterpreted** as a declared-white-threshold indicator.

Run in the same `.venv`, **without overwriting** the earlier audit:

```bash
cd /opt/hdr-real-state
git pull --ff-only
source .venv/bin/activate

python -m unittest discover -s tests -p 'test_raw_sensor_v5.py' -v

python poc/inspect_raw_sensor_v5.py \
  --photos photos \
  --output-dir output/v5-raw-audit-03 \
  --sample-stride 8

cat output/v5-raw-audit-03/raw_sensor_report.json
```

Seven tests are now defined (two additional plateau regression tests), but the new code still needs the user's WSL validation. The pre-existing V5 output and RAWs are unchanged. The actual result of this supplemental audit determines whether a 15360 threshold, a repeatable high code (potentially 16372), or another camera-calibrated criterion is appropriate for robust pixel confidence. This must be confirmed **before** enabling a sensor-clipping mask in the V5 fusion.

## Engineering plan for V5.2 after sensor results

The next stage **changes the source-of-truth for fusion reliability**, so it requires real camera validation. It should be implemented as a separate opt-in path, preserving `poc/v5_core.py` defaults:

1. Inspect actual Sony ARW CFA pattern and verify that sensor-visible dimensions, orientation and crop correspond to developed linear RGB geometry.
2. Produce sensor-native, per-channel saturation and noise-confidence maps using raw counts, channel black/white levels, shutter/ISO gain and read-noise estimates.
3. Use radiometrically consistent exposures and registration to blend demosaiced, **linear camera-space** data without clipping from display color transforms. Do not spatially merge Bayer mosaics with arbitrary fractional shifts; they have different CFA phase and require a CFA-aware or post-demosaic alignment strategy.
4. Transform into a documented working RGB space only after source fusion; apply white balance and camera profile thoughtfully, keeping out-of-gamut/highlight values as long as possible.
5. Run bounded tone mapping, export true full-resolution JPEG / 16-bit TIFF and compare against the prior V5 and a professional edit **from the same scene**.

Significant risks include exposure-normalization errors, demosaicing boundaries, Bayer CFA shifts, image rotation/mirroring, colored highlight casts, noise amplification and motion ghosts. No current V5.2 quality improvement is claimed. Do not silently replace the working V5 without source-level checks.

## The tone-map test correction

The previous `test_tone_map_is_finite_and_monotonic` asserted the output maximum must be strictly less than 1.0. In practice the working tone mapper can map some pixel channels exactly to the normalized maximum of 1.0. The test now accepts `<= 1.0`. This **corrects the test assertion**, not any underlying highlight clipping. Highlight-clipping evaluation remains a separate photographic quality issue.

All code modifications are directly on `main`, without branches, PRs or Actions.
