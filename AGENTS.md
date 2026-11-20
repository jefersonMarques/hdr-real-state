# Repository working agreement — HDR Real State

## Mission

Build an offline-first, commercial real-estate **AI photo editor** for three exposure-bracketed camera RAWs. The product needs true photographic enhancements learned by neural models; a classical HDR merger with a manually written preset is **not** an acceptable AI product.

## Current experimental milestone — V5 (2026-11-20)

User approved an independent, more reliable **three-camera-RAW photographic baseline**, to be evaluated before adding neural retouching. New code is in `poc/v5_core.py` with the five standalone V5 commands and `tests/test_v5_pipeline.py`. The runnable guide is [docs/v5-real-hdr-pipeline.md](docs/v5-real-hdr-pipeline.md).

V5:
- Decodes original RAWs using shared WB and linear gamma, infers exposure ratios from EXIF and saves float32 source frames.
- Aligns **translations only** using phase correlation; does not claim full optical registration or foliage deghosting.
- Merges original linear RGB with exposure-aware confidence weights, builds `merged_hdr.npy`, tone maps and exports real 16-bit sRGB TIFF plus JPEG.
- Is a **non-neural** baseline. It neither uses the old conventional HDR JPEG nor does trained AI fusion. Specialized triplet-aware AI remains the long-term goal.
- **Has not been run or visually approved on the user's WSL photos yet.** Synthetic tests are committed and need user-side validation.

Continue exclusively on `main`, preserving V2–V4 experiments. Do not create branches, PRs or GitHub Actions.

## Final photographic quality gate — V5.3 (2026-11-20)

User's comparison of the kitchen professional photograph and V5.2 was still unacceptable: the pipeline produced washed highlights and interior, with low contrast and less convincing window/exterior details. User requested **one more direct implementation**, prioritizing correct joining of actual exposure information, local contrast, preserving sharp external scenery and preventing false window light bleed; if unsuccessful, user may discontinue project.

- New `poc/source_fusion_v53.py` is a separate **opt-in** standalone end-to-end source composite and local photographic renderer. It reads the original V5 `triplet_v5.json` (under/reference/over original-camera RAW-developed linear frames) and constructs a new float32 HDR, not the previous V5.2 HDR. Highlights preferentially use valid underexposed detail; normal architecture/reference; true shadow signal from overexposed bracket. A reference-frame disagreement gate limits some ghosts.
- At preview scale, an automatic large-bright-area exterior mask is selected and guided-filter smoothed to control window tone without diffuse spill across structural edges. A **red-on-source mask overlay** is saved and **must be visually checked**. Outdoor mask mistakes are a rejection criterion, not a reason to claim better HDR.
- New tone mapper separates edge-aware low-frequency log luminance and high-frequency detail and controls shadow toe, with no generative hallucination. Outputs full camera-resolution JPEG quality 98 and source-derived 16-bit TIFF.
- `tests/test_source_fusion_v53.py`: nine synthetic coverage tests. They and the actual WSL Sony kitchen run remain **pending**. Documentation [docs/v53-last-quality-gate.md](docs/v53-last-quality-gate.md) has exact single-run commands and stop conditions.
- This remains conventional source-preserving photography, **not neural AI**. Professional quality and native RAW precision beyond existing LibRaw conversion are unproven. It must not be claimed as production-ready.
- User requested working exclusively on `main`, no branches, PRs or GitHub Actions. Preserve all earlier experiments.
- **Stop gate:** if the output fails to improve the exterior/window continuity, textures and natural contrast **materially** compared with same-scene V5.2 and the professional kitchen photograph, stop the current heuristic iteration and reassess viability/data/model strategy rather than promising random improvements.

## V5.2 opt-in native sensor-guided merge (2026-11-20)

The user's third real Sony RAW audit confirmed that all four CFA indices and all three brackets share a repeated upper code **16372**, which is above camera-declared white `15360` and near LibRaw global white `16380`; rare maximum `16596`. It is a **possible sensor clipping plateau**, not a validated physical saturation limit. The seven RAW audit tests **passed on the user's WSL**.

A separate opt-in V5.2 implementation was published **directly on `main`**:
- `poc/merge_hdr_raw_v52.py`: reads original ARW Bayer mosaics, detects a supported high-code plateau, creates 2×2 CFA-block confidence masks and aligns them to existing original RAW-developed linear RGB frames before fusion. Aborts on unsupported orientation, geometry or ambiguous clipping plateaus.
- `poc/v5_core.py`: `fusion_chunk(..., sensor_confidence=None)` optionally multiplies the weights; the existing V5 call and default behavior are unchanged.
- `tests/test_raw_guided_v52.py`: seven new synthetic regression tests for plateau selection, color-phase masks, zero/one confidence, and independent output. **New tests pending execution in WSL.**
- `docs/v52-raw-guided-hdr.md`: run the new fusion from existing `output/v5-new-photos-01/triplet_v5.json`, tone map using the same V5.1 settings, export full resolution, and compare with original V5.1 output of the **same scene**.
- **Limitations:** confidence is a heuristic, not a calibrated sensor noise model; fused pixels are still LibRaw-developed linear RGB, not native mosaic fusion; colored highlight information may be lost during development. No trained AI is applied in V5.2 and quality improvements are not yet verified.

Keep previous V5 and V5.1 outputs unchanged; no branches, PRs, Actions, or raw photo commits.

## Sony RAW plateau finding — fidelity blocker (2026-11-20)

The user's three real Sony `DSC01132.ARW`, `DSC01134.ARW`, `DSC01133.ARW` audits succeeded; full native mosaic dimensions `4024 x 6024`, 2x2 Bayer, sensor black `512` and **camera-declared white `15360`**. Several CFA color-channel p99 values normalize to `1.068157315`, consistent with raw code ~`16372`; this exceeds the declared threshold. The old `sensor_saturated_fraction` is **not proven physical clipping**, only the fraction reaching/exceeding camera-declared white. Do not implement binary clipping confidence from this number without checking the upper-code histogram.

- `poc/inspect_raw_sensor_v5.py` now records `libraw_global_white_level`, observed raw maximum, p99 raw code, high-code mode and fractions against camera white to identify possible native clipping plateaus, without changing the actual HDR fusion.
- `tests/test_raw_sensor_v5.py` now contains seven tests, including an artificial high-code plateau case. These new tests and RAW runs are **pending WSL validation**.
- `docs/v5-raw-sensor-audit.md` contains the real audit and a new output folder `output/v5-raw-audit-03`. Never overwrite previous outputs.
- Decide on physically justified per-channel clip thresholds and source confidence only after reviewing that new diagnostic. Do not confuse histogram plateau, ADC maximum, or camera-declared white with proof of retained highlight detail. Preserve production-facing V5 results.

## V5 RAW sensor audit and next V5.2 quality gate (2026-11-20)

The user's new Sony RAWs were successfully processed by V5, but user correctly requested exploiting **RAW sensor data**, not merely treated-as-image RGB. The existing `load_linear_raw` is 16-bit `rawpy.postprocess(no_auto_bright=True, gamma=(1,1))` then float32 linear developed sRGB — not native sensor mosaic fusion.

- `poc/inspect_raw_sensor_v5.py`: read-only audit of RAW CFA native values, black levels and white/saturation thresholds for all three original camera RAWs, without demosaicing, resampling or changing the existing V5 output.
- `tests/test_raw_sensor_v5.py`: five synthetic regression tests for CFA sampling, per-channel thresholds and report creation; WSL actual Sony camera test pending.
- `docs/v5-raw-sensor-audit.md`: user-run procedure and V5.2 sensor-source fusion engineering risks/gates.
- `tests/test_v5_pipeline.py`: corrected an overly strict assertion requiring maximum pixel value `<1`; exactly `1` is allowed by the existing tone mapper. This fixes **a test**, not clipping.
- User's current bracket: `DSC01132.ARW`(-1 EV), `DSC01134.ARW`(0 EV), `DSC01133.ARW`(+1 EV). The successful V5 photo output is preserved.

**Critical risk:** CFA/Bayer mosaics cannot be arbitrarily fractionally translated and merged as if they were color RGB; channel phase, white/black levels, orientation, camera color matrices and scene-linear sensor gain all matter. First inspect actual sensor audit and compare against existing photo; only then enable separate opt-in RAW-aware reliability masks and a camera-aware fusion pipeline. Do not silently change the working V5. No claim of a functioning V5.2 native sensor fusion yet.

## V5.1 shadow-contrast trial and new photo bracket (2026-11-20)

User verified the first V5 HDR output on the original room: it was closer to the Photoshop reference but washed out; median brightness 0.6766 vs reference 0.6402, with far fewer dark regions (0.0042 vs 0.1266). Preserve the reliable V5 source merge and test conservative tone-map changes rather than rebuilding the HDR network.

- `poc/v5_core.py`: optional `target_mid`, `local_strength`, `shadow_contrast` exposed through `poc/tone_map_v5.py`. Original defaults are unchanged.
- `tests/test_v5_pipeline.py`: four extra parameter and shadow-depth regression tests. **Their WSL execution is pending.**
- New photographs were copied by user into `/opt/hdr-real-state/photos`; filenames, exposure metadata and count **have not been inspected** by this assistant. Verify three matching RAWs, then run V5 into a new named output folder; never overwrite previous scenes.
- New full run and no-Photoshop-reference comparison: [docs/v5-real-hdr-pipeline.md](docs/v5-real-hdr-pipeline.md), V5.1 section.
- The prior expert `photos/DSC01858.jpg` applies only to the previous room unless explicitly proven to be the new scene; do not compare different scenes as ground truth.
- V5.1 is a **classical HDR photograph**, not true learned neural multi-exposure fusion. The future commercial product still needs authentic neural editing validated on paired RAW/expert data.

## Current approved architecture — V4 (2026-11-20)

**Single-branch rule: work directly on `main` only. Do not create branches, pull requests, or GitHub Actions unless explicitly authorized.**

User decision: **do not use a flawed conventional HDR output as the input to the neural image editor**. Instead, decode the original three camera RAWs separately and let an AI model see all three exposures, including real exterior detail from the darkest bracket and interior shadow information from the brightest.

- Short-term experiment: [docs/v4-triplet-neural-hdr.md](docs/v4-triplet-neural-hdr.md) — FLUX.2 Klein multi-reference conditioning with three independent camera RAW-derived display images.
- Long-term target: camera-calibrated and geometrically aligned, exposure-aware neural fusion + photographic editing, with faithful source-derived details, controlled color/tone, and genuine high-resolution export.
- Critical distinction: **FLUX multi-reference image generation is not a HDR-trained fusion model.** A good-looking 768px FLUX result cannot prove detail authenticity or native 24 MP quality.
- Do not copy Photoshop reference content into the camera geometry or invent leaves, window views, furniture, lights, shadows, or material colors.
- Retain all v2/v3 experiments as comparisons; the new V4 model must **not** load an existing conventional HDR as its neural input.

The older [v3 architecture](docs/v3-architecture.md) remains historical documentation and is no longer the current processing route.

## Current actual implementation

- `poc/prepare_triplet_v4.py`: V4 EXIF-sorted, independently developed three-RAW camera references; same WB, no HDR merge.
- `poc/flux_triplet_v4.py`: V4 actual FLUX.2 Klein call with a three-image list for reference conditioning, experimental.
- `tests/test_triplet_v4.py`: synthetic triplet ordering, geometry, manifest and provenance tests; user WSL run pending.
- `docs/v4-triplet-neural-hdr.md`: current architecture, run commands, quality gates and implementation sequence.
- `poc/hdr_afunet.py`: historical AFUNet HDR model experiment.
- `poc/hdr_diagnostics.py`: linear RAW baseline and comparison.
- `poc/hdr_window_recovery.py`: historical JPEG-level source highlight experiment.
- `poc/hdr_pipeline_v2.py`: historical experimental neural HDR + RAW highlight blending.
- `poc/benchmark_afunet.py`: three-checkpoint AFUNet comparison.
- `poc/benchmark_editor_v3.py`: comparison harness for neural **photo-editing** against conventional baseline and an optional professional reference.
- `poc/benchmark_3dlut_v3.py`: unvalidated-on-real-checkpoints neural adaptive 3D LUT inference with native PyTorch trilinear sampling and row-based application.
- `docs/v3-adaptive-lut-test.md`: next execution and Photoshop comparison instructions for the RTX 4060.
- `docs/v3-first-model-test.md`: executable first trial with **external** DeepLPF pretrained model, isolated editor environment and RTX 4060.
- `poc/prepare_editor_input.py`: prepare a bounded RGB image for memory-limited GPU inference, keeping existing source images intact.
- `poc/test_flux_klein_edit.py`: experimental FLUX.2 Klein image-to-image generative editing using a separate `.venv-flux`, 768px bounded input and sequential CPU offload; compare `flux_input.png` with `flux_edit.png`, do not assume output is faithful to a property.
- `poc/flux_klein_fidelity.py`: separate experimental Flux2KleinInpaintPipeline source-initialized run with two explicit denoising strengths, all-white edit mask, optional Photoshop appearance reference, and output difference maps.
- `docs/v3-flux-fidelity.md`: commands, quality gate and secondary reference risks. Reuse the same prepared 1008×672 input. Confirm current Diffusers supports this inpaint pipeline; actual 4B WSL run is pending.
- `poc/export_flux_fullres.py`: optional, source-resolution photograph export guided by smooth constrained changes inferred from the low-resolution FLUX edit; takes original full-resolution float32 conventional HDR and highlight mask to preserve 16-bit tonal precision. This does not run native-resolution FLUX or generate missing fine detail.
- `docs/v3-flux-fullres-export.md`: user-run export, side-by-side quality check and limitations.
- `docs/v3-flux-klein-test.md`: executable test and safety/memory limitations.

**DeepLPF was successfully executed at 1024 × 684 on the RTX 4060. The first real-estate quality gate FAILED:** approximate MAE vs Photoshop was 42.3678 for conventional vs 43.0569 for DeepLPF (reference resized, unregistered), and visual ceiling/window/color quality remained deficient. The original full-resolution run ran out of VRAM. See [docs/v3-deeplpf-evaluation.md](docs/v3-deeplpf-evaluation.md). The next step is to validate a different learned editor, not hand-tune DeepLPF with custom masks. External models, weights and user photos are intentionally not committed.

## Engineering constraints

- English-only identifiers, filenames, functions, arguments, classes and constants.
- Clean, readable, individually testable functions. Keep interfaces small and use SOLID/DRY/KISS/YAGNI pragmatically.
- Comments only where necessary, in impersonal technical Brazilian Portuguese.
- Do not silently change algorithm, color profile, training data, source model, output semantics or existing user photography.
- Never blend gamma-encoded display RGB into scene-linear HDR data without explicit conversion.
- Preserve old pipeline versions as reproducible baselines. Use new output directories to avoid losing historical results.
- Do not overwrite original RAW files, retouched references, checkpoints, or prior results.
- Audit model/code licenses and pretrained weight redistribution separately before packaging commercial products.
- Preserve GPU compatibility with CUDA 12.8/PyTorch 2.11 on the existing RTX 4060 WSL setup; use a separate venv for incompatible research tooling.
- Never claim a GPU running implies useful neural photographic editing.
- Reject fabricated scenery/objects, unlabeled hallucination and structural geometric modifications.

## Next work: validate RAW-guided V5.2 opt-in against V5.1

1. User runs existing V5 tests, all seven sensor-audit tests, and seven new `tests/test_raw_guided_v52.py` tests on the WSL Python `.venv`.
2. User runs `poc/merge_hdr_raw_v52.py` on the existing V5 manifest; this reads **three original ARWs** and their existing original RAW linear RGB arrays, never a pre-composited HDR image.
3. Check `output/v52-raw-guided-01/merge_report.json` for detected per-channel clip codes, actual camera orientation, registration confidence and fraction of pixels where all exposures are unreliable.
4. Tone map/export using **the same settings** as the V5.1 baseline and benchmark identical camera scene V5.1 vs V5.2. Inspect foliage, windows and dark noise at 100%.
5. Reject V5.2 if it adds halos, blocks, noise, cast or detail loss; only after photographic evidence should the default merge be reconsidered.
6. True high-fidelity RAW HDR beyond this PoC needs validated camera-space color pipeline, noise model, geometric/optical deghosting and eventually a separately trained three-exposure neural editor.

This file is the project's continuation context, not evidence that unexecuted work has passed tests.
