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

## Next work: V5 real RAW validation on WSL

1. Run `git pull --ff-only`, verify `cv2` and run `tests/test_v5_pipeline.py` in the existing `.venv`.
2. Prepare `output/v5-01` with exactly three original RAWs (not previous HDR JPEGs).
3. Run the V5 translation merge, tone mapping and full-resolution final export in sequence; use a fresh output folder for retries.
4. Inspect full-size foliage, frame alignment, neutral walls, shadows, color and local-tone halos against the Photoshop edit; submit `comparison.jpg`, `report.json` and `merge_report.json`.
5. **Do not assume V5 equals AI fusion.** If a reliable baseline is established, research a true learned three-bracket fusion/retouch network that receives calibrated source frames, not pre-composited low-resolution HDR.

This file is the project's continuation context, not evidence that unexecuted work has passed tests.
