# Repository working agreement — HDR Real State

## Mission

Build an offline-first, commercial real-estate **AI photo editor** for three exposure-bracketed camera RAWs. The product needs true photographic enhancements learned by neural models; a classical HDR merger with a manually written preset is **not** an acceptable AI product.

## Current approved architecture

**v3 architectural direction approved on 2026-11-20:**
1. Create a faithful RAW/linear HDR base and preserve its geometry and external window information.
2. Feed an appropriately rendered base into a trained **neural photographic editing** model.
3. Preserve original furniture, walls, floors, doors, windows, exterior views and property defects; no invented or removed objects.
4. Compare AI output with an unmodified non-neural baseline and professionally edited examples.
5. Continue development only after visual-quality evidence supports the neural contribution.

Source of truth: [docs/v3-architecture.md](docs/v3-architecture.md).
First model evaluation: [docs/v3-first-model-test.md](docs/v3-first-model-test.md).

## Current actual implementation

- `poc/hdr_afunet.py`: historical AFUNet HDR model experiment.
- `poc/hdr_diagnostics.py`: linear RAW baseline and comparison.
- `poc/hdr_window_recovery.py`: historical JPEG-level source highlight experiment.
- `poc/hdr_pipeline_v2.py`: historical experimental neural HDR + RAW highlight blending.
- `poc/benchmark_afunet.py`: three-checkpoint AFUNet comparison.
- `poc/benchmark_editor_v3.py`: comparison harness for neural **photo-editing** against conventional baseline and an optional professional reference.
- `docs/v3-first-model-test.md`: executable first trial with **external** DeepLPF pretrained model, isolated editor environment and RTX 4060.
- `poc/prepare_editor_input.py`: prepare a bounded RGB image for memory-limited GPU inference, keeping existing source images intact.

**DeepLPF's pretrained checkpoint loaded on CUDA, but the first native-resolution attempt hit a GPU out-of-memory exception on an 8 GB RTX 4060. No complete v3 inference output or photographic quality improvement has been verified yet.** The next test uses a 1024-pixel input, without overwriting previous images. External models, weights and user photos are intentionally not committed.

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

## Next work after the current user runs the v3 test

1. Review `output/v3-editor-review-01/comparison.jpg`, `report.json` and the resulting DeepLPF edited PNG.
2. If baseline beats the neural editor, classify it as a negative result; do not cover it with new code-based presets.
3. Evaluate a second candidate (adaptive 3D LUT / HDRNet) using the same image and professional reference.
4. When one architecture demonstrates benefit, prepare a licensed multi-property paired dataset, with train/validation/test split by property.
5. Only then implement a stable photo-editor service and future desktop GUI.

This file is the project's continuation context, not evidence that unexecuted work has passed tests.
