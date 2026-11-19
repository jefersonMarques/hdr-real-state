# HDR Real Estate

Local-first HDR image processing for professional real-estate photography.

## Goal

Combine **exactly three bracketed camera RAW files** (underexposed, reference, overexposed) with a **real neural HDR reconstruction model**, then develop the result with consistent white balance and natural interior/window exposure.

The intended desktop product must work offline on Windows and offer CPU fallback when an AMD, NVIDIA, or Intel GPU is incompatible.

## Current status

**Research and model-validation stage. This repository does not yet contain an approved production HDR engine or a distributable installer.**

The project will not misrepresent a conventional exposure-fusion implementation as an AI-based HDR engine. All neural model weights, dependencies, and licenses must be verified before bundling.

- [Open-source evaluation](docs/open-source-evaluation.md)
- [Implementation plan](docs/mvp-plan.md)

## Proposed stack

- Python + PyTorch for neural-model evaluation
- LibRaw / rawpy for camera RAW decoding
- Qt / PySide6 for the first desktop interface
- C++ and ONNX Runtime for optional production optimization
- Local model inference without uploads, cloud APIs, or per-image fees

## Model validation gates

1. Successfully import 3 camera RAW images, preserve relative exposures, camera color metadata, and alignment.
2. Run a real pretrained neural fusion model on CPU.
3. Compare output against professional real-estate edits with windows, mixed lights, white walls, and moving objects.
4. Validate GPU providers and memory-constrained inference.
5. Confirm licenses for **both** code and pretrained weights before bundling.
6. Build and verify an offline Windows installer.

## Notes

Commercial distribution of derivative works of GPL-3.0 / AGPL-3.0 projects has obligations, so CalibRaw and RapidRAW are only research references unless the licensing strategy changes.

No photos should be transmitted outside the user's computer.
