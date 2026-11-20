# First v3 editor evaluation — DeepLPF default checkpoint

Date: 2026-11-20

## Inputs and reproducibility

User ran the pretrained DeepLPF editor on a conventional HDR preview prepared at 1024 px long edge, after a full-resolution attempt on an 8 GB RTX 4060 exhausted VRAM. The model loaded and inference produced a PNG at 1024 × 684. The baseline and editor outputs were compared against the same property's Photoshop retouch.

Input metadata and SHA-256 values from the user-provided evaluation report:

- Baseline: `output/v3-input-1024/conventional_local.png`, sha256 `03a7f24306d6b147232a0a1f7c7c1856fa6d33913c1abdca9d8df16228fdc46f`
- DeepLPF: `output/v3-deeplpf-1024/conventional_local_enhanced.png`, sha256 `4358d952214fc39f76044efd790cce11ffad3417620a1b3d66489f00f4810870`
- Professional: `photos/DSC01858.jpg`, sha256 `904ab1f13a7e6c45cb9b1723f78ea8cee142f19593c7361c5062f1d7f401a10a`

The reference was resized; **no geometric registration or color-managed comparison** was applied.

## Quantitative observations

| Measure | Conventional | DeepLPF | Photoshop |
|---|---:|---:|---:|
| Median normalized brightness | 0.6777 | 0.6199 | 0.6401 |
| Dark pixels below 0.25 | 0.67% | 3.56% | 12.52% |
| Bright pixels above 0.95 | 0.24% | 0% | 2.28% |
| Approximate RGB MAE vs Photoshop (0–255) | 42.3678 | 43.0569 | — |

DeepLPF changed the baseline by **18.9614** average RGB levels per pixel. This is evidence that the learned photo-editor performed a nontrivial transformation, but it did **not** improve approximate pixel similarity to the reference. Relative MAE difference: DeepLPF was 0.6891 RGB levels worse, a small difference that should **not** be interpreted as statistically significant or as an objective perceptual-quality grade.

## Visual assessment

The uploaded three-panel comparison shows stronger darkness/contrast in the DeepLPF image, while ceiling neutralization, window foliage and property-photo finishing remain inferior to the Photoshop retouch. The editor did not deliver a distinct professional-quality improvement on this scene.

**Result:** reject this unmodified DeepLPF **checkpoint on this one real-estate scene** at the current quality gate. Do not conclude DeepLPF as an architecture is intrinsically unusable or that a professionally fine-tuned variant would necessarily fail.

## Technical explanation and next step

The bundled DeepLPF model is trained on MIT-Adobe FiveK expert edits, not the user's real-estate editing workflow. The default pretrained checkpoint does **not** include the optional cross-channel color mixer in the newer repository; this limits its ability to remove problematic white-balance/color casts.

No further hand-tuned brightness/mask filters should be added to make the neural score appear better.

Next candidate requirements:
- A **learned photo color/tone editor** (adaptive 3D LUT or bilaterally guided model) with usable pretrained weights and a stable inference path on PyTorch 2.11/CUDA 12.8, preferably predicting adjustments on a low-resolution preview and applying them at higher resolution.
- Evaluate the unchanged neural output against the same baseline and reference, then multiple independent properties.
- Verify code and **checkpoint** licensing. For example, VeraRetouch publishes sophisticated reference-based retouch functionality but its released pretrained weights explicitly prohibit commercial use and derived-model commercial products; therefore it is **not selected as a commercially distributable pretrained foundation** without separate permission.
- The older AdaInt/SepLUT implementations require old CUDA toolchains; avoid mutating the working WSL environment to build them. Research a modern pure-PyTorch LUT implementation separately and confirm weight rights.
- Collect licensed bracket/source and expert-edited pairs for specialized fine-tuning once a viable editor architecture is found.

**Project status:** neural editing model executed successfully in low resolution; image quality gate **failed**. Next development priority is model selection / training data, not more AFUNet HDR or arbitrary JPEG filters.
