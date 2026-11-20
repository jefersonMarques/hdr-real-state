# HDR Real State v3 — AI photographic editor architecture

Status: **proposed architecture / evaluation gate, not a completed editor**.
Decision: replace "neural HDR output is the final photograph" with a **faithful RAW/HDR base followed by a controlled neural photographic editor**. The earlier neural HDR attempts remain available as experimental comparisons; do not delete existing scripts, outputs or checkpoints.

## Product contract

Input:
- A set of three photographs of the same immovable scene, ordinarily exposure-bracketed RAWs.
- Optional reference style or an approved edited JPEG for supervised learning.

Output:
- A naturally finished image suited to a real-estate listing, whose identity and geometry correspond to the photographs captured.
- High-resolution JPEG and high-bit-depth TIFF, offline; later batch processing and a Windows desktop interface.

Success is not demonstrated by neural GPU execution. A neural editor must exceed a strong non-neural photographic baseline on held-out real-estate scenes while preserving authentic details.

## Components and processing sequence

1. **Bracket discovery and raw decoding** — read EXIF shutter/aperture/ISO, verify capture grouping, orientation, camera profile and working color space; decode with controlled black/white levels and stable linear encoding. Do not treat an sRGB JPEG or a camera tone curve as scene-linear radiance.
2. **Registration and motion handling** — align camera displacement and manage foliage/people movement; reject badly misregistered sequences instead of masking the issue.
3. **Reliable photographic base** — exposure-aware linear merge with clipping confidence, noise-aware shadow selection and full traceability back to the source brackets; explicit ICC/sRGB conversion for previews.
4. **Neural editing model** — given a low-/medium-resolution *rendered base*, predict bounded tone/color/local filter adjustments. This stage is **not** a free-form generative image model and must not change geometry or scene content. First evaluation uses an external pretrained model; later use supervised retoucher-pair training.
5. **Controlled high-resolution renderer** — apply validated photo adjustments while protecting original structures and source details, with an editable record of settings. Prefer parametric transforms, adaptive color LUTs and spatially smooth maps over unconstrained pixel hallucination.
6. **Validation and export** — compare against the baseline and professional references, assess bright windows, gray/white neutrality, color fidelity, edge stability and 100% crops. Export only approved variants.

Data flow:

```text
3 RAWs + EXIF
   |
   v
RAW decoding + color management + alignment
   |
   v
Source-traceable HDR radiance + non-neural base render
   |                                         
   +----> 16-bit/TIFF archival output
   |
   v
Neural photographic editor (DeepLPF / HDRNet / adaptive LUT candidate)
   |
   v
Constrained adjustments (tone curves, spatial masks, white balance, color)
   |
   v
High-resolution preview and final export + audit record
```

## v3 experiment — minimal first proof

Start from the **already-created conventional** preview. Apply a *pretrained neural photo editor*, with no masks copied from the previous neural HDR v2 pipeline. Compare its unaltered output to the conventional input and to the user's Photoshop reference.

Recommended first candidate:
- [DeepLPF](https://github.com/sjmoran/deeplpf-image-enhancement): lightweight model that predicts cubic, graduated and elliptical parametric filters, MIT-licensed **repository code** with pretrained photo retouch weights available in the checkout. Python >= 3.11, PyTorch >= 2.6; the upstream README documents a modern CLI. The pretrained default model **does not have the optional color-mixing head**; do not promise it will neutralize white balance or match a real-estate style without training.
- [AdaInt](https://github.com/ImCharlesY/AdaInt): Apache-2.0 code, adaptive learned 3D LUT, useful for efficient **global** color enhancement. The official reference version depends on old PyTorch/CUDA/MMCV tooling; porting, inference licensing and compatibility need review.
- [Google HDRNet](https://github.com/google/hdrnet): Apache-2.0 code, local bilateral-grid transforms suited to smooth photographic retouching. The original TensorFlow-based release is historical and requires a modern port for the WSL product stack.

Licenses for third-party **weights, datasets, images, dependencies and model export** must be verified separately from source-code licenses before including anything in a commercial installer.

## Component boundaries

```text
src/
  hdr/
    raw_decoder.py
    bracket_grouping.py
    registration.py
    radiance_merge.py
  editor/
    image_editor.py
    deeplpf_adapter.py
    adaptive_lut_adapter.py
    render_adjustments.py
  evaluation/
    reference_comparison.py
    image_metrics.py
    regions.py
  app/
    cli.py
tests/
docs/
  adr/
```

These are **target** paths, not yet-created files. The initial v3 evaluation stays in `poc/` until photographic quality passes. Implement components behind clear input/output interfaces; keep neural model source/checkpoints external at first.

Key interfaces (future): `RawBracket`, `LinearHdrImage`, `DisplayImage`, `EditParameters`, `EvaluationReport`. Do not silently reinterpret one image type as another. Metadata includes camera, lens, orientation, EXIF, color-space and exposure ratios.

## Quality gates

- **Q0 — baseline is valid:** same architecture and camera geometry across brackets, window view from real sensor data and reproducible color profile.
- **Q1 — AI adds measurable value:** against the same held-out properties, reviewers prefer the editor's output to the baseline for usable real-estate photography, without new structural mistakes; sample size > one scene.
- **Q2 — content authenticity:** geometry, windows, objects, fixtures and defects remain recognizable and spatially stable; no unrequested object removal, furniture generation or misleading external view.
- **Q3 — color:** neutral ceilings/walls under mixed light, plausible wood and vegetation colors, skin/brand colors when present; assess with region crops and consistent display color space.
- **Q4 — robustness:** >3 source exposures are grouped safely or rejected, missing/bad EXIF explicit, RAW decode failures diagnosed, CPU fallback works, no OOM or partial corrupted output.
- **Q5 — offline product:** final runtime bundles only licensed artifacts, no hidden network access, portable acceleration strategy and predictable exports.

A single RGB mean absolute error (MAE) score is **not** a trustworthy automatic ranking: the Photoshop reference may have another crop, perspective correction, exposure intent or local edits. Evaluate PSNR/SSIM/Delta E only after registration and consistent color conversion, and also use blind paired human comparisons. Split datasets **by property or photo session**, never random bracket files, to prevent leakage.

## Data and training plan

Record each authorized scene as a **scene group**, not isolated RGB pairs:

```text
dataset/
  scenes/
    scene_000001/
      raw/
        under.ARW
        reference.ARW
        over.ARW
      base/
        base_linear.tiff
        base_display.png
      target/
        retouched.png
      metadata.json
```

The metadata records property/session IDs (not addresses), source consent/license, camera, EXIF, editor, rendering profile and processing version. Keep original RAWs immutable and never publish third-party imagery without permission.

**Pilot:** 10–20 legitimately licensed scenes to verify alignment, training input equivalence and UI workflow. **Initial evaluation:** 30–50 scenes across distinct properties, lighting and exposure spacings. **Fine-tuning:** acquire several hundred consistently edited pairs if the pre-trained weights fail; quality depends more on consistency and diversity than an arbitrary minimum count. One Photoshop image is only a visual test, not a training dataset.

Train low-/mid-resolution conditional edits, use controlled losses:
- RGB reconstruction plus perceptual/structure similarity.
- Neutral wall/ceiling and controlled highlight losses where reliable masks or labels exist.
- Edge/identity consistency to penalize spatial structural changes.
- Mild regularization/identity behavior to avoid exaggerated color saturation and over-editing.

Monitor performance and memory on the 8 GB RTX 4060; use modest training crops, mixed precision if validated, gradient accumulation when useful. Full-resolution processing is a separate performance/quality validation step.

## Architecture risks and decisions

1. **Model domain mismatch:** pretrained FiveK image retouch models learn photographers' stylistic corrections, not necessarily real-estate whites/windows. First run is an *unmodified benchmark*, not a claimed solution.
2. **Color-space mismatch:** current RAW pipeline uses several decoders and tone curves; separate radiance and display RGB and document transforms.
3. **Incorrect exterior masking:** the previous v2 masks selected about 11% of a living room including some non-window areas. Avoid reusing this mask as truth.
4. **Performance:** a 1.7M-parameter model is not inherently low-memory at original RAW resolution. Test VRAM, time and CPU fallback, not only parameter count.
5. **Safety/trust:** do not let unconstrained generative models modify furniture, structure, damage, room proportions or outdoor views in commercial listings.
6. **Commercialization:** user-supplied photographs and pretrained weights may have separate redistribution/use restrictions.
7. **Model orchestration:** use one photo editor candidate at a time; do not invent a multi-model pipeline before establishing baseline quality.

## Near-term milestones

- M1: deep-learning photo editor benchmark over the *same* conventional render and Photoshop reference. **No changes to v2**.
- M2: measured sample of different real-estate scenes; select viable architecture or reject pretrained models.
- M3: common color-managed RAW base and supervised fine-tuning data schema.
- M4: controlled local editing and quality review.
- M5: offline Windows interface, batch workflow and cross-device inference.

**Current state:** M1 planned/in progress. A professional-looking final result is **not yet achieved**.
