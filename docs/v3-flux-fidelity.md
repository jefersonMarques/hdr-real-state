# FLUX.2 Klein v3.1 — conservative editing and fidelity comparison

## Goal

The first 1024-side FLUX.2 Klein test improved apparent room contrast and whitened the ceiling, but also introduced unwanted light/shadow changes and overbright highlights. The user's Photoshop edit is the visual target, not a source of permission to synthesize objects or scenery.

**This experiment is separate from the existing working `poc/test_flux_klein_edit.py`.** Do not change the old experiment, three camera RAWs, or previous outputs.

## What is actually different

The earlier `Flux2KleinPipeline` accepts a source image as reference conditioning but has no `strength` parameter. Altering the number of sampling steps is **not** the same as controlling the amount of change.

The new `poc/flux_klein_fidelity.py` uses Hugging Face Diffusers' `Flux2KleinInpaintPipeline`. It uses:
- The actual source photograph as the initialized image.
- A full white mask so the entire photograph may undergo photographic editing.
- Real `strength` conditioning, which controls the noise applied to the initialized image and the portion of denoising steps used.
- The same prompt, seed, resolution and model for comparison between strengths.
- A conservative instruction to avoid artificial shadows, sunny hotspots, orange wood and fake tree leaves.

These are **experimental image-to-image semantics**, not a guarantee of faithful geometry. The source photo can still be reinterpreted. The official pipeline supports `strength` and `image_reference` but the installed version in the user's isolated `.venv-flux` has **not yet been verified**. The official inpainting examples commonly use a base checkpoint, so the particular combination with the step-distilled Klein 4B remains to be checked on WSL.

With four total denoising steps, strengths of `0.50` and `0.75` give two and three effective steps respectively in the current Diffusers scheduler. Four-step Klein sampling is the experimental constraint; more steps do not automatically guarantee higher fidelity.

## Run on the existing WSL CUDA environment

~~~bash
cd /opt/hdr-real-state
git pull --ff-only
source .venv-flux/bin/activate

python -m unittest discover -s tests -p 'test_flux_klein_fidelity.py' -v

python -c "from diffusers import Flux2KleinInpaintPipeline; print('Inpaint pipeline OK')"
~~~

If the import fails because your older Diffusers checkout does not yet contain `Flux2KleinInpaintPipeline`, update **only** the existing isolated FLUX environment:

~~~bash
python -m pip install --upgrade git+https://github.com/huggingface/diffusers.git
~~~

Do not change PyTorch, CUDA drivers, or the separate HDR and DeepLPF virtual environments.

## First experiment: no Photoshop conditioning

Use the *identical 1008 × 672 FLUX input* generated in the previous run. This avoids introducing preprocessing differences.

~~~bash
python poc/flux_klein_fidelity.py \
  --input output/flux-klein-test-1024/flux_input.png \
  --output-dir output/flux-fidelity-01 \
  --max-side 1024 \
  --strengths 0.50 0.75 \
  --steps 4 \
  --seed 0 \
  --offload sequential
~~~

The script loads the model once and renders both variants sequentially. Its input and output are in the local WSL project. The model may still use significant system RAM because of CPU offloading. Use a new output directory on every run.

Expected files:

| File | Meaning |
|---|---|
| `flux_input.png` | Exact image used for both edits |
| `edit_mask.png` | Diagnostic full-white inpaint mask |
| `flux_strength_050.png` | Conservative result |
| `flux_strength_075.png` | More permissive result |
| `changes_strength_050.png`, `changes_strength_075.png` | Per-pixel difference visualizations, not proof of fidelity |
| `report.json` | Model, strength, seed, prompts, descriptive brightness and pixel changes |

If the official pipeline rejects the distilled checkpoint, do not replace the model or switch to the larger base model without first diagnosing the compatibility problem.

## Compare against the same Photoshop reference

~~~bash
python poc/benchmark_editor_v3.py \
  --before output/flux-fidelity-01/flux_input.png \
  --after output/flux-fidelity-01/flux_strength_050.png \
  --reference photos/DSC01858.jpg \
  --output-dir output/flux-fidelity-review-050

python poc/benchmark_editor_v3.py \
  --before output/flux-fidelity-01/flux_input.png \
  --after output/flux-fidelity-01/flux_strength_075.png \
  --reference photos/DSC01858.jpg \
  --output-dir output/flux-fidelity-review-075

explorer.exe "$(wslpath -w /opt/hdr-real-state/output/flux-fidelity-01)"
~~~

Inspect 100% crops of the **same** window/leaf edges, furniture contours and sunlit regions. The numeric MAE to Photoshop is auxiliary because that reference may differ in perspective, sharpening and color grading. Favor faithful original architecture over a superficially attractive but invented photograph.

## Optional second experiment: Photoshop as appearance reference

Only after reviewing the unconditioned results, try the optional `--style-reference`. This sends a second image as reference conditioning — **not merely a color histogram**. The neural editor can copy shadows, furniture, structure or details from the second image despite the restrictive prompt.

~~~bash
python poc/flux_klein_fidelity.py \
  --input output/flux-klein-test-1024/flux_input.png \
  --style-reference photos/DSC01858.jpg \
  --output-dir output/flux-fidelity-style-01 \
  --strengths 0.50 0.75 \
  --max-side 1024 \
  --steps 4 \
  --seed 0 \
  --offload sequential
~~~

**No model is retrained.** This is just multi-image conditioning. Compare it against the same no-reference baseline before accepting it.

## Acceptance criteria and next decision

A result passes only if it improves room whites and exposure while preserving real window/exterior content, leaf/branch identity, wood tone, furniture, architecture, wall surfaces and original illumination logic. Check for introduced sunbeams, moved items and changes to outside scenery.

`report.json` documents image-wide mean absolute RGB change and brightness/highlight fractions; these **cannot establish structural integrity**. Visual high-resolution scrutiny against the original RAW-derived photograph remains essential.

If neither controlled-strength variant is sufficiently faithful, stop trying prompt-only changes. The next engineering task is a constrained/region-aware editing mechanism that demonstrably respects real sensor information.

No full-resolution FLUX rendering or professionally approved photo is claimed by this test.
