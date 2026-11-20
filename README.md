# HDR Real Estate

**Offline-first neural HDR for professional real-estate photography.**

Target workflow: three bracketed camera RAW files (underexposed, reference, overexposed) -> real neural HDR fusion -> mixed-light white balance and natural window handling -> export -> local Windows desktop application.

## Project status

**V5.2 RAW-sensor-guided HDR fusion is now available as an opt-in experiment on `main`. The V5/V5.1 camera RAW baseline remains unchanged by default. Both are classical photographic fusion; neither is the final neural retoucher or a verified professional export.**

A first full-resolution Sony ARW neural test ran locally on an RTX 4060 through PyTorch CUDA. The photographed bedroom's generated preview was substantially underexposed and the window lacked useful exterior contrast. The cause has not been isolated; the neural TIFF, original brackets and preview mapping must be compared before choosing a correction.

## Current experiment: Sony RAW confidence fusion (V5.2)

The actual camera audits of three Sony ARW exposures found a strongly repeated **RAW code 16372** beyond the camera-declared white level 15360, near LibRaw's global white 16380. We cannot equate 15360 with confirmed physical clipping.

The new optional [V5.2 RAW-guided fusion](poc/merge_hdr_raw_v52.py) generates per-Bayer-block confidence masks directly from the camera sensor mosaics, aligns them with the developed linear RAW RGB frames and uses them to weight the three exposures. It preserves the baseline V5 output and existing settings. The new [seven tests](tests/test_raw_guided_v52.py) and [WSL comparison guide](docs/v52-raw-guided-hdr.md) are published but have **not yet been executed on the user's original RAWs**.

**Important:** this is sensor-guided HDR merging of decoded linear RGB, **not** direct CFA sample fusion and **not a trained neural model**. It may improve source selection but cannot recreate highlights already lost during RAW development. Use new output folders and compare V5.1 versus V5.2 on exactly the same photo session.

## Native RAW sensor diagnostics (current V5.2 preparation)

The [RAW sensor-level audit](docs/v5-raw-sensor-audit.md) adds a **read-only** inspection of the three original RAW mosaics, including per-CFA-channel black/white levels and fractions of pixels near sensor clipping. [Script](poc/inspect_raw_sensor_v5.py) · [Tests](tests/test_raw_sensor_v5.py). This identifies which exposures hold **real sensor headroom** rather than relying solely on already-developed RGB. The HDR merge itself has **not yet** been switched to a new sensor-native method; that will require validating the Sony source metadata, CFA orientation and registration.

The older tone-map test that rejected exactly-1.0 normalized output was corrected to accept the documented 0-to-1 range. This changes the test, **not** HDR clipping behavior.

## V5 — camera RAW HDR fusion baseline (new test)

The [V5 real HDR baseline](docs/v5-real-hdr-pipeline.md) has been implemented **directly on `main`**, without changing V4 or older experiments. It decodes three original RAW exposures with a shared camera white balance, registers frame translations, merges reliable source data into a float32 HDR, performs local tone mapping and exports full-resolution JPEG / 16-bit TIFF.

- [V5 core](poc/v5_core.py): linear RAW, translation registration, confidence-weighted merge, tone map and export
- [V5 independent automated tests](tests/test_v5_pipeline.py)
- [V5 WSL run commands and quality checks](docs/v5-real-hdr-pipeline.md)

**Important:** V5 is a photographic (non-neural) fusion baseline for future comparison with **true exposure-aware neural** models. It is not the final AI editor; its new synthetic tests and real photographic quality have not yet been confirmed on the user's WSL camera RAWs. Files are committed but no claim of production image quality is made.

## V4 — direct three-exposure neural experiment

The current development route is [V4: three camera RAWs directly into neural generation](docs/v4-triplet-neural-hdr.md).

- [Prepare three RAW images with one white balance, EXIF-ordered](poc/prepare_triplet_v4.py)
- [Run FLUX.2 Klein with all three reference images](poc/flux_triplet_v4.py)
- [V4 tests](tests/test_triplet_v4.py)

The scripts commit directly to `main`, with no branches, pull requests or GitHub Actions. This first test uses FLUX multi-image **reference conditioning**, not a learned sensor-linear fusion model. A specialized high-resolution exposure-aware neural HDR editor remains a later research milestone. No actual WSL three-RAW model run has yet been validated.

## Earlier experiments (v2/v3, retained as historical baselines)

## New direction: neural photo editor

The first real-estate benchmark showed that three AFUNet neural HDR checkpoints did not outperform the conventional HDR against the Photoshop reference. Kalantari/Tel outputs showed significant exposure and color problems; the Hu checkpoint degenerated into a nearly uniform image. Instead of continuing HDR-only model adjustments, the project now tests a **photo-editing AI on a reliable conventional HDR render**.

- [v3 architecture, constraints and roadmap](docs/v3-architecture.md)
- [First RTX 4060 photo-editor trial: DeepLPF](docs/v3-first-model-test.md)
- [Editorial AI benchmark script](poc/benchmark_editor_v3.py)
- [Pretrained 3D LUT neural editor](poc/benchmark_3dlut_v3.py): independent full-resolution photo color/tone AI without legacy CUDA extensions; [WSL validation commands](docs/v3-adaptive-lut-test.md)
- [FLUX.2 Klein experiment](poc/test_flux_klein_edit.py): generative image editing evaluation with 8 GB GPU offloading; [WSL instructions](docs/v3-flux-klein-test.md)
- [FLUX fidelity experiment](poc/flux_klein_fidelity.py): actual image-initialized editing with controlled denoising strength (0.50/0.75), conservative prompts and optional Photoshop style reference; [WSL test](docs/v3-flux-fidelity.md). Experimental; neither run nor visual improvement is yet verified.
- [Native-resolution FLUX-guided export](poc/export_flux_fullres.py): renders high-quality JPEG and high-bit-depth TIFF from the original HDR pixels using bounded FLUX edit adjustments; [commands and quality limits](docs/v3-flux-fullres-export.md)
- [Project continuation and decisions](AGENTS.md)

The DeepLPF model is **not bundled**. Its first RTX 4060 run succeeded at 1024 × 684, but its output failed the visual quality gate against Photoshop; see [DeepLPF evaluation and metrics](docs/v3-deeplpf-evaluation.md). A full-resolution run exceeded 8 GB VRAM. The current priority is a different learned photographic editor (adaptive color/LUT with local control) rather than further hand-written postprocessing. Prior experiments remain unchanged.

The repository contains:

- [AFUNet RAW evaluation](poc/hdr_afunet.py): experimental neural model inference using an external upstream checkout and checkpoint; outputs floating-point HDR TIFF and an initial JPEG preview
- [Independent AFUNet checkpoint benchmark](poc/benchmark_afunet.py): compare Kalantari, Hu and Tel weights without hybrid masks, using the same conventional HDR and optional Photoshop reference; see [qualification guide](docs/neural-qualification.md)
- [Pipeline v2](poc/hdr_pipeline_v2.py): auto-orders three camera RAWs using actual EXIF exposure, runs the neural model, creates linear conventional HDR and blends protected real highlights into the HDR working image **before** tone mapping; see [WSL instructions and limitations](docs/pipeline-v2.md)
- [HDR diagnostic comparison](poc/hdr_diagnostics.py): reads exposure metadata, develops the three RAWs, produces a simple conventional HDR baseline and re-renders the previously saved neural TIFF with the same tone map
- [Window detail recovery experiment](poc/hdr_window_recovery.py): tests a source-guided highlight mask using saved underexposed and reference previews, without rerunning AFUNet or replacing the HDR pipeline
- [Quality validation guide](docs/quality-validation.md): exact WSL commands, comparison workflow, known limitations and decision gate
- [Tests](tests/): synthetic tests for neural tiling, exposure weighting and preview conversion
- [Research and license review](docs/open-source-evaluation.md): external open-source candidates
- [MVP plan](docs/mvp-plan.md): staged product goals

## Reproduce the quality comparison on WSL

With the three example Sony RAWs in `photos/`, the existing neural output in `output/`, and ExifTool available:

```bash
cd /opt/hdr-real-state
source .venv/bin/activate
python -m unittest discover -s tests -v

python poc/hdr_diagnostics.py \
  --under photos/DSC02055.ARW \
  --reference photos/DSC02054.ARW \
  --over photos/DSC02056.ARW \
  --neural-hdr output/teste2.tiff \
  --neural-preview output/teste2_preview.jpg \
  --output-dir output/review-test2
```

Inspect `output/review-test2/comparison.jpg` and `metrics.json`. The script **does not rerun AFUNet**. Add `--final-reference photos/final.jpg` if the corresponding professional edit is available. Instructions and limitations: [quality validation](docs/quality-validation.md).

Run the original neural proof of concept using the [MVP plan](docs/mvp-plan.md). **No upstream AFUNet source or pretrained weights are bundled**; verify their redistribution terms before commercial use.

## Product goals

- Three RAW camera exposures per photograph with correct metadata and camera color
- Genuine neural HDR reconstruction, with an auditable conventional baseline
- Accurate mixed-light white balance, bright natural interiors and preserved exterior window detail
- Local Windows desktop installer; no mandatory cloud services
- Compatible GPU inference and CPU fallback
- Professional batch TIFF/JPEG export

## Current limitations

- The conventional baseline does not align photos or deghost moving objects.
- Neural input calibration against AFUNet's benchmark dataset is pending; current RAW developing may not match the model's training input.
- Neither pathway has a professional local tone mapper, calibrated color workflow or automatic neutral-wall correction.
- The RTX 4060/CUDA 12.8 environment was confirmed by a local user test; cross-vendor Windows acceleration is unimplemented.
- Quality metrics are descriptive image statistics only, not a score of HDR correctness.
- Code from GPL/AGPL alternative editors has not been copied. Commercial rights for third-party neural weights must be reviewed.
