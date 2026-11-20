# HDR Real Estate

**Offline-first neural HDR for professional real-estate photography.**

Target workflow: three bracketed camera RAW files (underexposed, reference, overexposed) -> real neural HDR fusion -> mixed-light white balance and natural window handling -> export -> local Windows desktop application.

## Project status

**v3 design and first neural photo-editor evaluation are ready to test. This is not yet a production editor, completed neural editor pipeline, or an installer.**

A first full-resolution Sony ARW neural test ran locally on an RTX 4060 through PyTorch CUDA. The photographed bedroom's generated preview was substantially underexposed and the window lacked useful exterior contrast. The cause has not been isolated; the neural TIFF, original brackets and preview mapping must be compared before choosing a correction.

## New direction: neural photo editor

The first real-estate benchmark showed that three AFUNet neural HDR checkpoints did not outperform the conventional HDR against the Photoshop reference. Kalantari/Tel outputs showed significant exposure and color problems; the Hu checkpoint degenerated into a nearly uniform image. Instead of continuing HDR-only model adjustments, the project now tests a **photo-editing AI on a reliable conventional HDR render**.

- [v3 architecture, constraints and roadmap](docs/v3-architecture.md)
- [First RTX 4060 photo-editor trial: DeepLPF](docs/v3-first-model-test.md)
- [Editorial AI benchmark script](poc/benchmark_editor_v3.py)
- [Pretrained 3D LUT neural editor](poc/benchmark_3dlut_v3.py): independent full-resolution photo color/tone AI without legacy CUDA extensions; [WSL validation commands](docs/v3-adaptive-lut-test.md)
- [FLUX.2 Klein experiment](poc/test_flux_klein_edit.py): generative image editing evaluation with 8 GB GPU offloading; [WSL instructions](docs/v3-flux-klein-test.md)
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
