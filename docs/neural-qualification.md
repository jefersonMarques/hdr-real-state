# Neural checkpoint qualification gate

## Current decision

The existing HDR v2 output did not demonstrate a useful neural quality advantage over a conventional RAW blend on the first professionally edited real-estate example. Stop tuning JPEG masks or shipping a neural model until neural inference itself is validated. The v2 experiment is retained for comparison, not promoted to production.

The current research compares three **AFUNet** pretrained weights from the original upstream repository: Kalantari, Hu and Tel. All three are the same network architecture but were trained on different HDR benchmarks. The benchmark is not a replacement for testing the model on its official native testing datasets.

A separate PyTorch implementation of HDR-Transformer exists at https://github.com/liuzhen03/HDR-Transformer-PyTorch, with pretrained weights described in its README. It requires an independent environment/license/dependency check before adoption. The official MegEngine implementation at https://github.com/megvii-research/HDR-Transformer carries an Apache-2.0 LICENSE; the PyTorch port and its weights do **not** automatically inherit that license.

## Prerequisites

- Existing WSL machine with the working PyTorch CUDA 12.8 virtual environment and the original AFUNet checkout
- The same three Sony RAWs used for v2, left in `photos/`
- `output/v2-new-photos/conventional_hdr.tiff` and `output/v2-new-photos/report.json` from **the same three photographs**
- Three upstream checkpoints at `$HOME/AFUNet/pretrain_model/Kalantari.pth`, `Hu.pth`, and `Tel.pth`
- Optionally, the edited Photoshop photograph in `photos/DSC01858.jpg`

Note: The Photoshop JPEG was uploaded to the chat but may not exist on the user's WSL disk. Copy it into `photos/` using Windows Explorer. The benchmark ignores JPEGs while discovering RAWs. If the professional JPEG is missing, remove `--professional-jpeg` from the command.

## Run on WSL

```bash
cd /opt/hdr-real-state
git pull --ff-only
source .venv/bin/activate

python -m py_compile poc/benchmark_afunet.py
python -m unittest discover -s tests -v

time python poc/benchmark_afunet.py \
  --photos photos \
  --afunet-source "$HOME/AFUNet" \
  --checkpoint-dir "$HOME/AFUNet/pretrain_model" \
  --baseline-hdr output/v2-new-photos/conventional_hdr.tiff \
  --baseline-report output/v2-new-photos/report.json \
  --professional-jpeg photos/DSC01858.jpg \
  --output-dir output/afunet-benchmark-01 \
  --max-side 1280 \
  --tile-size 128 \
  --overlap 32

explorer.exe "$(wslpath -w /opt/hdr-real-state/output/afunet-benchmark-01)"
```

Use a new output folder for a repeated run. The CLI refuses to overwrite an existing nonempty destination. The baseline report must list the same filenames and exposure order as the current RAW input. If the actual RAW contents were overwritten but the names stayed identical, this check is not sufficient; regenerate the baseline for that group.

The comparison uses one conventional HDR TIFF and three neural-only predictions; it does **not** use the v2 hybrid pipeline, highlight masks or local color adjustments. All four HDR images pass through the same experimental **global** tone-mapping function. Its exposure normalization is computed independently for each image, so it does not prove radiometric accuracy. Neural model weights are not copied into this repository.

## Result files

- `conventional_preview.png`: baseline RAW HDR with the same preview rendering
- `kalantari_preview.png`, `hu_preview.png`, `tel_preview.png`: model-only HDR previews
- `kalantari_hdr.tiff`, `hu_hdr.tiff`, `tel_hdr.tiff`: untouched model predictions, float32
- `comparison.jpg`: side-by-side contact sheet, with Photoshop reference when provided
- `benchmark.json`: inputs, exposure order, device, inference duration, checkpoint hashes, raw HDR intensity statistics and image comparisons

The optional Photoshop photo is not modified or uploaded to any service.

## How to read the benchmark

- `mean_absolute_difference_from_baseline_0_255` shows **how different** each prediction is from the conventional result, **not** whether it is better.
- `approximate_rgb_mae_from_professional_0_255` is only an exploratory image-space comparison. Photoshop may differ by crop, perspective, lens correction, local edits, color grading and sharpness. A higher/lower value alone is not a quality verdict.
- `raw_hdr_median` and `raw_hdr_p99` help detect output scaling differences or degeneracy across checkpoints; they are not HDR quality scores.
- All model runs use the same three camera RAWs and identical input development, exposure ratios, patch size and tone map; this isolates checkpoint selection while keeping **known** RAW-to-model input calibration errors constant.
- The default downsample `--max-side 1280` speeds up exploratory comparisons. Rerun promising candidates at full resolution with `--max-side 0` and inspect artifacts at 100%.
- Confirm the exterior window remains genuinely textured and the white ceiling neutral. Examine edges and thin patterns for ghosts or false detail.
- This single scene **cannot** prove overall quality or generalization. A larger, authorized real-estate dataset with professional references is necessary.

## Next decision

1. Check if different pretrained weights deliver meaningfully different, plausible reconstruction and better visual fidelity than conventional RAW blending.
2. If all AFUNet results are poor, validate the upstream official AFUNet checkpoint on its **native** public benchmark before blaming the network itself. The real-estate RAW conversion/input preprocessing may be incompatible with its training data.
3. Only after the native checkpoint is proven functional, port a second neural candidate or gather licensed, paired real-estate training examples for adaptation.

Never treat GPU activity alone as evidence of photographic quality or model advantage.
