# v3 first model test: DeepLPF vs. conventional HDR vs. Photoshop

## Goal

Prove or falsify that an actual pretrained **neural photo-editing** model contributes a useful improvement over the non-neural HDR result. Do not execute AFUNet in this test. Do not apply v2 hybrid masks. Do not change any old outputs.

The first candidate is [DeepLPF](https://github.com/sjmoran/deeplpf-image-enhancement), a neural editor which predicts interpretable local photographic filter parameters. Its MIT-licensed code, PyTorch implementation and pretrained Adobe FiveK retouch checkpoint were inspected. **The pretrained default does not include the optional color-mixing head**, and its FiveK training is not equivalent to professional real-estate retouching. Better output is a hypothesis to test, not a promised result.

## Inputs

- `output/v2-new-photos/conventional_local.jpg`: existing conventional HDR preview (our baseline input)
- `photos/DSC01858.jpg`: user-provided professional Photoshop retouch, if present; optional
- The original three RAWs are not required for this **first editing-model inference**; the network edits the rendered conventional RGB image.

## Installation: isolated editor virtual environment (WSL)

The original `.venv` contains a working CUDA-compatible PyTorch. Preserve it. Create **a separate** `.venv-editor` for the new model and its dependencies. This requires an additional installation of CUDA/PyTorch packages but avoids breaking the AFUNet experiment.

```bash
cd /opt/hdr-real-state

git pull --ff-only

test -d "$HOME/deeplpf-image-enhancement/.git" || \
  git clone https://github.com/sjmoran/deeplpf-image-enhancement.git \
  "$HOME/deeplpf-image-enhancement"

python3 -m venv .venv-editor
source .venv-editor/bin/activate
python -m pip install --upgrade pip

python -m pip install \
  torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install -e "$HOME/deeplpf-image-enhancement"

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m unittest discover -s tests -p 'test_benchmark_editor_v3.py' -v
```

Do **not** install `$HOME/deeplpf-image-enhancement/requirements.txt`, which explicitly pins other Torch versions and might replace your working CUDA build. The editor's editable package declaration already provides the remaining dependencies and allows the installed compatible Torch version to satisfy its minimum requirement. The new environment must print `True` for CUDA before inference.

## Run editor with bounded GPU memory

**Real WSL evaluation on 2026-11-20:** the initial full-resolution call exhausted 8 GB of VRAM on an RTX 4060 in `unet.py`, even though PyTorch correctly reported `CUDA: True` and loaded the DeepLPF checkpoint. This is not a CUDA driver failure or an absent model. The U-Net processes a full-size feature map, so memory scales with the input's pixel count. The upstream CLI has no `--max-side` option.

Prepare a **new** 1024-pixel-long-edge input (existing output files remain intact):

```bash
cd /opt/hdr-real-state
git pull --ff-only
source .venv-editor/bin/activate

python -m unittest discover -s tests -p 'test_prepare_editor_input.py' -v
python -m unittest discover -s tests -p 'test_benchmark_editor_v3.py' -v

python poc/prepare_editor_input.py \
  --source output/v2-new-photos/conventional_local.jpg \
  --output output/v3-input-1024/conventional_local.png \
  --max-side 1024

deeplpf enhance output/v3-input-1024/conventional_local.png \
  --out output/v3-deeplpf-1024 \
  --device cuda
```

Expected output, according to the current upstream CLI:
`output/v3-deeplpf-1024/conventional_local_enhanced.png`.

This executes the pretrained editor, without additional hand-tuned HDR masks. The input is downscaled **before** neural inference, not afterward. An output at 1024 pixels is diagnostic only; this does not yet solve full-resolution professional inference.

**If GPU memory is still insufficient**, create a second input with `--max-side 768` and new output paths (e.g. `v3-input-768`, `v3-deeplpf-768`); do not overwrite the existing trial. Avoid reducing the input far below this range because photo-quality judging will become unreliable.

## Compare against professional reference

```bash
cd /opt/hdr-real-state
source .venv-editor/bin/activate

reference_args=()
if [[ -f photos/DSC01858.jpg ]]; then
  reference_args=(--reference photos/DSC01858.jpg)
fi

python poc/benchmark_editor_v3.py \
  --before output/v3-input-1024/conventional_local.png \
  --after output/v3-deeplpf-1024/conventional_local_enhanced.png \
  --output-dir output/v3-editor-review-1024 \
  "${reference_args[@]}"

explorer.exe "$(wslpath -w /opt/hdr-real-state/output/v3-editor-review-1024)"
```

The professional JPEG is resized automatically **for the numerical comparison**. The neural output and its baseline input are the same dimensions. Output: `comparison.jpg` and `report.json`, without altering the source photos.

A future full-resolution renderer must preserve model context and avoid patch seam artifacts. Simple independent 1024-pixel tile inference followed by stitching is **not** automatically equivalent to a global photographic edit.

**Judge first by visual quality** — ceiling/white neutrality, preserved external foliage, saturation, true architectural edges, shadows and textures. The report also records descriptive differences including raw RGB MAE, hashes and brightness percentiles. MAE alone is not a model quality grade; differences in perspective, crop, retouch decisions or source color transform can influence it.

Send these files for review:
1. `output/v3-editor-review-1024/comparison.jpg`
2. `output/v3-editor-review-1024/report.json`
3. `output/v3-deeplpf-1024/conventional_local_enhanced.png`

If the pretrained DeepLPF is worse, record that failure and move to an adaptive LUT candidate; do **not** hide it with a new hand-selected postprocessing preset. If it works, repeat on multiple independent property scenes before productizing it.

## Practical caveats

- `photos/DSC01858.jpg` is only a **reference** for the current evaluation, not a training dataset.
- DeepLPF's default trained checkpoint does not have cross-channel color mixing; the prototype may retain the yellow cast of the incoming conventional HDR. Specialized real-estate training or a candidate with native white-balance support may be necessary.
- A simple display RGB source is sufficient for model qualification. Production-quality editing should preserve high bit depth, color profiles, geometric registration, and RAW exposure detail, and may require model adaptation.
- Do not commit the photo samples, model weights, new virtual environment or other local assets.
- No offline Windows installer is being developed until the editor passes the image-quality gate.
