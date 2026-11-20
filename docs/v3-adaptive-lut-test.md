# v3: pretrained neural Image-Adaptive 3D LUT editor

## Purpose

The DeepLPF model successfully executed on the RTX 4060 but did not improve the Photoshop comparison (approximate RGB MAE: 42.3678 conventional, 43.0569 neural). We are testing a different pretrained **photo retouching AI**: https://github.com/HuiZeng/Image-Adaptive-3DLUT.

This model predicts the weights of three learned 3D LUTs based on the photographed scene and applies the resulting color/tone transformation to pixels. The original implementation requires old custom CUDA extensions. Our independent PyTorch adapter at poc/benchmark_3dlut_v3.py uses native trilinear grid_sample with the official classifier structure. The global LUT is shared by all image bands, and therefore does not generate tile-specific grading changes.

The first five local synthetic CPU tests passed, but **actual upstream pretrained checkpoints and CUDA inference have not yet been tested by the user**. This model has global adaptive color control, not semantic local editing, window reconstruction or independent relighting. Better results must be demonstrated, not assumed.

The upstream code carries Apache-2.0. The weights are hosted in the upstream source repository, but their redistribution and the underlying MIT-Adobe FiveK training-data terms need verification before commercial inclusion. Keep the upstream checkout external for evaluation.

## Install and verify (WSL)

Use the existing compatible editor virtual environment. Do not reinstall CUDA/PyTorch.

~~~bash
cd /opt/hdr-real-state
git pull --ff-only
source .venv-editor/bin/activate

test -d "$HOME/image-adaptive-3dlut/.git" || \
  git clone https://github.com/HuiZeng/Image-Adaptive-3DLUT.git \
  "$HOME/image-adaptive-3dlut"

python -m unittest discover -s tests -p 'test_benchmark_3dlut_v3.py' -v
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

ls -lh "$HOME/image-adaptive-3dlut/pretrained_models/sRGB/classifier.pth"
ls -lh "$HOME/image-adaptive-3dlut/pretrained_models/sRGB/LUTs.pth"
~~~

GPU mode requires CUDA True and both official checkpoint files.

## Run AI editing at the original HDR preview resolution

~~~bash
python poc/benchmark_3dlut_v3.py \
  --input output/v2-new-photos/conventional_local.jpg \
  --checkpoint-dir "$HOME/image-adaptive-3dlut/pretrained_models/sRGB" \
  --output-dir output/v3-adaptive-lut-01 \
  --device cuda \
  --rows-per-chunk 256
~~~

Outputs:
- output/v3-adaptive-lut-01/adaptive_lut_enhanced.png
- output/v3-adaptive-lut-01/model_report.json (original image and checkpoint hashes, weights, device, dimensions)

This preserves the original pixels' structure and does not postprocess the learned editor with manually tuned brightness or masks. The folder must not already contain files. If GPU memory is exhausted, retry in a **new folder** with --rows-per-chunk 128; the actual GPU memory footprint still needs to be verified.

## Compare to the same Photoshop photograph

~~~bash
python poc/benchmark_editor_v3.py \
  --before output/v2-new-photos/conventional_local.jpg \
  --after output/v3-adaptive-lut-01/adaptive_lut_enhanced.png \
  --reference photos/DSC01858.jpg \
  --output-dir output/v3-adaptive-lut-review-01

explorer.exe "$(wslpath -w /opt/hdr-real-state/output/v3-adaptive-lut-review-01)"
~~~

Send comparison.jpg and report.json from the review folder, plus model_report.json from the LUT folder. If the Photoshop reference is not on the disk, omit the --reference line.

Review the ceiling/white neutrality, vegetation color, furniture materials, outdoor highlights, structural details, and whether the learned edit improves actual professional quality. Approximate RGB MAE alone is not a valid perceptual grade, because the Photoshop target may be cropped, shifted, locally edited or differently color-managed.

If the global learned LUT does not add useful quality, record the negative result and prioritize a spatially aware photo retoucher or licensed expert training pairs. Do not cover failed learning with hand-selected postprocessing presets.
