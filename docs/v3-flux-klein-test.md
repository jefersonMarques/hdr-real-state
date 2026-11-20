# FLUX.2 Klein 4B: local real-estate photo edit experiment

## Current status

This experiment tests whether a **generative** image editor can approach the supplied professional Photoshop edit. It is **not part of the approved faithful HDR production pipeline**. It can change leaves, window scenery, furniture or geometry despite instructions to preserve them.

The official model is `black-forest-labs/FLUX.2-klein-4B`, licensed Apache-2.0. Its BF16 model requires approximately 13 GB VRAM under typical assumptions. The user's NVIDIA RTX 4060 has 8 GB; therefore the script defaults to **sequential CPU offload** and downscaling to a maximum 768-pixel side. This may be very slow and requires sufficient system RAM; even with offloading, success is **not guaranteed**.

## Verified local machine status

The user has installed these packages in `.venv-flux`:

- PyTorch 2.13.0 + CUDA 12.9 (separate environment, do not change the working HDR `.venv` or DeepLPF `.venv-editor`)
- Diffusers 0.44.0 development checkout
- Transformers, Accelerate, SentencePiece, Safetensors and Pillow

Their CUDA check returned `True` and identified an NVIDIA GeForce RTX 4060 Laptop GPU.

The previous attempt to run Python source directly in Bash failed with messages such as `Command 'from' not found`. The subsequent `python poc/test_flux_klein_edit.py` failed because the file had not been created. Those were **shell and missing-file errors**, not FLUX inference failures. The correct script has now been committed to `poc/test_flux_klein_edit.py`.

## Run

~~~bash
cd /opt/hdr-real-state
git pull --ff-only
source .venv-flux/bin/activate

python -m unittest discover -s tests -p 'test_flux_klein_edit.py' -v

python poc/test_flux_klein_edit.py   --input output/v2-new-photos/conventional_local.jpg   --output-dir output/flux-klein-test-01   --max-side 768   --offload sequential   --steps 4

explorer.exe "$(wslpath -w /opt/hdr-real-state/output/flux-klein-test-01)"
~~~

The first successful invocation downloads model weights into the Hugging Face local cache, potentially requiring substantial disk space. The image and output files remain local; model downloads require Internet only for their initial acquisition.

Results:
- `output/flux-klein-test-01/flux_edit.png`
- `output/flux-klein-test-01/report.json`

The output directory must be empty or nonexistent. For retries, use a new output directory. Do not delete cached models or existing experiments.

## Comparison to Photoshop

When `flux_edit.png` exists:

~~~bash
python poc/benchmark_editor_v3.py   --before output/v2-new-photos/conventional_local.jpg   --after output/flux-klein-test-01/flux_edit.png   --reference photos/DSC01858.jpg   --output-dir output/flux-klein-review-01
~~~

**Note:** the existing editor benchmark refuses differing baseline/editor dimensions. Therefore to perform a fair comparison, first prepare the same reduced source using `poc/prepare_editor_input.py` with `--max-side 768` and use that image as `--before`. The source must have the same dimensions as the FLUX inference input. The Photoshop reference will be resized as an exploratory visual reference. Do not judge pixel MAE in isolation.

## Memory troubleshooting

If CPU RAM is insufficient or the model still hits CUDA OOM with `--offload sequential`, stop and report the complete error. A quantized checkpoint or ComfyUI GGUF workflow may be required. Do not repeatedly reinstall CUDA, replace HDR environments or silently switch to a remote inference service.

This is a controlled *visual feasibility experiment*, not production release-quality editing.
