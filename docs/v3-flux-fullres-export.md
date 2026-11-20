# High-resolution export of a FLUX-guided photographic result

## Why an export stage is needed

The successful FLUX.2 Klein experiment produced a **768 × 512** image. That is enough to inspect the proposed lighting, contrast and color, but it is not a professional high-resolution output. Resizing those generated pixels to 6000 pixels does not restore genuine leaf texture, small furniture details or window frames. Generative super-resolution can also invent detail.

This independent stage **transfers smooth, bounded exposure/color adjustments suggested by FLUX to the actual original-resolution HDR scene**. It is not high-resolution diffusion inference; it is a source-preserving photographic render guided by a generative editor.

The implementation is at `poc/export_flux_fullres.py`. It is intentionally separate from the experimental neural model and from the existing HDR merge.

## Source image selection

Use all five existing paths:

1. `output/v2-new-photos/conventional_local.jpg`: original-resolution display RGB reference used as FLUX input; it also verifies final image geometry
2. `output/v2-new-photos/conventional_hdr.tiff`: original-resolution float32 HDR scene, preserving substantially more tonal precision than a JPEG
3. `output/v2-new-photos/highlight_mask.png`: the highlight mask generated along with that HDR, needed for consistent local tone mapping
4. `output/flux-klein-test-01/flux_input.png`: exact 768 × 512 image the neural editor saw
5. `output/flux-klein-test-01/flux_edit.png`: generative edited result at the same resolution

If the HDR TIFF or highlight mask was generated for a **different group of RAWs**, do not use it. The exporter checks geometry but cannot prove that two same-sized photos show the same property. Check source provenance first.

## Run (in the original HDR environment, not FLUX)

The script uses NumPy, Pillow and tifffile, all in `.venv`. No CUDA inference is needed and the original FLUX files are not modified.

```bash
cd /opt/hdr-real-state
git pull --ff-only
source .venv/bin/activate

python -m unittest discover -s tests -p 'test_export_flux_fullres.py' -v

python poc/export_flux_fullres.py \
  --source output/v2-new-photos/conventional_local.jpg \
  --linear-hdr output/v2-new-photos/conventional_hdr.tiff \
  --highlight-mask output/v2-new-photos/highlight_mask.png \
  --flux-input output/flux-klein-test-01/flux_input.png \
  --flux-edit output/flux-klein-test-01/flux_edit.png \
  --output-dir output/flux-final-fullres-01 \
  --strength 0.75 \
  --max-exposure-ev 0.65 \
  --max-color-log 0.12 \
  --blur-radius 12

explorer.exe "$(wslpath -w /opt/hdr-real-state/output/flux-final-fullres-01)"
```

The output folder must be new/empty; nothing is overwritten.

## Output and photographic properties

| File | Meaning |
|---|---|
| `photo_fullres.jpg` | Source-resolution RGB JPEG at quality 98, **4:4:4 chroma**, embedded sRGB ICC |
| `photo_fullres_16bit.tiff` | Source-resolution **16-bit RGB TIFF** with lossless Deflate compression and embedded sRGB ICC |
| `export_report.json` | Input paths, dimensions, output paths, limits and known shortcomings |

**What the 16-bit TIFF means:** with `--linear-hdr` the tone mapper starts with a float32 HDR image, applies continuous bounded adjustments and quantizes once to a 16-bit encoded RGB TIFF. It carries more tonal gradation than rendering from an already 8-bit JPEG. It is **not** a new 16-bit RAW file; some colors and highlights may already have been altered by the current RAW development or conventional fusion. Without `--linear-hdr`, the TIFF is still a 16-bit container but uses only 8-bit original display RGB as input.

**What high resolution means:** pixels retain the original image geometry and spatial details from the source, not from the 768-pixel FLUX output. The exporter never hallucinates new furniture, leaves or texture. It cannot restore already clipped edges in the original source. The high-resolution JPG and TIFF are **rendered interpretations of FLUX's color/exposure choices**, not an identical high-resolution replay of the generative edit.

**Controls:** exposure gains are clipped to +/-0.65 EV, color log ratios to +/-0.12 and the resulting maps are smoothly filtered; `--strength 0.75` limits how strongly the neural reference influences final RGB values. The changes can still introduce incorrect shadows or grading, so visual inspection remains mandatory.

## Compare at native resolution

```bash
python poc/benchmark_editor_v3.py \
  --before output/v2-new-photos/conventional_local.jpg \
  --after output/flux-final-fullres-01/photo_fullres.jpg \
  --reference photos/DSC01858.jpg \
  --output-dir output/flux-final-review-01
```

Send `photo_fullres.jpg`, `export_report.json` and, if possible, a 100%-zoom crop from `photo_fullres_16bit.tiff` containing tree leaves/window frames. The TIFF and source should also be opened in Photoshop or another color-managed editor at 100% zoom to judge actual fine detail, banding, halos and color fidelity.

**This export path is a technical quality preview, not a confirmed market-ready result.** Native high-resolution FLUX inference with verified image fidelity, trained adjustment prediction, source registration and full camera color management remain separate research tasks.
