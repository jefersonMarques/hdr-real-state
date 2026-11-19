# HDR Real Estate

**Offline-first neural HDR for professional real-estate photography.**

Target workflow: three bracketed camera RAW files (underexposed, reference, overexposed) -> real neural HDR fusion -> mixed-light white balance and natural window handling -> export -> local Windows desktop application.

## Project status

**Stage 0: experimental neural proof of concept. Not yet an MVP or an installer.**

The repository contains:

- [AFUNet RAW evaluation script](poc/hdr_afunet.py): uses the upstream neural model from a separate local checkout, takes three RAW images, reconstructs the image in overlapping patches, and exports float32 TIFF plus JPEG preview
- [Unit tests](tests/test_hdr_afunet.py): validates channel preparation, tile coverage and blending using a synthetic stand-in model
- [Research and license review](docs/open-source-evaluation.md): alternatives already available on GitHub
- [MVP plan and local usage](docs/mvp-plan.md): instructions and current limitations
- [Python dependencies](requirements.txt)

### Run the experimental neural pipeline

Install an appropriate Python/PyTorch environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Check out the original [AFUNet repository](https://github.com/eezkni/AFUNet), review its licensing/weights, and use the CLI described in [the MVP plan](docs/mvp-plan.md). The app never downloads or uploads photos.

**No model code or pretrained weight files are bundled here.** The upstream source and weights are external pending licensing review.

## Production objectives

- Three RAW camera exposures per photograph, preserving accurate color and camera metadata
- Neural reconstruction and mixed-light white balance, with professional real-estate finishing
- Windows installer with models packaged locally and no cloud inference
- AMD/NVIDIA/Intel acceleration where supported, CPU fallback
- High-resolution TIFF/JPEG export, batch processing, natural windows and neutral walls

## Important limitations

- Real RAW + upstream checkpoint end-to-end operation has **not** been validated yet.
- The initial PoC assumes equally spaced -2/0/+2 EV brackets; EXIF exposure extraction and exact camera profiles are pending.
- Initial PyTorch PoC selects CUDA when available, otherwise CPU. Cross-vendor Windows acceleration is future work.
- No calibrated white-balance AI, high-end real-estate tone mapping, GUI or offline installer exists yet.
- Code and pretrained weights from GPL/AGPL projects are not copied into this independent implementation.

A conventional Mertens HDR pipeline alone would not satisfy this project's mandatory AI requirement.
