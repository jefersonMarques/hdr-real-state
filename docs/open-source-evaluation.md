# Open-source evaluation — 2026-11-19

Technical references for an offline Windows real-estate HDR product built around **three RAW exposures and real neural inference**.

| Project | URL | Reuse potential | Key limitation |
| --- | --- | --- | --- |
| CalibRaw | https://github.com/Duecki1/CalibRaw | Full RAW editor, HDR merge, GPU and local AI | GPL-3.0-or-later; HDR merge is not a neural HDR model |
| RapidRAW | https://github.com/CyberTimon/RapidRAW | RAW GUI, alignment, deghosting, local AI | AGPL-3.0; Rust/Tauri; HDR merge is a classical pipeline |
| AFUNet | https://github.com/eezkni/AFUNet | Three-exposure HDR reconstruction neural network and checkpoints | README says MIT, but no standalone license file found; checkpoint terms require review |
| HDR-Transformer | https://github.com/megvii-research/HDR-Transformer | Neural reconstruction and deghosting | Apache-2.0 reference implementation, separate PyTorch port and checkpoints to audit |
| EMEF | https://github.com/medalwill/EMEF | Neural fusion framework, MIT code, pretrained weights available separately | Documented workflow mainly uses two input exposures |
| HDRMerge | https://github.com/jcelaya/hdrmerge | Camera RAW HDR DNG merge | GPL-3.0; no AI-based real-estate finishing |
| Zigzag Studio | https://github.com/jnsfeir/Zigzag-Studio-Photo-Merging | Three-frame interior exposure fusion baseline | JPEG/PNG and conventional OpenCV Mertens, no neural HDR |
| PhotoManager | https://github.com/Hawkynt/PhotoManager | Desktop RAW workflow, HDR merge, local AI image enhancements | LGPL-3.0; HDR reconstruction itself is not neural |

## Decision

Use an **independent** Python proof of concept based on externally checked-out AFUNet source. Do not copy GPL/AGPL project code or vendor pretrained weights until redistribution rights are verified.

AFUNet's official test path is benchmark-focused. Its published `test.py` builds an 18-channel network and uses a Kalantari checkpoint. The local harness in this repo provides three camera RAW inputs, 18-channel preparation, CPU/CUDA inference in overlapping patches, linear TIFF output and a JPEG preview.

## Product risk gates

- Independent verification of the licenses of pretrained checkpoints and dependencies.
- Compare RAW development against the LDR TIFF training domain and verify camera profiles.
- Preserve accurate mixed-light white balance, straight perspective and window details.
- Handle exposure metadata, alignment, ghosting, color management and final tone mapping.
- Test AMD and Intel GPU paths; initial PyTorch harness falls back to CPU on unsupported cards.
- Validate real full-resolution RAWs before calling this a product.
