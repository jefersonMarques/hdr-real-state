# V4 — Direct three-RAW neural photograph experiment

Status: **research PoC, published on `main` only**. This is the new direction requested after the v3 HDR-then-editor approach repeatedly lost real window/leaf detail. Older v2/v3 files remain available for baselines but are NOT part of the V4 neural input.

## Technical distinction

The current installed FLUX.2 Klein 4B supports **multiple reference images** in one inference. This allows one output conditioned simultaneously on three different real camera exposures. However, **multi-reference conditioning is NOT the same as a HDR-trained, exposure-calibrated three-image fusion network**.

The first V4 experiment explicitly measures whether this existing generative model can use multiple exposure references effectively, without claiming a physically correct radiance map or pixel-level fidelity.

A truly specialized multi-exposure AI must eventually have a model/data representation that learns correspondence and source confidence between aligned exposures, and predicts a photographic output; that stage requires architecture selection and paired professional targets.

## New execution sequence

```text
3 real camera RAWs + camera metadata
               |
               v
EXIF exposure sorting + shared reference RAW white balance
               |
               v
Independent camera RAW development (three different exposure levels)
               |
               v
under.png + reference.png + over.png
               |
               v
FLUX.2 Klein 4B multi-reference generation
               |
               v
neural_triplet.png + diagnostic report
```

**No conventional HDR is formed or loaded for model input.** The RAW decoding is necessary because FLUX accepts decoded RGB reference images, not RAW sensor mosaics. The underexposed frame is the source of authentic window information; the middle frame supplies target composition; the bright frame supplies interior shadow evidence.

The first PoC works at reduced resolution on the user's RTX 4060 8 GB with sequential CPU offload. Its JPEG is a **neural preview**, not a 24 MP source-faithful final export.

## Files

- `poc/prepare_triplet_v4.py` — reads exactly three camera RAWs, sorts by EXIF shutter × ISO / aperture², sets common WB from the middle exposure, produces three separately processed sRGB references and `triplet.json`
- `poc/flux_triplet_v4.py` — loads the ordered reference images and asks FLUX to produce a single professionally balanced photo using each exposure for its useful information
- `tests/test_triplet_v4.py` — synthetic tests for input validation, dimension handling, exposure ordering, output structure and provenance
- `docs/v4-triplet-neural-hdr.md` — this document

Model weights and user photographs remain local and outside the Git repository.

## Run on the existing machine

### Step 1 — verify project on main and prepare three RAWs

The RAW preparation uses `.venv`, where `rawpy` and the HDR dependencies are installed, and requires the existing `exiftool` executable. Leave the original RAWs untouched.

```bash
cd /opt/hdr-real-state
git pull --ff-only
source .venv/bin/activate

python -m unittest discover -s tests -p 'test_triplet_v4.py' -v

python poc/prepare_triplet_v4.py \
  --photos photos \
  --output-dir output/v4-triplet-input-01 \
  --max-side 768
```

The script discovers exactly three supported RAWs in `photos/`. JPEG/PNG Photoshop references do not count. If there are more than three RAWs, organize the new exposures into a separate folder before executing. The script prints the detected exposure order and relative exposure factors.

Result:
- `output/v4-triplet-input-01/under.png`
- `output/v4-triplet-input-01/reference.png`
- `output/v4-triplet-input-01/over.png`
- `output/v4-triplet-input-01/triplet.json`

These should visibly show **different** exposure levels. If the dark exposure has no real window detail or the bright exposure has no real room information, no neural model can reliably recover those missing pixels without inventing data.

### Step 2 — run FLUX using all three

Switch to the existing isolated environment `.venv-flux`, which has FLUX and working RTX 4060 CUDA offload. No CUDA reinstallation or second model download should be required if the previous FLUX experiments already loaded successfully.

```bash
cd /opt/hdr-real-state
source .venv-flux/bin/activate

python poc/flux_triplet_v4.py \
  --triplet-dir output/v4-triplet-input-01 \
  --output-dir output/v4-triplet-neural-01 \
  --steps 4 \
  --seed 0 \
  --offload sequential
```

Expected outputs:
- `neural_triplet.png` — actual FLUX-generated image based on **three** reference exposures
- `neural_triplet_preview.jpg` — high-quality **preview** JPEG (not 24 MP)
- `changes.png` — difference compared with middle exposure only, for diagnostic inspection
- `report.json` — exact input roles and EXIF, rendering metadata, model and known limitations

The additional image references increase memory and attention costs. Even on 768-pixel references the WSL RTX 4060 8 GB run can exceed available GPU/system memory. If that happens, retry the **preparation** with `--max-side 512` in a **new** input folder and use a new neural output folder. Do not change installed CUDA packages to hide an OOM.

### Step 3 — compare against the Photoshop result

```bash
python poc/benchmark_editor_v3.py \
  --before output/v4-triplet-input-01/reference.png \
  --after output/v4-triplet-neural-01/neural_triplet.png \
  --reference photos/DSC01858.jpg \
  --output-dir output/v4-triplet-review-01

explorer.exe "$(wslpath -w /opt/hdr-real-state/output/v4-triplet-review-01)"
```

The quality benchmark here compares the generated photo to the actual reference exposure and Photoshop edit. **This does not reintroduce the conventional HDR into the model**, it is only evaluation.

## Quality gates

Assess in order:

1. **Source fidelity:** are the same walls, fixtures, sofa, chairs, window frames, artwork and outside tree branches still there in exactly their captured positions?
2. **Exterior authenticity:** are the individual leaves and branches consistent with the darkest RAW (including dense vegetation, bright gaps and glass reflections), rather than plausible but invented new leaves?
3. **Real lighting:** no new shadows, sunlight beams, bright furniture highlights or altered window reflection patterns.
4. **Color:** neutral ceilings, no orange-wood oversaturation, natural vegetation and restrained interior white balance.
5. **Photography:** brightness and shadow detail closer to the approved retouched image than the middle RAW reference, with no geometry changes.
6. **Scalability:** characterize runtime/memory on an 8 GB RTX 4060. Fine details require a native-resolution solution rather than resizing the neural preview.

A lower RGB pixel error compared with the Photoshop edit is not sufficient alone. The difference may be affected by geometric/tonal edits made in the professional reference.

## Product architecture after the first experiment

```text
RAW ingest and calibrated metadata
  -> geometric registration and moving-object detection
  -> common camera WB / color-managed linear inputs
  -> supervised exposure-aware neural fusion + local color/tone prediction
  -> source-consistency/geometry constraints
  -> full-resolution renderer / export (JPEG and genuine 16-bit TIFF)
  -> optional manual approval UI and batch processing
```

A candidate network should consume the three aligned exposure inputs (or their multi-channel features), know shutter/ISO ratios, predict source-content confidence and learn to generate the professional final look. It should never require an imperfect gamma-compressed HDR JPEG as ground truth.

Future dataset groups:
- exactly matched RAW brackets, EXIF, camera/lens metadata
- professionally edited references with rights to train
- property/session-level train/validation/test splits
- varied interior/exterior lighting, white walls, vegetation and moving foliage

The model will need quantitative and visual quality validation on multiple properties and 100% crops. No pretrained general-purpose generative FLUX checkpoint should be presented as a physically trustworthy 24 MP multi-exposure HDR solution.

## Development rules

- **Only `main`** until explicitly authorized otherwise.
- No branches, PRs, GitHub Actions or hidden publication steps.
- Preserve the previous experiments and their outputs; use unique output directories.
- English-only code identifiers; comments, when necessary, technical and impersonal Brazilian Portuguese.
- Keep model checkpoints, photography, virtual environments and generated images out of the repository.
- Do not claim inferential success before the user has tested the actual three RAWs on WSL.
