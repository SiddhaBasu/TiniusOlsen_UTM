# Codebase Reference

Detailed, file-by-file record of what's in this repo, what's actually live vs. archived, and known
inconsistencies to be aware of. See [README.md](../README.md) for the project overview.

## Active pipeline (repo root)

These are the files the working system actually depends on at runtime.

### `tinius_gui.py` — GUI entry point
PyQt5 desktop app; the thing an operator actually runs on the Raspberry Pi. Shows two live camera
feeds (top-down + side, via OpenCV `VideoCapture`), a "Capture & Measure" button, an outline preview,
and a measurements panel. Capture writes `captures/specimen_<unix_timestamp>.jpg`, then runs
`outline.measure("captures/<file>")` on a background `QThread` (via `MeasureWorker`) so the UI doesn't
freeze, and displays the resulting `outputs/<base>_outline.png` plus the metrics dict (length, width,
neck width, surface area, shape, inferred material + confidence, ASTM standard).

This file previously had unresolved git merge-conflict markers running through ~10 blocks (camera
index, capture directory, worker signature, layout sizing, and the metrics-display code, which also
had an f-string syntax bug and a width/neck-width label bug). All of that has been resolved in favor
of the branch that matches `outline.py`'s actual current signature (`measure(image_name)`, single
arg, material inferred by the CNN — there's no manual material override anymore).

### `outline.py` — the core pipeline
Imported directly by the GUI (`from outline import measure`). Given an image path, it:
1. Segments the specimen from the background (`rembg`, with a luminance-based fallback).
2. Computes length / max width / neck width / surface area via PCA-alignment + a smoothed width
   profile, and infers shape (`DOGBONE` vs `COUPON`) from the width-to-neck ratio.
3. Samples ~100 random 16×16 patches from inside the segmented region and classifies each with a
   small CNN (`MaterialPatchNet`), majority-voting to a material label + confidence.
4. Maps (material, shape, dimensions) to an ASTM standard via `astm_standard()`.
5. Saves `outputs/<base>_outline.png` and `outputs/<base>_annotation.png`, and returns a metrics dict.

Loads its model checkpoint from `MODEL_PATH = models/material_cnn_2.pth` (see the model version-skew
note below). Also carries its own `argparse` CLI that runs unconditionally at import time (no
`__main__` guard) — this is why `tinius_gui.py` has to monkey-patch `sys.argv` before importing it.
Paths (`images/`, `captures/`, `outputs/`, `models/`) are all relative to the current working
directory, not to the script's own location — run everything from the repo root.

### `create_dataset.py` — training data builder
Segments every image in `images/` the same way `outline.py` does, samples labeled 16×16 patches
(material label is currently inferred from filename prefix — `D638`/`D790` → plastic, else metal),
and writes the patch dataset to `dataset/patch_dataset_2.npz`. Has its own `argparse` CLI (`--test`,
default `D638.jpg`, or `all`) that also runs unconditionally at import (no `__main__` guard).

### `train_model.py` — classifier trainer
Trains `MaterialPatchNet` (a separate, hand-duplicated copy of the same architecture defined in
`outline.py` — the two must be kept in sync by hand) on `dataset/patch_dataset_2.npz`, with
augmentation (flips, brightness/contrast jitter, Gaussian noise), a `WeightedRandomSampler` for class
balance, Adam + weight decay + dropout, cosine-annealing LR, and early stopping. Writes the trained
checkpoint to `models/material_cnn_3.pth`.

> **Known bug — model version skew:** `train_model.py` currently writes `material_cnn_3.pth`, but
> `outline.py` still loads `material_cnn_2.pth`. The newest trained model is not the one the GUI
> actually uses. Not fixed as part of this pass — flagging it here since it's easy to miss.

### `server/server.py` — standalone dev HTTP API
A FastAPI service (`POST /measure`) that reimplements the same PCA-based measurement math as
`outline.py`, independently, as an HTTP endpoint returning JSON + base64-encoded PNGs. Support for
either a fixed `ppmm`, FOV-based, or ArUco-marker-based scale calibration. **Not used by the GUI**,
shares no code with `outline.py`, and has no material classification at all. Uses a hardcoded dev API
key (`API_KEY = "some-long-secret"`, flagged in its own comments as needing to move to an environment
variable) — treat as a local dev/test tool, not something to expose as-is.

## Archive (`archive/`)

Superseded or experimental scripts kept for reference. None of these are imported by the active
pipeline, and the active pipeline doesn't import them either — each is self-contained.

| File | What it was |
|---|---|
| `old_outline.py` | The predecessor to `outline.py` — same PCA measurement approach, no material classification. Executes a hardcoded batch of 5 images on import (no `__main__` guard). |
| `outlineandmeasure.py` | Another independent copy of the same PCA measurement pipeline as `old_outline.py`/`server.py`. Despite the name, it has no CLI flags — `docs/commands.md` used to (incorrectly) document this file; the CLI it describes actually belongs to `extract_outline.py`. |
| `extract_outline.py` | A more complete standalone measurement CLI with two calibration modes (`--use_fov`/FOV-based, or ArUco-marker-based) instead of a fixed pixel-per-mm constant. This is the file `docs/commands.md`'s example command actually runs. |
| `capture_and_outline1.py` | Early Raspberry Pi camera (`picamera2`) capture + segmentation experiment; superseded by the GUI's own OpenCV camera handling. |
| `grabcut_outline.py` | One-off experiment using OpenCV's `grabCut` instead of `rembg` for segmentation. Flat script, no functions, hardcoded `sample.jpg` input. |
| `dryrun_unet_mbv3.py` | Smoke test for a `segmentation_models_pytorch` U-Net (MobileNetV3 encoder) with **random, untrained weights** — just proves the tensor shapes work. Unrelated to the CNN actually used for classification. |
| `smp_model.py` | A 9-line snippet constructing an `smp.Unet` (ResNet34 encoder) — a "how do I build this" reference note, never wired to any pipeline. |
| `classify_materials.py` | A density/color heuristic material classifier + ASTM lookup, written before the CNN-based approach in `outline.py`. Not imported anywhere; also contains a `run(self, ...)` function that takes `self` outside of any class — looks like a fragment left over from a larger class that was never finished. |
| `logs/runs.json`, `logs/outputs_rf_temp.txt` | Large (15 MB / 2.3 MB) data/log dumps with no code references anywhere in the repo — likely leftovers from an external tool run (naming suggests Roboflow). Kept rather than deleted, but safe to ignore. |

## Data directories

- **`images/`** — reference specimen photos (by ASTM family: `D638*`, `D790*`, `E8*`, `E8_B557*`) used for dataset-building and manual pipeline testing.
- **`captures/`** — camera frames captured live by the GUI (`specimen_<timestamp>.jpg`).
- **`outputs/`** — outline/annotation images written by `outline.py` (and the archived variants) for each processed image.
- **`dataset/`** — `patch_dataset_2.npz` is the live training set (written by `create_dataset.py`, read by `train_model.py`). `patch_dataset.npz` is an older, orphaned v1 dataset with no current code reference.
- **`models/`** — `material_cnn_2.pth` is the one `outline.py` actually loads. `material_cnn.pth` (v1) and `material_cnn_3.pth` (newest, but currently unused — see the version-skew note above) are also present.

## Other folders

- **`samples/`** — loose test/reference images (`black_sample.jpg`, `metallic_sample.jpg`, etc.) not read by any current script; used historically for ad hoc testing of the archived measurement variants.
- **`assets/`** — `icon.jpeg`, unreferenced by any code; kept in case it's used as an app/window icon later.
- **`deploy/config.txt`** — a Raspberry Pi `/boot/firmware/config.txt` reference (camera/I2C/SPI overlay settings) for provisioning a new Pi, not read by any Python code in this repo.
- **`docs/`** — this file, `commands.md` (corrected command reference), and `images/` (figures used in the top-level README, pulled from the capstone final report).

## Known issues not fixed by this pass

- **Model version skew**: `outline.py` loads `material_cnn_2.pth`; `train_model.py` produces `material_cnn_3.pth`. Decide which is actually best and repoint `MODEL_PATH` in `outline.py` if `_3` should be live.
- **No `__main__` guards** on `outline.py`'s CLI section, `create_dataset.py`, `old_outline.py`, or `outlineandmeasure.py` — importing any of these runs pipeline code immediately. `tinius_gui.py` works around this for `outline.py` specifically with a `sys.argv` monkey-patch.
- **Triplicated measurement logic**: `old_outline.py`, `outlineandmeasure.py`, and `server/server.py` each independently reimplement the same PCA-based measurement math instead of sharing a module.
- **No `requirements.txt`** — dependencies observed across the codebase: `opencv-python`, `numpy`, `Pillow`, `rembg`, `torch`, `PyQt5`, `fastapi`, `uvicorn`, `segmentation_models_pytorch` (archive only), `picamera2` (archive only, Pi-specific).
