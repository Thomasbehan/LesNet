# CLAUDE.md — LesNet (branch: M-4)

> **Scope note.** LesNet is a **standalone open-source project** (`Thomasbehan/LesNet`). It is **not** part of the IONA platform — no Go, SvelteKit, Zitadel, Kong, `iona-ui`, or IONA workflow conventions apply. This file is the only authority for working in this repo.
>
> **Branch note.** This describes the **`M-4`** branch (the active development line, package version `4.0.2`), which differs substantially from `main`/`3.1.0`. On M-4 the Python package was renamed `skinvestigatorai` → **`lesnet`**, the model backbone is **ResNet152V2**, and the data pipeline was reworked into discrete scrape → extract → balance/split → train stages.

## What this is

LesNet (Python package **`lesnet`**, v4.0.2) is a deep-learning tool for **skin lesion classification** from dermoscopic images. One repo, three concerns:

1. **Data pipeline** — scrape the ISIC archive, sort images into class folders from `metadata.csv`, balance classes by augmentation, and split into `train/val/test`.
2. **Training** — transfer-learning on a **ResNet152V2** backbone (`imagenet` weights), 42 classes, 224×224 input; exports `.keras`.
3. **Web app** — a Pyramid + Jinja2 site that classifies an uploaded image and returns the **top-5** labels with probabilities (live demo: `lesnet.onrender.com`).

**Medical disclaimer (keep it intact).** LesNet is a research tool, **not** a diagnostic device. Never reword the README/UI disclaimer to imply clinical fitness.

## Tech stack (M-4 pins, from `setup.py`)

| Concern | Choice |
|---|---|
| Language | Python **3.9–3.11** (ruff targets `py311`) |
| Web framework | **Pyramid 2.0.2** + `pyramid_jinja2`, served by **waitress** via `pserve` — not Flask/FastAPI |
| ML | **TensorFlow 2.16.1 / Keras**, **ResNet152V2** backbone. `torch 2.5.1`/`torchvision 0.20.1` also pinned (no source import found — see audit) |
| Data wrangling | **pandas 2.2.3**, **scikit-learn 1.6.1** (`train_test_split`, class weights), **albumentations 2.0.0** |
| Model formats | `KERAS` (`.keras`) / `TFLITE` (`.tflite`), via `ModelConfig.MODEL_TYPE` (default `KERAS`) |
| Data source | **ISIC Archive API v2** (`api.isic-archive.com/api/v2/images`) |
| Packaging | `setup.py` (setuptools), editable install, entry point `paste.app_factory → lesnet:main` |
| Lint / tests | **ruff 0.9.1**, **pytest 8.3.4** + `pytest-cov`, `pytest-mock`, `WebTest` |
| License | **MPL 2.0** |

## Layout (M-4)

```
lesnet/                        # the package (renamed from skinvestigatorai)
  __init__.py                  # main(): Pyramid WSGI app factory
  routes.py                    # routes: home, supported-diagnoses, train(uuid), tensorboard, predict, labels, dashboard
  config/
    model.py                   # ModelConfig — hyperparameters, MODEL_URLS, TRAIN_DIR='data/training', CATEGORIES=42
    data.py                    # DataConfig — ISIC API URL, output dirs
  services/
    model.py                   # SVModel — build (ResNet152V2) / train / evaluate / quantize / save / load
    inference.py               # Inference — predict() returns top-5; + dead similarity/embedding code
    data.py                    # Data — dataset loaders (load_preprocessed_dataset, load_dataset), prediction preprocessing, augmentation
    data_scaper.py             # DataScraper — ISIC downloader (NOTE: filename misspelled "scaper")
    tuner.py                   # SVModelHPTuner (subclass of SVModel)
  models/downloader.py         # pull a released model by id from MODEL_URLS
  views/
    api.py                     # predict (POST /predict) + labels (GET /labels)  ⚠ instantiates Inference() at import
    default.py                 # home + supported-diagnoses pages
    notfound.py                # 404
  templates/*.jinja2           # layout, evaluate (home), supported-diagnoses, 404
  static/                      # CSS/JS/images + PWA service worker; multiple JS upload handlers
commands/                      # CLI scripts (run from repo root)
  run_data_scraper.py          # download ISIC images
  run_data_extraction.py       # ⚠ sorts images into class folders from metadata.csv — RUNS AT IMPORT (no __main__ guard)
  run_data_pre_prep.py         # balance classes + split into train/val/test
  detect_corrupted_images.py   # ⚠ deletes unreadable images — RUNS AT IMPORT (no __main__ guard)
  run_data_augmenter.py · run_experiment.py · run_model_quantize.py · run_train_model.py · download_model.py
tests/                         # pytest suite (⚠ test_inference_service.py is stale vs the new predict() contract)
models/                        # artifacts (.keras/.tflite gitignored) + label JSON (only LesNet_labels_old.json is tracked)
development.ini / production.ini / testing.ini   # Pyramid/waitress config (port 6543)
Dockerfile / docker-compose*.yml                 # ⚠ run `pserve development.ini` (debug toolbar) — see audit
*.sh                           # train.sh, download_dataset.sh, dataset_compressor.sh, open_tensorboard.sh
```

`/data/`, `/logs/`, and model weight files (`*.h5`, `*.tflite`, `*.keras`) are **gitignored** — never commit them.

## Data → training → inference flow

1. **Scrape:** `commands/run_data_scraper.py` pages the ISIC API and downloads images into `data/train/<diagnosis>/`.
2. **Extract/sort:** `commands/run_data_extraction.py` reads `data/training/metadata.csv` and moves `<isic_id>.jpg` into per-diagnosis folders (diagnosis-column priority logic).
3. **Balance + split:** `commands/run_data_pre_prep.py` augments minority classes up to the largest class, then splits into `data/training/{train,val,test}/<class>/`.
4. **Train:** `commands/run_train_model.py` → `SVModel.build_model()` (ResNet152V2, last `TRAINABLE_START=-10` layers unfrozen + custom Conv/Dense head) → `Data.load_preprocessed_dataset()` (reads the train/val/test dirs) → `train_model()` (TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger) → `save_model()` writes `<name>.keras` + `<name>_labels.json`.
5. **Infer:** `views/api.py` holds a module-level `Inference()`; `POST /predict` preprocesses the upload (resize 224×224, /255) and returns the **top-5** `{label, probability}` pairs as JSON. `GET /labels` returns the class list.

## Common commands

```bash
# dev setup (Python 3.9–3.11)
python -m pip install --upgrade pip setuptools
python -m pip install -e ".[testing]"

# run the web app (needs a model present — see gotcha below), http://localhost:6543
pserve development.ini --reload

# tests + lint
python -m pytest
python -m ruff check

# fetch a released model into models/
python commands/download_model.py -m M-4s

# data pipeline
python commands/run_data_scraper.py -p 2
python commands/run_data_extraction.py
python commands/run_data_pre_prep.py
python commands/run_train_model.py        # or: ./train.sh
```

## Conventions & gotchas (M-4)

- **The web app loads the model at *import time*.** `views/api.py:13` does `inference_service = Inference()` at module scope, which calls `load_model()`. `config.scan()` imports that module, so **the app will not start unless `models/LesNet.keras` and `models/LesNet_labels.json` exist** (only `LesNet_labels_old.json` is tracked). Download a model first.
- **Model id mismatch:** `MODEL_URLS` has key `M-4s` (→ `LesNet.M-4.keras`), but the README example says `-m M-4`. Use the key that exists. Also `load_model` expects the file named exactly `models/LesNet.keras`, while the downloader saves under the release filename — you may need to rename.
- **Match the existing style:** plain framework-light Python, classes in `services/`, config as class attributes in `config/model.py`, `print()` for pipeline progress. Hyperparameters live in `config/model.py` — change them there.
- **`data_scaper.py` is misspelled** ("scaper"); imports depend on the typo.
- **Two `commands/` scripts run on import** (`run_data_extraction.py`, `detect_corrupted_images.py`) and perform destructive file operations with no `if __name__ == '__main__'` guard. Don't import them casually; run them as scripts deliberately.
- **`commands/*` use bare `from run_data_scraper import main`** — relies on `commands/` being `sys.path[0]` (run from that dir / repo root).
- **Keep lines ≤120 chars** and pass `ruff check`. CI matrix is Python 3.9/3.10/3.11.
- **Tests mock the model** (no GPU/weights/data needed). NOTE: `test_inference_service.py` is currently **stale** against the rewritten `predict()` contract — see `docs/code-audit.md`.
- **Never commit datasets or weights** — large and gitignored.

## Git / contributing

- This branch: **`M-4`**. Repo under the **`Thomasbehan`** GitHub account (ensure `gh auth status` shows it active, not `thomasbehaniona`).
- Commit style: short subjects, often gitmoji + milestone/issue tag (e.g. `M-4 - :zap: Changing to ResNet152V2 base model`). Not strict Conventional Commits — match history.
- CI (`.github/workflows/test.yaml`) runs `ruff check` + `pytest` on PRs to `main`; CodeQL also runs; Dependabot manages bumps.
- See `docs/code-audit.md` for the current list of known flaws on this branch.
```
