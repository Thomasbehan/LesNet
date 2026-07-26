# CLAUDE.md — LesNet

> **Scope note.** LesNet is a **standalone open-source project** (`Thomasbehan/LesNet`). It is **not** part of the IONA platform — no Go, SvelteKit, Zitadel, Kong, `iona-ui`, or IONA workflow conventions apply. This file is the only authority for working in this repo.
>
> **Git identity.** Commit as `thomasbehan4@gmail.com` (repo-local), never the IONA email. Ensure `gh auth status` shows `Thomasbehan` active.

## What this is

LesNet (Python package **`lesnet`**, v4.2.0) is a deep-learning tool for **skin lesion triage** from dermoscopic images. One repo, three concerns:

1. **Data pipeline** — download the ISIC archive and build a leakage-free, patient/lesion-grouped manifest CSV.
2. **Training** — a multi-task EfficientNetV2-S model (image + patient metadata) producing a 3-way **triage** head (benign / suspicious / malignant) plus an auxiliary fine-grained diagnosis head; temperature-calibrated, OOD-gated, conformal-wrapped, sensitivity-first operating point.
3. **Web app** — a Pyramid + Jinja2 site that runs the calibrated, **abstaining** triage predictor on an uploaded image (live demo: `lesnet.onrender.com`).

**Medical disclaimer (keep it intact).** LesNet is a research/triage tool, **not** a diagnostic device. Never reword the README/UI disclaimer to imply clinical fitness.

## Tech stack (pins in `setup.py`)

| Concern | Choice |
|---|---|
| Language | Python **3.11–3.12** |
| Web framework | **Pyramid 2.1** + `pyramid_jinja2`, served by **waitress** via `pserve` |
| ML | **TensorFlow 2.21 / Keras 3.14**, **EfficientNetV2-S** backbone |
| Numerics / data | **numpy 2.4**, **scipy 1.17**, **scikit-learn 1.9**, **Pillow 12**, **opencv-python-headless** (DullRazor hair removal) |
| Model format | `.keras` + a JSON artifact bundle (calibration / conformal / OOD / thresholds) |
| Data source | **ISIC Archive API v2** + optional PAD-UFES-20 |
| Packaging | `setup.py` (setuptools), editable install, entry point `paste.app_factory → lesnet:main` |
| Lint / tests | **ruff**, **pytest** (+ `pytest-cov`, `pytest-mock`, `WebTest`) |
| License | **MPL 2.0** |

## Layout

```
lesnet/                        # the package
  __init__.py                  # main(): Pyramid WSGI app factory (pyramid imported lazily)
  routes.py                    # routes: home, supported-diagnoses, predict, labels
  config/model.py              # ModelConfig — released-model registry (MODEL_URLS) for the web app + downloader
  data/                        # data subsystem (stage 1: source -> sort -> balance -> manifest; no TF)
    config.py                  # SourcingConfig
    sources/                   # isic, pad_ufes, fitzpatrick17k, ddi (+ registry) -> LesionRecords
    taxonomy.py · canonical.py # diagnosis -> triage bucket; curated diagnosis-name canonicalisation
    quality.py · sort.py · balance.py    # quality + pHash dedup; bucket/diagnosis folders; fair 1:1 balance
    records.py · splits.py · manifest.py # LesionRecord + manifest IO; grouped leakage-free splits
  ml/                          # the triage MODEL system (training-coupled, imports TF)
    config.py                  # PipelineConfig — all training/inference hyperparameters
    data_loader.py             # tf.data pipeline from a manifest
    preprocessing.py           # DullRazor hair removal + Shades-of-Gray colour constancy + resize/scale
    features.py                # metadata encoding (age/sex/site/Fitzpatrick)
    model.py                   # build_triage_model (EfficientNetV2-S) + triage/fine/feature sub-models
    losses.py · metrics.py     # focal loss + class weights; clinical metrics (sens/spec/ROC/ECE/AURC)
    training.py                # orchestrator: fit (optionally metric-gated) -> calibrate -> conformal -> OOD -> save
    calibration.py · conformal.py · ood.py   # temperature scaling, split-conformal, Mahalanobis OOD + quality gate
    evaluation.py              # build_report + fairness gate + model card
    inference.py               # TriagePredictor — quality/OOD gate -> calibrated triage -> abstention/conformal
    synthetic.py               # synthetic records for the CPU smoke path
    artifacts.py               # model + bundle save/load
  models/downloader.py         # pull a released model by id from ModelConfig.MODEL_URLS
  views/{api,default,notfound}.py   # /predict + /labels (lazy TriagePredictor); home + supported-diagnoses; 404
  templates/*.jinja2 · static/      # web UI
commands/                      # CLI scripts (run from repo root)
  build_dataset.py                               # stage 1: source -> canonicalise -> quality -> balance -> sort -> manifest
  run_train_triage.py                            # train (--smoke for a CPU synthetic end-to-end run)
  run_evaluate.py                                # evaluation report + model card
  run_triage_inference.py · run_validation_check.py   # inference (single image / validation set)
  download_model.py · package_model.py           # fetch / package release assets
tests/                         # pytest suite (CPU-only via conftest.py)
scripts/train_full_m4.sh       # full GPU pipeline (download -> manifest -> train -> package -> evaluate), nix-shell
development.ini / production.ini / testing.ini   # Pyramid/waitress config (port 6543)
Dockerfile / docker-compose.yml                  # container (production.ini); compose serves development.ini
```

`/data/`, `/logs/`, `/artifacts/`, and model weight files (`*.keras`, `*.tflite`, `models/triage*/`) are **gitignored** — never commit them.

## Data → training → inference flow

1. **Source + curate (stage 1):** `commands/build_dataset.py` (→ `lesnet.data.pipeline`) downloads (ISIC malignant-maximal; PAD-UFES auto; Fitzpatrick17k/DDI on disk), canonicalises diagnosis names, quality-gates + pHash-dedupes, sorts into `benign/not_sure/malignant/<diagnosis>/`, balances ~1:1 benign:malignant (group-safe, fairness-aware), and writes a leakage-free `manifest.csv` + `report.json`.
3. **Train:** `commands/run_train_triage.py --manifest …` → `lesnet.ml.training.train` (EfficientNetV2-S, focal loss, optional metric-gated rounds) → calibrate (temperature) → choose sensitivity-first operating point → split-conformal → fit Mahalanobis OOD → save `triage_model.keras` + `artifacts.json`.
4. **Evaluate:** `commands/run_evaluate.py` writes `evaluation_report.json` + `model_card.md` (incl. fairness gate).
5. **Infer:** web `POST /predict` (lazy `TriagePredictor`) or `commands/run_triage_inference.py`. Pipeline = quality gate → OOD gate → calibrated triage → abstention/conformal set + auxiliary fine-grained prediction. `GET /labels` returns the class list.

## Conventions & gotchas

- **The web app loads the model lazily**, on first `/predict` or `/labels` (`views/api.py:_get_predictor`). It boots without a model and self-heals by fetching the released **M-4s** model into `LESNET_TRIAGE_ARTIFACTS` (default `models/triage`) if absent; if the model is unavailable, `/predict` returns a graceful `503` abstain, not a crash.
- **No XLA/JIT.** `lesnet/ml/training.py` disables XLA (`jit_compile=False`, `set_jit(False)`, `TF_XLA_FLAGS`): the pip TensorFlow wheel's Triton-GEMM CPU path and GPU `libdevice` lookup are unavailable on many hosts, and the model trains fine on the standard executor. Don't re-enable it without a libdevice-complete CUDA environment.
- **Tests run on CPU.** `conftest.py` sets `CUDA_VISIBLE_DEVICES=-1` so the suite is deterministic and GPU-independent (CI has no GPU). Real GPU training runs outside the test suite via `scripts/train_full_m4.sh` inside a CUDA nix-shell.
- **Match the existing style:** plain framework-light Python, pure-logic modules in `lesnet/ml/`, hyperparameters as dataclass fields in `lesnet/ml/config.py` (`PipelineConfig`). Released-model URLs live in `config/model.py` (`ModelConfig.MODEL_URLS`).
- **Keep lines ≤120 chars** and pass `ruff check`. CI matrix is Python 3.11 / 3.12.
- **Never commit datasets, weights, or artifacts** — large and gitignored.

## Common commands

```bash
# dev setup (Python 3.11–3.12)
python -m pip install --upgrade pip setuptools
python -m pip install -e ".[testing]"

# run the web app, http://localhost:6543
pserve development.ini --reload

# tests + lint (lesnet/data is gated at 100% coverage)
python -m pytest
python -m pytest --cov=lesnet.data --cov-fail-under=100
ruff check

# stage 1: source + build a balanced, sorted, leakage-safe dataset
python commands/build_dataset.py --sources isic pad_ufes_20 fitzpatrick17k ddi --dest data/dataset

# CPU smoke (synthetic end-to-end: train -> calibrate -> save -> load -> infer)
python commands/run_train_triage.py --smoke --artifacts /tmp/lesnet_art

# single-image inference against a released/trained artifacts dir
python commands/run_triage_inference.py --artifacts models/triage --image sample_malignant.jpg --age 60 --sex male --site torso

# fetch a released model
python commands/download_model.py -m M-4s
```

## Git / contributing

- **Always branch off the latest `origin/main`** (`git fetch origin main && git checkout -B <branch> origin/main`) — never off a stale release/feature branch. `main` is the source of truth and may carry squash-merged PRs that raw branches lack.
- CI (`.github/workflows/test.yaml`) runs `ruff check` + `pytest` on PRs to `main`; CodeQL also runs; Dependabot manages bumps.
- Commit style: short subjects, often gitmoji + milestone/issue tag. Not strict Conventional Commits — match history.
- `docs/model-redesign.md` is the authoritative design for the triage system.
```
