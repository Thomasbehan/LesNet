# LesNet — Code Audit Report (branch `M-4`)

**Date:** 2026-06-09
**Branch:** `M-4` (package `lesnet`, version 4.0.2)
**Scope:** Full repository scan — `lesnet/` package, `commands/`, tests, frontend, and configuration.
**Purpose:** Triage list of flaws (security, correctness, algorithm/complexity, SOLID/DRY, standards) on the M-4 line. Not a set of applied fixes.

Severity: **BLOCKER** (security / breaks in normal use) · **MAJOR** (wrong behaviour, dead features, serious design debt) · **MINOR** (quality/cleanup).

---

## Resolution status — 2026-06-09

A remediation pass landed on this branch (local working tree). Verified afterwards: `ruff check` clean, **56/56 pytest pass**, and the app boots and serves `/`, `/labels`, `/supported-diagnoses` (now even with no model present), with the predict error path returning a generic message instead of leaking internals.

**Fixed:** 1.1 (Dockerfile serves `production.ini`), 1.2 (lazy model load — app boots without a model), 1.3 (in-memory decode, no `/tmp` leak), 1.4 (upload validation + generic errors + size cap), 2.1 (`class_weight` passed to `fit`), 2.2 (split before augment), 2.3 (`__main__` guards on the destructive scripts), 2.4 (bounded random search), 2.5 (`L2_LAYER_3` typo), 2.6 (`get_latest_model`), 2.7 (`raise` instead of `exit(1)`), 2.8 (augment uses `img_size`), 2.9 (low-confidence flag), 3.2/3.3 (removed dead `get_latest_model` + `MODEL_TYPE` in `api.py`), 3.4 (removed duplicated scraper CLI), the stale/tautological tests (§5), plus MINOR: `data_scaper`→`data_scraper` rename, `urllib3` import, `pd.notna`, `save_model` splitext, `M-4` URL alias, removed `pyserve`, `print`→`logging` in `model.py`, dropped the `..` path hardcode.

**Deferred (design-debt, not defects — need a trainable model / browser pass to verify safely):** 3.1 (split the `SVModel` god class), 3.2 (full KERAS/TFLITE strategy pattern + remove the dead embedding/similarity subsystem), 3.5 (consolidate the duplicate front-end upload handlers), full `print`→`logging` sweep across `commands/`, and removing the unused `torch`/`torchvision` pins (left as a likely-intentional future dependency).

**Out of scope (not a code fix):** the M-4 model/labels are not published — the `M-4s` release URL 404s — so real predictions can't be served until the weights are released.

---

> Several issues carry over unchanged from `main`/3.1.0; these are marked **[carried]**. Issues new to or changed by the M-4 rework are marked **[M-4]**.

---

## 1. BLOCKER — security & correctness

### 1.1 Production container runs the development configuration **[carried]**
- **Where:** `Dockerfile`, `docker-compose.yml` — `CMD ["pserve", "development.ini", "--reload"]`.
- **Problem:** `development.ini` enables `pyramid_debugtoolbar` (interactive in-browser debugger that exposes internals and can execute code) plus auto-reload. A correct `production.ini` exists but is never used. The live demo ships the debug toolbar.

### 1.2 The web app loads the model at *import time* — and won't boot without it **[M-4, new]**
- **Where:** `lesnet/views/api.py:13` — `inference_service = Inference()` at module scope; `Inference.__init__` → `load_model()`.
- **Problem:** `config.scan()` imports `api.py`, so simply starting the app (or importing the module in a test) deserializes a model from disk. If `models/LesNet.keras` / `models/LesNet_labels.json` are absent (only `LesNet_labels_old.json` is tracked), **import fails and the app cannot start**. This is the M-4 "fix" for the old per-request reload, but it trades one problem for an import-time side effect. The model should be created once during `main()`/app config and stored on the registry, not at module import.

### 1.3 Uploaded temp files are written to `/tmp` and never deleted **[M-4, new]**
- **Where:** `lesnet/services/data.py:96-107` (`load_image_for_prediction`).
- **Problem:** every prediction writes `/tmp/uploaded_image_<ts>_<rand>.jpg` and never removes it — unbounded disk growth on a long-running server. Use an in-memory decode (`tf.io.decode_image` on the bytes) or delete the temp file in a `finally`.

### 1.4 Unvalidated upload + internal-detail leak **[carried]**
- **Where:** `lesnet/views/api.py:30` (`request.POST['image']`), `services/data.py:98` (`uploaded_file.file.read()`), `services/inference.py:99-101`.
- **Problems:** missing `image` field → unhandled `KeyError` (500); no content-type/size limit (memory DoS, decompression bomb); on exception `inference.predict` returns `str(e)` straight to the client (`HTTPBadRequest(detail=str(e))`).

---

## 2. MAJOR — logic & algorithm

### 2.1 Class weights are computed but never used **[M-4, regression]**
- **Where:** `lesnet/services/model.py:316-319` computes `class_weights`, but `model.fit(...)` at `349-354` **omits the `class_weight=` argument** (it was present on `main`).
- **Problem:** the expensive full-dataset label scan still runs, but the result is discarded — so class imbalance is no longer compensated during training, despite the pipeline going to lengths to balance/weight. Either pass `class_weight=class_weights` or remove the dead computation.

### 2.2 Augment-before-split causes train/test data leakage **[M-4, new]**
- **Where:** `commands/run_data_pre_prep.py` — `balance_classes()` augments images in place, then `split_dataset()` splits the (now augmented) folder into train/val/test.
- **Problem:** augmented copies of the same source image can land in *different* splits, so the test set contains near-duplicates of training images. This inflates reported accuracy/precision/recall and undermines the headline metrics. Split first, then augment **only the training split**.

### 2.3 Two `commands/` scripts execute destructive work at import **[M-4, new]**
- **Where:** `commands/run_data_extraction.py` (moves files based on `metadata.csv`) and `commands/detect_corrupted_images.py` (`os.remove` on unreadable images) run all logic at module top level with **no `if __name__ == '__main__'` guard**.
- **Problem:** importing either module (tooling, test collection, accidental import) immediately moves/deletes files. Wrap in a `main()` + guard.

### 2.4 Hyperparameter search is a combinatorial explosion **[carried]**
- **Where:** `model.py:run_experiments` (210-270).
- **Problem:** full Cartesian product of ~10 hyperparameters, each trained 15 epochs — can never complete. `keras-tuner` is a declared dependency but unused here.

### 2.5 Silent typo disables layer-3 L2 during tuning **[carried]**
- **Where:** `model.py:260` — `ModelConfig.L3_LAYER_3 = hparams_dict['l2_layer_3']` (should be `L2_LAYER_3`). The sampled value is written to a non-existent attribute and dropped. Still present on M-4.

### 2.6 `get_latest_model()` default can never match **[carried]**
- **Where:** `model.py:186-192` — filters `endswith(extension)` with `extension="KERAS"`; real files end `.keras`/`.tflite`, so `max()` raises on an empty list.

### 2.7 `exit(1)` inside a service method **[carried]**
- **Where:** `model.py:366` (`load_labels`) calls `exit(1)` when labels are missing — kills the WSGI worker. Must raise.

### 2.8 Fragile on-the-fly augmentation **[carried]**
- **Where:** `data.py:augment_image` (46-64) uses static `image.shape[0]` inside a `tf.data.map` pipeline where the shape is often `None` → `int(None)` failures.

### 2.9 Out-of-distribution guard removed **[M-4, behavioural regression]**
- **Where:** `services/inference.py:67-98`.
- **Problem:** on `main`, predictions below a confidence threshold returned a friendly "not sure about this one" rejection. M-4 always returns the top-5 classes regardless of confidence, so non-lesion / garbage uploads get a confident-looking label. For a medical-adjacent tool this is a meaningful UX/safety regression. (The dead similarity/embedding code in 3.2 was the intended OOD mechanism and is still unused.)

---

## 3. MAJOR — SOLID & DRY

### 3.1 `SVModel` is a god class **[carried]**
- **Where:** `model.py` (~385 lines): builds, trains, evaluates, quantizes, saves, loads, computes embeddings, runs experiments, logs metrics. Split into focused units.

### 3.2 Duplicated functions and dead branches **[carried]**
- `get_latest_model` duplicated in `model.py:186` and `views/api.py:16` (the latter unused).
- `calculate_dataset_embedding` near-identical in `SVModel` and `Inference`; `is_image_similar`/embedding path is never reached (dead OOD feature).
- KERAS-vs-TFLITE `if/elif` branching copy-pasted across `create_feature_extractor`, `calculate_dataset_embedding` (×2), `_predict_similar`, `load_model` — should be a strategy/adapter (OCP/DIP).

### 3.3 `MODEL_TYPE` has multiple sources of truth + global mutable config **[carried]**
- `ModelConfig.MODEL_TYPE='KERAS'`, a dead module-level `MODEL_TYPE='TFLITE'` at `views/api.py:12`, and per-instance copies. `ModelConfig` class attributes are mutated at runtime (`ModelConfig.BATCH_SIZE = …` in `run_experiments`) — untestable, race-prone.

### 3.4 Data-scraper CLI duplicated **[carried]**
- `commands/run_data_scraper.py` duplicates the `__main__` block inside `services/data_scaper.py`; the service module also carries its own argparse entry point (mixing CLI into the service layer).

### 3.5 Two/three competing front-end upload paths **[carried/M-4]**
- `templates/evaluate.jinja2`, `static/js/app.js`, and the new `static/js/file-upload.js` / `explicit-image-handler.js` overlap in handling the upload/predict flow — duplicated logic, easy to drift. Consolidate to one handler.

---

## 4. MINOR — standards, quality, dependencies

- **`print()` instead of `logging`** across `model.py`, `models/downloader.py`, `services/data_scaper.py` (inconsistent with `inference.py`).
- **Misspelled module** `services/data_scaper.py` ("scaper"); imports depend on the typo.
- **README ↔ code drift:** README's downloader example is `-m M-4`, but `MODEL_URLS` only defines `M-4s`. Also `load_model` hardcodes `models/LesNet.keras` while the downloader saves under the release filename (`LesNet.M-4.keras`) — a rename is required for the app to find it.
- **Dependency cruft in `setup.py`:** `pyserve==0.2.8` appears unused (app uses `pserve`+`waitress`); `torch==2.5.1`/`torchvision==0.20.1` pinned but no import found in source — heavy, possibly removable.
- **`run_data_pre_prep.py` hardcodes a `..` path** (`os.path.join("..", ModelConfig.TRAIN_DIR)`), so it only works when run from a subdirectory; and it passes the same dir as source *and* destination to `split_dataset`, risking re-processing of already-split folders on re-run.
- **Unreliable pandas NA checks:** `run_data_extraction.py` uses `is not pd.NA` after `.replace('None', pd.NA)`; prefer `pd.isna(...)`. The diagnosis-column override order (general `diagnosis` overrides finer `diagnosis_3`) is suspicious and worth confirming.
- **Legacy import:** `data_scaper.py` uses the deprecated `requests.packages.urllib3…` shim.
- **`save_model` label-filename derivation** (`model.py:302`) uses `filename.replace('.keras', …)`; wrong for non-`.keras` filenames.
- **Unsanitized paths from API/CSV data:** `data_scaper.py` / `run_data_extraction.py` build paths from `isic_id`/`diagnosis` fields without sanitization (low risk; trusted source).

---

## 5. Tests

- **Stale suite — failing on M-4:** `tests/test_inference_service.py:35-55` asserts `predict()` returns a dict with `prediction`/`confidence` keys and a 400 `Response` on low confidence. M-4's `predict()` returns `{'predictions': [{label, probability}, …]}` (top-5) and never returns a 400. Additionally it feeds a `BytesIO` where the new code calls `uploaded_file.file.read()` (no `.file` attribute). These tests cannot pass against the current code.
- **Tautological test** `tests/test_model_service.py` mocks a method then asserts the mock was called.
- **Coverage gaps:** nothing exercises the import-time model load (1.2), the dropped `class_weight` (2.1), the data-leakage split (2.2), the `L3_LAYER_3` typo (2.5), or `get_latest_model` (2.6).

---

## 6. Suggested triage order

1. **1.1** — serve `production.ini` in the container.
2. **1.2** — move the `Inference()` construction out of import scope into app startup.
3. **2.1** — pass (or remove) the computed `class_weight`.
4. **2.2** — split before augmenting to remove train/test leakage.
5. **2.3** — add `__main__` guards to the destructive `commands/` scripts.
6. **5** — update the stale inference tests to the new `predict()` contract.
7. **1.3 / 1.4 / 2.9** — temp-file cleanup, upload validation, restore an out-of-distribution guard.
8. Then the carried SOLID/DRY items (section 3) and MINOR cleanups (section 4), each with a test that would have caught the bug.
