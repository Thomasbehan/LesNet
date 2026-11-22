"""HTTP API: selective triage prediction (paper §7).

Wired to the new calibrated TriagePredictor (abstaining, referral-biased) instead of the
old top-5 classifier. The predictor is loaded lazily so the app boots without artifacts.
Point LESNET_TRIAGE_ARTIFACTS at a trained artifacts directory.
"""
import logging
import os

import numpy as np
import requests
from PIL import Image, ImageOps
from pyramid.view import view_config

from lesnet.config.model import ModelConfig
from lesnet.ml.artifacts import BUNDLE_FILE, MODEL_FILE
from lesnet.data.records import LesionRecord
from lesnet.data.taxonomy import TRIAGE_CLASSES, triage_index

log = logging.getLogger(__name__)
TRIAGE_ARTIFACTS_DIR = os.environ.get('LESNET_TRIAGE_ARTIFACTS', 'models/triage')
DEFAULT_MODEL_ID = os.environ.get('LESNET_TRIAGE_MODEL', 'M-4s')
# When set, serve the self-supervised JEPA encoder + probe head instead of the TF triage model
# (research demo). Avoids importing TensorFlow at all in a torch-only environment.
JEPA_ARTIFACTS_DIR = os.environ.get('LESNET_JEPA_ARTIFACTS')
# The live demo serves the JEPA family's small variant by default: it scores level with medium on the
# held-out test split (docs/backbone-bakeoff.md) at ~165 MB RSS instead of ~435 MB, which matters on a
# 512 MB instance. LESNET_JEPA_VARIANT=medium|large to override, LESNET_USE_TF=1 for the TF stack.
JEPA_VARIANT = os.environ.get('LESNET_JEPA_VARIANT', 'small')
JEPA_HOME = os.environ.get('LESNET_JEPA_HOME', 'models/jepa')
USE_TF = os.environ.get('LESNET_USE_TF', '') == '1'
# Uploads are shrunk to this long side at decode time — the size of the ISIC thumbnails the model
# was trained on — and anything over UPLOAD_MAX_PIXELS is refused outright (see load_upload).
UPLOAD_MAX_SIDE = 256
UPLOAD_MAX_PIXELS = 100_000_000

_predictor = None


def _ensure_triage_model(directory):
    """Download the released triage model into `directory` if it isn't already there.

    Makes the deployed app self-healing — it fetches the model on first use even if the
    container image didn't bake it in.
    """
    model_path = os.path.join(directory, MODEL_FILE)
    bundle_path = os.path.join(directory, BUNDLE_FILE)
    if os.path.exists(model_path) and os.path.exists(bundle_path):
        return
    model_url = ModelConfig.MODEL_URLS.get(DEFAULT_MODEL_ID)
    if not model_url:
        return
    os.makedirs(directory, exist_ok=True)
    for url, destination in [(model_url, model_path),
                             (model_url.replace('.keras', '.artifacts.json'), bundle_path)]:
        if os.path.exists(destination):
            continue
        log.info("Fetching triage model asset: %s", url)
        response = requests.get(url, timeout=600)
        response.raise_for_status()
        with open(destination, 'wb') as handle:
            handle.write(response.content)


def _ensure_jepa_model(variant, home):
    """Download + extract the released JEPA variant if absent, so the demo self-heals."""
    import tarfile
    import tempfile

    target = os.path.join(home, variant)
    if os.path.exists(os.path.join(target, 'jepa_config.json')):
        return target
    url = ModelConfig.JEPA_URLS.get(variant)
    if not url:
        raise FileNotFoundError(f'no released JEPA variant {variant!r}')
    os.makedirs(home, exist_ok=True)
    log.info("Fetching JEPA %s from %s", variant, url)
    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as handle:
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()
        for chunk in response.iter_content(1 << 20):
            handle.write(chunk)
        archive = handle.name
    with tarfile.open(archive) as tar:
        tar.extractall(home)
    os.unlink(archive)
    return target


def _get_predictor():
    global _predictor
    if _predictor is None:
        if JEPA_ARTIFACTS_DIR:  # explicit override: point at any artifacts directory
            from lesnet.jepa.serve import JEPADemoPredictor
            _predictor = JEPADemoPredictor(JEPA_ARTIFACTS_DIR)
        elif not USE_TF:        # default: the released JEPA family, fetched on first use
            from lesnet.jepa.serve import JEPADemoPredictor
            _predictor = JEPADemoPredictor(_ensure_jepa_model(JEPA_VARIANT, JEPA_HOME))
        else:
            from lesnet.ml.inference import TriagePredictor  # lazy: imports TensorFlow
            _ensure_triage_model(TRIAGE_ARTIFACTS_DIR)
            _predictor = TriagePredictor(TRIAGE_ARTIFACTS_DIR)
    return _predictor


class UploadTooLarge(ValueError):
    """The upload's pixel count is beyond anything a camera produces (likely a decompression bomb)."""


def load_upload(stream, max_side=UPLOAD_MAX_SIDE, max_pixels=UPLOAD_MAX_PIXELS):
    """Decode an upload to a small RGB array WITHOUT ever materialising it at full resolution.

    A 12 MP phone photo decoded in full and then colour-corrected in float64 needs >1 GB, which
    OOM-killed the 512 MB demo instance on every real-world upload. The model only ever sees a
    256 px ISIC-thumbnail-sized image, so shrinking first loses nothing: JPEG `draft` downscales in
    the DCT domain while decoding, EXIF rotation is honoured (phones store portrait sideways), and
    `thumbnail` brings the long side to `max_side`.
    """
    image = Image.open(stream)
    width, height = image.size
    if width * height > max_pixels:
        raise UploadTooLarge(f"Image is {width}x{height}; the limit is {max_pixels // 1_000_000} megapixels.")
    image.draft('RGB', (max_side, max_side))
    image = ImageOps.exif_transpose(image).convert('RGB')
    image.thumbnail((max_side, max_side), Image.LANCZOS)
    return np.asarray(image)


def _optional_float(request, name):
    try:
        return float(request.POST.get(name))
    except (TypeError, ValueError):
        return None


def _record_from_request(request):
    fitzpatrick = _optional_float(request, 'fitzpatrick')
    return LesionRecord(
        image_path='upload', source_dataset='query', raw_label='unknown', group_id='query',
        fitzpatrick=int(fitzpatrick) if fitzpatrick else None,
        anatomical_site=request.POST.get('site'),
        age=_optional_float(request, 'age'),
        sex=request.POST.get('sex'),
    )


def _abstain(request, status, reason, message):
    """Every /predict failure is JSON with a machine-readable `reason` — never an HTML error page —
    so the UI can always show a human sentence instead of a parser exception."""
    request.response.status = status
    return {'triage': 'abstain', 'valid_image': False, 'reason': reason, 'message': message}


@view_config(route_name='predict', request_method='POST', renderer='json')
def predict_api(request):
    upload = request.POST.get('image')
    if upload is None or not hasattr(upload, 'file'):
        return _abstain(request, 400, 'no_image', 'No image file was uploaded.')
    try:
        image = load_upload(upload.file, max_side=384 if USE_TF else UPLOAD_MAX_SIDE)  # TF stack: 384 px
    except UploadTooLarge as error:
        return _abstain(request, 413, 'too_large', str(error))
    except Exception as error:  # noqa: BLE001 - user-supplied input
        log.warning("Unreadable upload: %s", error)
        return _abstain(request, 400, 'unreadable', 'Could not read the uploaded image.')

    try:
        predictor = _get_predictor()
    except Exception as error:  # noqa: BLE001 - fetch or model-load failure should degrade gracefully
        log.exception("Triage model unavailable: %s", error)
        return _abstain(request, 503, 'model_unavailable', 'Triage model is not available yet.')

    try:
        return predictor.predict(image, _record_from_request(request))
    except Exception as error:  # noqa: BLE001 - one bad image must not surface as a stack trace
        log.exception("Prediction failed: %s", error)
        return _abstain(request, 500, 'internal', 'The image could not be analysed.')


def _fine_labels(predictor):
    """Diagnosis vocabulary of whichever predictor is loaded (JEPA demo head or TF bundle)."""
    bundle = getattr(predictor, 'bundle', None)
    if bundle is not None:
        return list(bundle['label_maps']['fine_vocabulary'])
    return sorted(str(c) for c in getattr(predictor, 'classes', []))


def _bucket_of(predictor, label):
    """'benign' | 'not_sure' | 'malignant' for a diagnosis, from the served head when it knows."""
    known = getattr(predictor, 'bucket_of', None) or {}
    if label in known:
        return known[label]
    index = triage_index(label)
    if index is None:
        return 'not_sure'
    return {'suspicious': 'not_sure'}.get(TRIAGE_CLASSES[index], TRIAGE_CLASSES[index])


@view_config(route_name='labels', request_method='GET', renderer='json')
def labels_api(request):
    try:
        predictor = _get_predictor()
        fine = _fine_labels(predictor)
        return {'triage_classes': TRIAGE_CLASSES, 'fine': fine,
                'buckets': {label: _bucket_of(predictor, label) for label in fine}}
    except Exception as error:  # noqa: BLE001 - the labels page should degrade, not 500
        log.warning("Labels unavailable: %s", error)
        return {'triage_classes': TRIAGE_CLASSES, 'fine': [], 'buckets': {}}
