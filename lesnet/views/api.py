"""HTTP API: selective triage prediction (paper §7).

Wired to the new calibrated TriagePredictor (abstaining, referral-biased) instead of the
old top-5 classifier. The predictor is loaded lazily so the app boots without artifacts.
Point LESNET_TRIAGE_ARTIFACTS at a trained artifacts directory.
"""
import logging
import os

import numpy as np
import requests
from PIL import Image
from pyramid.httpexceptions import HTTPBadRequest
from pyramid.view import view_config

from lesnet.config.model import ModelConfig
from lesnet.ml.artifacts import BUNDLE_FILE, MODEL_FILE
from lesnet.data.records import LesionRecord
from lesnet.data.taxonomy import TRIAGE_CLASSES

log = logging.getLogger(__name__)
TRIAGE_ARTIFACTS_DIR = os.environ.get('LESNET_TRIAGE_ARTIFACTS', 'models/triage')
DEFAULT_MODEL_ID = os.environ.get('LESNET_TRIAGE_MODEL', 'M-4s')
# When set, serve the self-supervised JEPA encoder + probe head instead of the TF triage model
# (research demo). Avoids importing TensorFlow at all in a torch-only environment.
JEPA_ARTIFACTS_DIR = os.environ.get('LESNET_JEPA_ARTIFACTS')
# The live demo serves the JEPA family by default (medium). Set LESNET_JEPA_VARIANT=small for the
# smaller edge build, or LESNET_USE_TF=1 to fall back to the TensorFlow triage stack.
JEPA_VARIANT = os.environ.get('LESNET_JEPA_VARIANT', 'medium')
JEPA_HOME = os.environ.get('LESNET_JEPA_HOME', 'models/jepa')
USE_TF = os.environ.get('LESNET_USE_TF', '') == '1'

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


@view_config(route_name='predict', request_method='POST', renderer='json')
def predict_api(request):
    upload = request.POST.get('image')
    if upload is None or not hasattr(upload, 'file'):
        return HTTPBadRequest(detail="No image file was uploaded.")
    try:
        image = np.asarray(Image.open(upload.file).convert('RGB'))
    except Exception as error:  # noqa: BLE001 - user-supplied input
        log.warning("Unreadable upload: %s", error)
        return HTTPBadRequest(detail="Could not read the uploaded image.")

    try:
        predictor = _get_predictor()
    except Exception as error:  # noqa: BLE001 - fetch or model-load failure should degrade gracefully
        log.exception("Triage model unavailable: %s", error)
        request.response.status = 503
        return {'triage': 'abstain', 'valid_image': False, 'reason': 'model_unavailable',
                'message': 'Triage model is not available yet.'}

    return predictor.predict(image, _record_from_request(request))


@view_config(route_name='labels', request_method='GET', renderer='json')
def labels_api(request):
    try:
        bundle = _get_predictor().bundle
        return {'triage_classes': TRIAGE_CLASSES, 'fine': list(bundle['label_maps']['fine_vocabulary'])}
    except (FileNotFoundError, OSError):
        return {'triage_classes': TRIAGE_CLASSES, 'fine': []}
