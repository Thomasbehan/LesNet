"""HTTP API: selective triage prediction (paper §7).

Wired to the new calibrated TriagePredictor (abstaining, referral-biased) instead of the
old top-5 classifier. The predictor is loaded lazily so the app boots without artifacts.
Point LESNET_TRIAGE_ARTIFACTS at a trained artifacts directory.
"""
import logging
import os

import numpy as np
from PIL import Image
from pyramid.httpexceptions import HTTPBadRequest
from pyramid.view import view_config

from lesnet.ml.datasets import LesionRecord
from lesnet.ml.inference import TriagePredictor
from lesnet.ml.taxonomy import TRIAGE_CLASSES

log = logging.getLogger(__name__)
TRIAGE_ARTIFACTS_DIR = os.environ.get('LESNET_TRIAGE_ARTIFACTS', 'models/triage')

_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
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
    except (FileNotFoundError, OSError):
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
