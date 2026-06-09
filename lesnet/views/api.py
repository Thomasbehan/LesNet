import logging

from pyramid.httpexceptions import HTTPBadRequest
from pyramid.view import view_config

from lesnet.services.inference import Inference
from lesnet.services.model import SVModel

log = logging.getLogger(__name__)

_inference_service = None


def _get_inference_service():
    """Lazily build the Inference service once, on first prediction.

    Avoids loading the model at import time (which would block app startup
    and break test collection when no model file is present).
    """
    global _inference_service
    if _inference_service is None:
        _inference_service = Inference()
    return _inference_service


@view_config(route_name='predict', request_method='POST', renderer='json')
def predict_api(request):
    upload = request.POST.get('image')
    if upload is None or not hasattr(upload, 'file'):
        return HTTPBadRequest(detail="No image file was uploaded.")

    return _get_inference_service().predict(upload)

@view_config(route_name='labels', request_method='GET', renderer='json')
def labels_api(request):
    model_service = SVModel()

    return model_service.load_labels()
