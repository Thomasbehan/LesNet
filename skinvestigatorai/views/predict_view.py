import os
import json
import numpy as np
from PIL import Image
from pyramid.response import Response
from pyramid.view import view_config
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.lite.python.interpreter import Interpreter
from skinvestigatorai.services.feature_extraction_service import FeatureExtractionService
import logging

log = logging.getLogger(__name__)
model_dir = 'models/'

MODEL_TYPE = 'TFLITE'  # Set this to 'H5' or 'TFLite' as needed


def get_latest_model(model_dir, extension):
    """
    Returns the path of the latest model file in the specified directory with the specified extension.
    """
    list_of_files = [os.path.join(model_dir, basename) for basename in os.listdir(model_dir) if
                     basename.endswith(extension)]
    latest_model = max(list_of_files, key=os.path.getctime)
    print("LATEST MODEL:")
    print(latest_model)
    return latest_model

from skinvestigatorai.services.inference import Inference

@view_config(route_name='predict', request_method='POST', renderer='json')
def predict_view(request):
    image_file = request.POST['image'].file
    inference_service = Inference()

    return inference_service.predict(image_file)


@view_config(route_name='dashboard', renderer='skinvestigatorai:templates/dashboard.jinja2')
def dashboard_view(request):
    if 'prediction' in request.session:
        prediction = request.session['prediction']
        confidence = request.session['confidence']
        del request.session['prediction']
        del request.session['confidence']
    else:
        prediction = None
        confidence = None

    return {
        'prediction': prediction,
        'confidence': confidence
    }
