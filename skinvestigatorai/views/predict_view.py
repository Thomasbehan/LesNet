import logging
import os

from pyramid.view import view_config

from skinvestigatorai.services.inference import Inference

log = logging.getLogger(__name__)
model_dir = 'models/'

MODEL_TYPE = 'TFLITE'  # Set this to 'KERAS' or 'TFLite' as needed


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
