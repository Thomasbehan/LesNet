import logging
import os

from pyramid.view import view_config

from lesnet.services.inference import Inference
from lesnet.services.model import SVModel

log = logging.getLogger(__name__)
model_dir = 'models/'

MODEL_TYPE = 'TFLITE'  # Set this to 'KERAS' or 'TFLite' as needed
inference_service = Inference()


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
def predict_api(request):
    image_file = request.POST['image']

    return inference_service.predict(image_file)

@view_config(route_name='labels', request_method='GET', renderer='json')
def labels_api(request):
    model_service = SVModel()

    return model_service.load_labels()
