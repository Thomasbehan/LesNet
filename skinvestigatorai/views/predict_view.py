import os
import json
import numpy as np
from PIL import Image
from pyramid.response import Response
from pyramid.view import view_config
from pyramid.httpexceptions import HTTPBadRequest
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.lite.python.interpreter import Interpreter
from skinvestigatorai.services.feature_extraction_service import FeatureExtractionService

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


def load_model_type(model_type):
    """
    Load the latest model based on the provided model type.
    """
    if model_type.upper() == 'H5':
        model_path = get_latest_model(model_dir, '.h5')
        model = load_model(model_path)
    elif model_type.upper() == 'TFLITE':
        model_path = get_latest_model(model_dir, '.tflite')
        model = Interpreter(model_path)
    else:
        raise ValueError(f"Unsupported model type {model_type}. Please choose 'H5' or 'TFLite'.")

    return model


model = load_model_type(MODEL_TYPE)
print('Model loaded. Start serving...')

# Define the class labels
class_labels = ['benign', 'malignant', 'unknown']

# Initialize the feature extraction service
feature_service = FeatureExtractionService(model, MODEL_TYPE)
feature_service.create_feature_extractor()


@view_config(route_name='predict', request_method='POST', renderer='json')
def predict_view(request):
    # Read the image file from the request
    image_file = request.POST['image'].file

    try:
        # Open and preprocess the image
        image = Image.open(image_file).convert('RGB')
        image = image.resize((128, 128))
        image_array = img_to_array(image)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        is_similar = feature_service.is_image_similar(image_array)
        if not is_similar:
            error_message = """
                            Please make sure the image is of a skin lesion, is clear, focused, and occupies most of the 
                            frame while leaving sufficient space around the edges.
                            """
            return Response(status=400, body=json.dumps({"error": error_message.strip()}),
                            content_type='application/json')

        # Make a prediction
        if isinstance(model, Interpreter):  # If the model is a TFLite Interpreter
            model.allocate_tensors()
            model.invoke()
            output_details = model.get_output_details()
            predictions = model.get_tensor(output_details[0]['index'])
        else:  # If the model is a full Keras model
            predictions = model.predict(image_array)

        predicted_class = class_labels[np.argmax(predictions)]

        # Return the prediction result
        return {
            'prediction': predicted_class,
            'confidence': float(predictions[0][np.argmax(predictions)]) * 100
        }
    except Exception as e:
        return HTTPBadRequest(reason=str(e))


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
