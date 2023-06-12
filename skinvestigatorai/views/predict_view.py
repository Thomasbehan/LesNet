import numpy as np
from PIL import Image
from pyramid.view import view_config
from pyramid.httpexceptions import HTTPBadRequest
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from skinvestigatorai.core.ai.detector import SkinCancerDetector

# Load your trained model
model_path = 'models/best_model.h5'
custom_metrics = {
    'f1_score': SkinCancerDetector.f1_score,
    'specificity': SkinCancerDetector.specificity
}
model = load_model(model_path, custom_objects=custom_metrics)

# Define the class labels
class_labels = ['benign', 'malignant', 'unknown']


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

        # Make a prediction
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
