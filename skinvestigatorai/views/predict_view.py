from pyramid.view import view_config

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
