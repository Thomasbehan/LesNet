import io
from unittest.mock import MagicMock
from pyramid import testing
from PIL import Image
from skinvestigatorai.views.predict_view import predict_view, dashboard_view


def test_predict_view():
    request = testing.DummyRequest()
    img = Image.new('RGB', (150, 150))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    # Creating a dummy file upload object
    dummy_file_upload = MagicMock()
    dummy_file_upload.file = buf
    request.POST['image'] = dummy_file_upload

    response = predict_view(request)
    assert 'prediction' in response
    assert 'confidence' in response
    assert response['prediction'] in ['benign', 'malignant']
    assert 0 <= response['confidence'] <= 1


def test_dashboard_view():
    request = testing.DummyRequest()
    request.session['prediction'] = 'benign'
    request.session['confidence'] = 0.8

    response = dashboard_view(request)

    assert 'prediction' in response
    assert 'confidence' in response
    assert response['prediction'] == 'benign'
    assert response['confidence'] == 0.8
