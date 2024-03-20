import os
import random
from unittest.mock import MagicMock, patch
from skinvestigatorai.views.predict_view import predict_view, dashboard_view
from pyramid import testing
import pytest


@pytest.fixture
def mock_is_image_similar():
    with patch(
            'skinvestigatorai.services.feature_extraction_service.FeatureExtractionService.is_image_similar',
            return_value=True
    ):
        yield


def test_predict_view(mock_is_image_similar):
    request = testing.DummyRequest()

    benign_directory = 'data/train/benign'

    jpg_files = [f for f in os.listdir(benign_directory) if f.endswith('.JPG')]

    random_file = random.choice(jpg_files)
    file_path = os.path.join(benign_directory, random_file)

    with open(file_path, 'rb') as image_file:
        dummy_file_upload = MagicMock()
        dummy_file_upload.file = image_file
        request.POST['image'] = dummy_file_upload

        response = predict_view(request)
        assert 'prediction' in response
        assert 'confidence' in response
        assert response['prediction'] in ['benign', 'malignant']
        assert 0 <= response['confidence'] <= 100


def test_dashboard_view():
    request = testing.DummyRequest()
    request.session['prediction'] = 'benign'
    request.session['confidence'] = 0.8

    response = dashboard_view(request)

    assert 'prediction' in response
    assert 'confidence' in response
    assert response['prediction'] == 'benign'
    assert response['confidence'] == 0.8
