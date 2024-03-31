import os
import random
from unittest.mock import MagicMock, patch
from skinvestigatorai.views.predict_view import predict_view, dashboard_view
from pyramid import testing
import pytest
import glob


@pytest.fixture
def mock_is_image_similar():
    with patch(
            'skinvestigatorai.services.feature_extraction_service.FeatureExtractionService.is_image_similar',
            return_value=True
    ):
        yield


def find_random_jpg(data_directory):
    # Search recursively for JPG files within the specified directory
    jpg_files = glob.glob(os.path.join(data_directory, '**/*.jpg'), recursive=True)

    # Ensure there is at least one JPG file found
    if not jpg_files:
        raise FileNotFoundError("No JPG files found in the directory.")

    # Select a random file from the list
    random_file = random.choice(jpg_files)

    return random_file


def test_predict_view(mock_is_image_similar):
    request = testing.DummyRequest()

    data_directory = 'data'
    random_file = find_random_jpg(data_directory)

    with open(random_file, 'rb') as image_file:
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
