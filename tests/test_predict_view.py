import pytest
import os
from pyramid import testing
from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.lite.python.interpreter import Interpreter
from unittest.mock import patch, MagicMock

from skinvestigatorai.views.predict_view import get_latest_model, load_model_type, predict_view


# Mock setup for filesystem and models
@pytest.fixture
def filesystem_mock(mocker):
    mocker.patch('os.listdir', return_value=['skinvestigator-sm.tflite'])
    mocker.patch('os.path.getctime', side_effect=lambda x: {'skinvestigator-sm.tflite': 1}[os.path.basename(x)])
    mocker.patch('os.path.join', side_effect=lambda *args: os.sep.join(args))

@pytest.fixture
def model_mock(mocker):
    # Mock for H5 model
    h5_model = Sequential([Dense(2, activation='softmax')])
    h5_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Mock for TFLite model
    tflite_model = MagicMock(spec=Interpreter)
    tflite_model.get_input_details.return_value = [{'index': 1}]
    tflite_model.get_output_details.return_value = [{'index': 2}]

    mocker.patch('tensorflow.keras.models.load_model', return_value=h5_model)
    mocker.patch('tensorflow.lite.python.interpreter.Interpreter', return_value=tflite_model)


# Test model loading
@pytest.mark.usefixtures("filesystem_mock")
def test_load_model_type():
    h5_model = load_model_type('H5')
    assert isinstance(h5_model, Sequential), "Failed to load H5 model correctly"

    tflite_model = load_model_type('TFLITE')
    assert isinstance(tflite_model, Interpreter), "Failed to load TFLite model correctly"


# Test prediction functionality
@pytest.mark.usefixtures("model_mock")
def test_predict_view_success():
    request = testing.DummyRequest()
    request.method = 'POST'
    request.POST['image'] = MagicMock(file=MagicMock(spec=Image.Image))

    with patch('PIL.Image.open', return_value=Image.new('RGB', (128, 128), 'white')) as mock_img_open:
        response = predict_view(request)
        assert 'prediction' in response, "Prediction missing in response"
        assert 'confidence' in response, "Confidence missing in response"


def test_predict_view_failure():
    request = testing.DummyRequest()
    request.method = 'POST'
    # Simulating a bad request without an image
    with pytest.raises(Exception) as exc_info:
        predict_view(request)
    assert 'image' in str(exc_info.value), "Expected failure when image is missing"


# Additional tests can be added for other functionalities like dashboard view, feature extraction, etc.

if __name__ == "__main__":
    pytest.main()
