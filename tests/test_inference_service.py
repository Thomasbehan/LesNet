from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
from pyramid.response import Response
import pytest
from PIL import Image

from skinvestigatorai.services.inference import Inference
from skinvestigatorai.services.model import SVModel


@pytest.fixture
def mock_svmodel():
    sv_model = MagicMock(SVModel)
    sv_model.load_model.return_value = (MagicMock(), ["class1", "class2"])
    sv_model.preprocess_image_for_tflite = lambda x: x
    return sv_model


@pytest.fixture
def inference(mock_svmodel):
    with patch('skinvestigatorai.services.model.SVModel', return_value=mock_svmodel):
        return Inference()


def create_mock_image():
    image = Image.new('RGB', (100, 100))
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = BytesIO(img_byte_arr.getvalue())
    return img_byte_arr


def test_predict_success(inference):
    mock_image = create_mock_image()

    inference.model.predict = MagicMock(return_value=np.array([[0.1, 0.9]]))

    result = inference.predict(mock_image)

    assert isinstance(result, dict)
    assert 'prediction' in result
    assert 'confidence' in result


def test_predict_failure(inference):
    mock_image = create_mock_image()

    inference.model.predict = MagicMock(return_value=np.array([[0.3, 0.2]]))

    result = inference.predict(mock_image)

    assert isinstance(result, Response)
    assert result.status_code == 400


def test_is_image_similar(inference):
    mock_image = np.random.rand(100, 100, 3)

    inference.dataset_embedding = np.random.rand(2048)
    inference._predict_similar = MagicMock(return_value=np.random.rand(2048))

    result = inference.is_image_similar(mock_image, threshold=0.5)

    assert result in [True, False]


def test__predict_similar_keras(inference):
    mock_image = np.random.rand(100, 100, 3)
    inference.model.predict = MagicMock(return_value=np.random.rand(1, 2048))

    with patch('skinvestigatorai.config.model.ModelConfig.MODEL_TYPE', 'KERAS'):
        result = inference._predict_similar(mock_image)

    assert result is not None


def test__predict_similar_tflite(inference):
    mock_image = np.random.rand(100, 100, 3)
    inference.model.get_input_details = MagicMock(return_value=[{'index': 0, 'dtype': np.float32}])
    inference.model.get_output_details = MagicMock(return_value=[{'index': 1}])
    inference.model.set_tensor = MagicMock()
    inference.model.invoke = MagicMock()
    inference.model.get_tensor = MagicMock(return_value=np.random.rand(1, 2048))

    with patch('skinvestigatorai.config.model.ModelConfig.MODEL_TYPE', 'TFLITE'):
        result = inference._predict_similar(mock_image)

    assert result is not None
