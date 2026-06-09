from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lesnet.services.inference import Inference
from lesnet.services.model import SVModel


@pytest.fixture
def mock_svmodel():
    sv_model = MagicMock(SVModel)
    sv_model.load_model.return_value = (MagicMock(), ["class1", "class2"])
    sv_model.preprocess_image_for_tflite = lambda x: x
    return sv_model


@pytest.fixture
def inference(mock_svmodel):
    with patch('lesnet.services.inference.SVModel', return_value=mock_svmodel):
        service = Inference()
    # Bypass real image decoding — predict() only needs an array of the right rank.
    service.data_service.load_image_for_prediction = MagicMock(
        return_value=np.zeros((224, 224, 3), dtype=np.float32)
    )
    return service


def test_predict_returns_ranked_predictions(inference):
    inference.model.predict = MagicMock(return_value=np.array([[0.1, 0.9]]))

    result = inference.predict(MagicMock())

    assert isinstance(result, dict)
    assert 'predictions' in result
    assert result['predictions'][0]['label'] == 'class2'
    assert result['low_confidence'] is False


def test_predict_flags_low_confidence(inference):
    inference.model.predict = MagicMock(return_value=np.array([[0.3, 0.2]]))

    result = inference.predict(MagicMock())

    assert isinstance(result, dict)
    assert result['low_confidence'] is True


def test_is_image_similar(inference):
    mock_image = np.random.rand(100, 100, 3)

    inference.dataset_embedding = np.random.rand(2048)
    inference._predict_similar = MagicMock(return_value=np.random.rand(2048))

    result = inference.is_image_similar(mock_image, threshold=0.5)

    assert result in [True, False]


def test__predict_similar_keras(inference):
    mock_image = np.random.rand(100, 100, 3)
    inference.model.predict = MagicMock(return_value=np.random.rand(1, 2048))

    with patch('lesnet.config.model.ModelConfig.MODEL_TYPE', 'KERAS'):
        result = inference._predict_similar(mock_image)

    assert result is not None


def test__predict_similar_tflite(inference):
    mock_image = np.random.rand(100, 100, 3)
    inference.model.get_input_details = MagicMock(return_value=[{'index': 0, 'dtype': np.float32}])
    inference.model.get_output_details = MagicMock(return_value=[{'index': 1}])
    inference.model.set_tensor = MagicMock()
    inference.model.invoke = MagicMock()
    inference.model.get_tensor = MagicMock(return_value=np.random.rand(1, 2048))

    with patch('lesnet.config.model.ModelConfig.MODEL_TYPE', 'TFLITE'):
        result = inference._predict_similar(mock_image)

    assert result is not None
