from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from skinvestigatorai.config.model import ModelConfig
from skinvestigatorai.services.model import SVModel


@pytest.fixture
def sv_model():
    return SVModel()


def test_create_feature_extractor_tflite(sv_model):
    sv_model.model_type = 'TFLITE'
    mock_model = MagicMock()
    sv_model.model = mock_model
    sv_model.create_feature_extractor()
    assert sv_model.feature_extractor == mock_model


def test_create_feature_extractor_invalid_model_type(sv_model):
    sv_model.model_type = 'INVALID'
    with pytest.raises(ValueError, match="Unsupported model type. Please use 'KERAS' or 'TFLITE'."):
        sv_model.create_feature_extractor()


def test_preprocess_image_for_tflite(sv_model):
    img = np.random.rand(224, 224, 3).astype(np.float32)
    processed_img = sv_model.preprocess_image_for_tflite(img)
    assert processed_img.shape == (ModelConfig.IMG_SIZE[0], ModelConfig.IMG_SIZE[1], 3)
    assert np.max(processed_img) <= 1.0
    assert np.min(processed_img) >= 0.0


def test_evaluate_model(sv_model):
    sv_model.model = MagicMock()
    sv_model.model.evaluate.return_value = [0.5, 0.8, 0.7, 0.6]
    test_datagen = MagicMock()
    test_loss, test_acc, test_precision, test_recall = sv_model.evaluate_model(test_datagen)
    assert test_loss == 0.5
    assert test_acc == 0.8
    assert test_precision == 0.7
    assert test_recall == 0.6


@patch('tensorflow.summary.create_file_writer')
def test_run_experiments(mock_create_file_writer, sv_model):
    sv_model.run_experiments = MagicMock()
    train_ds = MagicMock()
    val_ds = MagicMock()
    sv_model.run_experiments(train_ds, val_ds)
    sv_model.run_experiments.assert_called_once_with(train_ds, val_ds)


def test_save_model(sv_model):
    sv_model.model = MagicMock()
    with patch('builtins.open', MagicMock()):
        with patch('tensorflow.keras.models.Model.save', MagicMock()):
            sv_model.save_model()
            sv_model.model.save.assert_called_once()


def test_load_model(sv_model):
    with patch('os.path.exists', return_value=True):
        with patch('tensorflow.keras.models.load_model', return_value=MagicMock()):
            sv_model.load_model()
            assert isinstance(sv_model.model, MagicMock)
