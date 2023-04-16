import os
import tempfile
import pytest
from skinvestigatorai.core.ai.detector import SkinCancerDetector

train_dir = 'data/train'
val_dir = 'data/validation'
test_dir = 'data/test'


@pytest.fixture
def detector():
    return SkinCancerDetector(train_dir, val_dir, test_dir)


def test_preprocess_data(detector):
    train_generator, val_generator, test_datagen = detector.preprocess_data()
    assert train_generator is not None
    assert val_generator is not None
    assert test_datagen is not None


def test_build_model(detector):
    num_classes = 5
    detector.build_model(num_classes)
    assert detector.model is not None
    assert len(detector.model.layers) > 0


def test_train_model(detector):
    # Add test code to train the model here
    pass


def test_evaluate_model(detector):
    # Add test code to evaluate the model here
    pass


def test_save_model(detector):
    detector.build_model(5)  # Change this to the number of classes you have
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'test_model.h5')
        detector.save_model(model_path)
        assert os.path.exists(model_path)
