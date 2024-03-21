import os
import shutil
import pytest
import tensorflow as tf
from PIL import Image
from skinvestigatorai.core.ai.detector import SkinCancerDetector

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"
LOG_DIR = "logs"
MODEL_DIR = "models"
IMG_SIZE = (180, 180)


# Setup and Teardown Functions
@pytest.fixture(scope="module")
def setup_and_teardown_dirs():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    for i in range(5):
        for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
            subdir = os.path.join(directory, str(i))
            os.makedirs(subdir, exist_ok=True)
            img = Image.new('RGB', (100, 100), color='red')
            img.save(os.path.join(subdir, f"img_{i}.jpeg"))

    # Provide setup data for tests
    detector = SkinCancerDetector(TRAIN_DIR, VAL_DIR, TEST_DIR, LOG_DIR, 32, MODEL_DIR, IMG_SIZE)
    yield detector

    # Teardown: remove created directories
    shutil.rmtree("data")
    shutil.rmtree(LOG_DIR)
    shutil.rmtree(MODEL_DIR)


# Tests
def test_verify_images(setup_and_teardown_dirs):
    detector = setup_and_teardown_dirs
    # Intentionally corrupt an image to test verification
    open(os.path.join(TRAIN_DIR, "0/img_0.jpeg"), "w").close()
    invalid_images = detector.verify_images(TRAIN_DIR)
    assert len(invalid_images) == 1
    assert "img_0.jpeg" in invalid_images[0]


def test_preprocess_data(setup_and_teardown_dirs):
    detector = setup_and_teardown_dirs
    train_gen, val_gen, test_gen = detector.preprocess_data()
    assert train_gen is not None
    assert val_gen is not None
    assert test_gen is not None


def test_build_and_train_model(setup_and_teardown_dirs):
    detector = setup_and_teardown_dirs
    detector.build_model()
    train_gen, val_gen, _ = detector.preprocess_data()
    history = detector.train_model(train_gen, val_gen, epochs=1)
    assert 'loss' in history.history


def test_evaluate_model(setup_and_teardown_dirs):
    detector = setup_and_teardown_dirs
    _, _, test_gen = detector.preprocess_data()
    loss, acc, precision, recall, auc = detector.evaluate_model(test_gen)
    assert 0 <= acc <= 1


def test_save_and_load_model(setup_and_teardown_dirs):
    detector = setup_and_teardown_dirs
    detector.build_model()
    filename = "test_model.h5"
    detector.save_model(filename)
    assert os.path.exists(filename)
    assert os.path.exists(filename.replace('.h5', '-quantized.tflite'))
    detector.load_model(filename)
    assert detector.model is not None


if __name__ == "__main__":
    pytest.main()
