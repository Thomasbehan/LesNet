import os
import pytest
from PIL import Image
from skinvestigatorai.services.detector_service import SkinCancerDetector

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"
LOG_DIR = "logs"
MODEL_DIR = "models"
IMG_SIZE = (180, 180)


# Setup and Teardown Functions
@pytest.fixture(scope="module")
def get_detector():
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


# Tests
def test_verify_images(get_detector):
    detector = get_detector
    # Intentionally corrupt an image to test verification
    open(os.path.join(TRAIN_DIR, "0/img_0.jpeg"), "w").close()
    invalid_images = detector.verify_images(TRAIN_DIR)
    assert len(invalid_images) == 1
    assert "img_0.jpeg" in invalid_images[0]


def test_preprocess_data(get_detector):
    detector = get_detector
    train_gen, val_gen, test_gen = detector.preprocess_data()
    assert train_gen is not None
    assert val_gen is not None
    assert test_gen is not None


def test_build_model_and_process_data(get_detector):
    detector = get_detector
    detector.build_model()
    train_gen, val_gen, _ = detector.preprocess_data()


def test_evaluate_model(get_detector):
    detector = get_detector
    _, _, test_gen = detector.preprocess_data()
    test_loss, test_acc, test_precision, test_recall, test_auc, test_binary_accuracy = detector.evaluate_model(test_gen)
    assert isinstance(test_acc, (int, float))
    assert isinstance(test_loss, (int, float))
    assert isinstance(test_precision, (int, float))
    assert isinstance(test_recall, (int, float))
    assert isinstance(test_auc, (int, float))
    assert isinstance(test_binary_accuracy, (int, float))


if __name__ == "__main__":
    pytest.main()
