import os
import pytest
from PIL import Image
from skinvestigatorai.services.inference import Inference

TRAIN_DIR = "data/train"
LOG_DIR = "logs"
MODEL_DIR = "models"
IMG_SIZE = (180, 180)


# Setup and Teardown Functions
@pytest.fixture(scope="module")
def get_detector():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    for i in range(5):
        for directory in [TRAIN_DIR]:
            subdir = os.path.join(directory, str(i))
            os.makedirs(subdir, exist_ok=True)
            img = Image.new('RGB', (100, 100), color='red')
            img.save(os.path.join(subdir, f"img_{i}.jpeg"))

    # Provide setup data for tests
    detector = Inference(TRAIN_DIR, LOG_DIR, 8, MODEL_DIR, IMG_SIZE)
    yield detector


# Tests
def test_verify_images(get_detector):
    detector = get_detector
    # Intentionally corrupt an image to test verification
    open(os.path.join(TRAIN_DIR, "0/img_0.jpeg"), "w").close()
    invalid_images = detector.verify_images(TRAIN_DIR)
    assert len(invalid_images) == 1
    assert "img_0.jpeg" in invalid_images[0]


def test_build_model_and_process_data(get_detector):
    detector = get_detector
    detector.build_model()


if __name__ == "__main__":
    pytest.main()
