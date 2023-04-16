import pytest
from unittest.mock import MagicMock, patch

# Import the main function from train.py
from skinvestigatorai.core.ai.train import main, train_dir, val_dir, test_dir

# Mock the SkinCancerDetector class and its methods
@pytest.fixture
def mock_detector(monkeypatch):
    with patch('skinvestigatorai.core.ai.train.SkinCancerDetector') as mock:
        yield mock

def test_main(mock_detector):
    # Set the return values of the preprocess_data() method
    mock_detector.return_value.preprocess_data.return_value = (MagicMock(), MagicMock(), MagicMock())

    # Run the main function
    main('skin_cancer_detection_model_all_GPU.h5')

    # Check if the SkinCancerDetector constructor is called with the correct arguments
    mock_detector.assert_called_once_with(train_dir, val_dir, test_dir)

    # Get the detector instance from the constructor call
    detector_instance = mock_detector.return_value

    # Check if the instance methods are called in the correct order
    detector_instance.preprocess_data.assert_called_once()
    detector_instance.build_model.assert_called_once()
    detector_instance.train_model.assert_called_once()
    detector_instance.evaluate_model.assert_called_once()
    detector_instance.save_model.assert_called_once_with('skin_cancer_detection_model_all_GPU.h5')
