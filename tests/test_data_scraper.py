import os
import shutil
from unittest.mock import patch
from skinvestigatorai.core.data_scraper import DataScraper


def test_create_output_folders():
    test_train_dir = "test_data/train"
    test_val_dir = "test_data/validation"
    test_test_dir = "test_data/test"

    data_scraper = DataScraper(train_dir=test_train_dir, val_dir=test_val_dir, test_dir=test_test_dir)

    # Call the internal function
    data_scraper._create_output_folders()

    # Check if the directories were created
    assert os.path.exists(test_train_dir)
    assert os.path.exists(test_val_dir)
    assert os.path.exists(test_test_dir)

    # Cleanup
    shutil.rmtree("test_data")


@patch("skinvestigatorai.core.data_scraper.DataScraper.download_images")
def test_download_images(mock_download_images):
    data_scraper = DataScraper()
    data_scraper.download_images()

    # Test if the download_images() method is called once
    mock_download_images.assert_called_once()

# You may add more tests to cover other functions like _image_safe_check, _download_and_save_image, etc.
# However, some of these tests may require mocking external calls to the API and may not be as straightforward.
