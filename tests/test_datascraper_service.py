import os
from unittest.mock import patch, mock_open

import pytest

from skinvestigatorai.config.data import DataConfig
from skinvestigatorai.services.data import Data
from skinvestigatorai.services.data_scaper import DataScraper


@pytest.fixture
def mock_data_scraper():
    with patch('skinvestigatorai.services.data.Data', spec=Data):
        scraper = DataScraper(output_dir='test_output', max_pages=1)
        yield scraper


def test_create_session(mock_data_scraper):
    session = mock_data_scraper._create_session()
    assert session is not None
    assert session.adapters['http://'].max_retries.total == 5
    assert session.adapters['https://'].max_retries.total == 5


@patch('os.makedirs')
def test_create_output_folders(mock_makedirs, mock_data_scraper):
    mock_data_scraper._create_output_folders()
    mock_makedirs.assert_any_call(os.path.join('test_output', 'train'), exist_ok=True)
    mock_makedirs.assert_any_call(os.path.dirname(os.path.join('test_output', DataConfig.FAILED_DOWNLOADS_LOG_PATH)),
                                  exist_ok=True)


@patch('os.path.exists', return_value=False)
@patch('builtins.open', new_callable=mock_open)
@patch('requests.Session.get')
def test_download_image_success(mock_get, mock_open_file, mock_path_exists, mock_data_scraper):
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = b"fake_image_data"
    image_data = ('http://example.com/image.jpg', 'test_output/train/category/image.jpg')

    file_path, success = mock_data_scraper._download_image(image_data)

    assert success
    assert file_path == 'test_output/train/category/image.jpg'
    mock_open_file.assert_called_once_with('test_output/train/category/image.jpg', 'wb')


@patch('os.path.exists', return_value=True)
def test_download_image_exists(mock_path_exists, mock_data_scraper):
    image_data = ('http://example.com/image.jpg', 'test_output/train/category/image.jpg')

    file_path, success = mock_data_scraper._download_image(image_data)

    assert success
    assert file_path == 'test_output/train/category/image.jpg'


@patch('requests.Session.get')
@patch('os.makedirs')
@patch('os.path.exists', return_value=False)
@patch('builtins.open', new_callable=mock_open)
def test_download_images(mock_open_file, mock_path_exists, mock_makedirs, mock_get, mock_data_scraper):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "results": [{
            "isic_id": "image1",
            "files": {"thumbnail_256": {"url": "http://example.com/image1.jpg"}},
            "metadata": {"clinical": {"diagnosis": "melanoma"}}
        }],
        "next": None
    }

    mock_data_scraper.download_images()

    mock_makedirs.assert_called()
    mock_get.assert_called()
    mock_open_file.assert_called()


if __name__ == "__main__":
    pytest.main()
