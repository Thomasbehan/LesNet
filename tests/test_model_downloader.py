import os
from unittest.mock import patch, mock_open

import pytest

from skinvestigatorai.config.model import ModelConfig
from skinvestigatorai.models.downloader import downloader


# Mocking the requests.get function to prevent actual HTTP requests
class MockResponse:
    def __init__(self, url, status_code=200, headers=None, content=b''):
        self.url = url
        self.status_code = status_code
        self.headers = headers if headers is not None else {'content-length': str(len(content))}
        self.content = content

    def iter_content(self, block_size):
        return iter([self.content[i:i + block_size] for i in range(0, len(self.content), block_size)])

    def close(self):
        pass


@pytest.fixture
def mock_requests_get():
    with patch('requests.get') as mock_get:
        yield mock_get


@pytest.fixture
def mock_os_path_exists():
    with patch('os.path.exists') as mock_exists:
        yield mock_exists


@pytest.fixture
def mock_os_makedirs():
    with patch('os.makedirs') as mock_makedirs:
        yield mock_makedirs


@pytest.fixture
def mock_open_file():
    with patch('builtins.open', mock_open()) as mock_file:
        yield mock_file


@pytest.fixture
def mock_tqdm():
    with patch('tqdm.tqdm') as mock_tqdm:
        yield mock_tqdm


def test_downloader_success(mock_requests_get, mock_os_path_exists, mock_os_makedirs, mock_open_file, mock_tqdm):
    model_name = 'M-0003'
    url = ModelConfig.MODEL_URLS[model_name]
    mock_response = MockResponse(url, content=b'test content')
    mock_requests_get.return_value = mock_response
    mock_os_path_exists.return_value = True

    result = downloader(model_name)

    mock_requests_get.assert_called_once_with(url, stream=True)
    mock_open_file.assert_called_once_with(
        os.path.join(ModelConfig.MODEL_DIRECTORY, 'skinvestigator_nano_40MB_91_38_acc.h5'), 'wb')
    assert result is True


def test_downloader_url_not_found():
    model_name = 'M-9999'
    result = downloader(model_name)
    assert result is False


def test_downloader_failed_request(mock_requests_get):
    model_name = 'M-0003'
    url = ModelConfig.MODEL_URLS[model_name]
    mock_requests_get.return_value = MockResponse(url, status_code=404)

    result = downloader(model_name)

    assert result is False


def test_downloader_directory_creation(mock_requests_get, mock_os_path_exists, mock_os_makedirs, mock_open_file,
                                       mock_tqdm):
    model_name = 'M-0003'
    url = ModelConfig.MODEL_URLS[model_name]
    mock_response = MockResponse(url, content=b'test content')
    mock_requests_get.return_value = mock_response
    mock_os_path_exists.return_value = False

    result = downloader(model_name)

    mock_os_makedirs.assert_called_once_with(ModelConfig.MODEL_DIRECTORY)
    assert result is True
