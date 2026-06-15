from lesnet.config.model import ModelConfig


def test_model_directory_type():
    assert isinstance(ModelConfig.MODEL_DIRECTORY, str)


def test_model_urls_type():
    assert isinstance(ModelConfig.MODEL_URLS, dict)
    for key, value in ModelConfig.MODEL_URLS.items():
        assert isinstance(key, str)
        assert isinstance(value, str)


def test_released_model_present():
    assert 'M-4s' in ModelConfig.MODEL_URLS
