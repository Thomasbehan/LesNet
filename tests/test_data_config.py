from skinvestigatorai.config.data import DataConfig


def test_api_url():
    assert DataConfig.API_URL == "https://api.isic-archive.com/api/v2/images"


def test_api_url_type():
    assert isinstance(DataConfig.API_URL, str)


def test_output_dir_type():
    assert isinstance(DataConfig.OUTPUT_DIR, str)


def test_failed_downloads_log_path_type():
    assert isinstance(DataConfig.FAILED_DOWNLOADS_LOG_PATH, str)


def test_n_splits_type():
    assert isinstance(DataConfig.N_SPLITS, int)
