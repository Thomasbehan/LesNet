from skinvestigatorai.config.model import ModelConfig


def test_model_directory_type():
    assert isinstance(ModelConfig.MODEL_DIRECTORY, str)


def test_model_urls_type():
    assert isinstance(ModelConfig.MODEL_URLS, dict)
    for key, value in ModelConfig.MODEL_URLS.items():
        assert isinstance(key, str)
        assert isinstance(value, str)


def test_log_dir_type():
    assert isinstance(ModelConfig.LOG_DIR, str)


def test_img_size_type():
    assert isinstance(ModelConfig.IMG_SIZE, tuple)
    assert len(ModelConfig.IMG_SIZE) == 2
    assert isinstance(ModelConfig.IMG_SIZE[0], int)
    assert isinstance(ModelConfig.IMG_SIZE[1], int)


def test_batch_size_type():
    assert isinstance(ModelConfig.BATCH_SIZE, int)


def test_learning_rate_type():
    assert isinstance(ModelConfig.LEARNING_RATE, float)


def test_layer_1_type():
    assert isinstance(ModelConfig.LAYER_1, int)


def test_layer_2_type():
    assert isinstance(ModelConfig.LAYER_2, int)


def test_layer_3_type():
    assert isinstance(ModelConfig.LAYER_3, int)


def test_dropout_1_type():
    assert isinstance(ModelConfig.DROPOUT_1, float)


def test_base_layers_to_unfreeze_type():
    assert isinstance(ModelConfig.BASE_LAYERS_TO_UNFREEZE, int)


def test_l2_layer_1_type():
    assert isinstance(ModelConfig.L2_LAYER_1, float)


def test_l2_layer_2_type():
    assert isinstance(ModelConfig.L2_LAYER_2, float)


def test_l2_layer_3_type():
    assert isinstance(ModelConfig.L2_LAYER_3, float)


def test_aug_total_type():
    assert isinstance(ModelConfig.AUG_TOTAL, int)


def test_categories_type():
    assert isinstance(ModelConfig.CATEGORIES, int)


def test_epochs_type():
    assert isinstance(ModelConfig.EPOCHS, int)


def test_min_lr_type():
    assert isinstance(ModelConfig.MIN_LR, float)


def test_min_lr_delta_type():
    assert isinstance(ModelConfig.MIN_LR_DELTA, float)


def test_lr_patience_type():
    assert isinstance(ModelConfig.LR_PATIENCE, int)


def test_lr_cooldown_type():
    assert isinstance(ModelConfig.LR_COOLDOWN, int)


def test_es_patience_type():
    assert isinstance(ModelConfig.ES_PATIENCE, int)


def test_max_aug_per_image_type():
    assert isinstance(ModelConfig.MAX_AUG_PER_IMAGE, int)


def test_train_dir_type():
    assert isinstance(ModelConfig.TRAIN_DIR, str)


def test_model_type_type():
    assert isinstance(ModelConfig.MODEL_TYPE, str)


def test_model_name_type():
    assert isinstance(ModelConfig.MODEL_NAME, str)


def test_labels_name_type():
    assert isinstance(ModelConfig.LABELS_NAME, str)
