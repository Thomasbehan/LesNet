class ModelConfig(object):
    MODEL_DIRECTORY = 'models/'
    MODEL_URLS = {
        'M-0003':
            'https://github.com/Thomasbehan/LesNet/releases/download/0.0.3/skinvestigator_nano_40MB_91_38_acc.h5',
        'M-0015': 'https://github.com/Thomasbehan/LesNet/releases/download/0.1.5/skinvestigator-lg.h5',
        'M-0015s': 'https://github.com/Thomasbehan/LesNet/releases/download/0.1.5/skinvestigator-sm.tflite',
        'M-0031': 'https://github.com/Thomasbehan/LesNet/releases/download/0.3.1/LesNetM31.keras',
    }
    LOG_DIR = "logs"
    IMG_SIZE = (160, 160)
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    LAYER_1 = 2048
    LAYER_2 = 1024
    LAYER_3 = 1024
    DROPOUT_1 = 0.25
    BASE_LAYERS_TO_UNFREEZE = 15
    L2_LAYER_1 = 0.005
    L2_LAYER_2 = 0.005
    L2_LAYER_3 = 0.005
    AUG_TOTAL = 50000
    CATEGORIES = 30
    EPOCHS = 3000
    MIN_LR = 1e-8
    MIN_LR_DELTA = 1e-4
    LR_PATIENCE = 7
    LR_COOLDOWN = 5
    ES_PATIENCE = 42
    MAX_AUG_PER_IMAGE = 5000000
    TRAIN_DIR = 'data/train'
    MODEL_TYPE = "KERAS"
    MODEL_NAME = "LesNet.keras"
    LABELS_NAME = "LesNet_labels.json"
