class ModelConfig(object):
    MODEL_DIRECTORY = 'models/'
    QUERY_DIR = 'fsd/query/'
    SUPPORT_DIR = 'fsd/support/'
    FS_NUM_IMAGES_PER_CLASS = 3
    MODEL_URLS = {
        'M-0003':
            'https://github.com/Thomasbehan/LesNet/releases/download/0.0.3/skinvestigator_nano_40MB_91_38_acc.h5',
        'M-0015': 'https://github.com/Thomasbehan/LesNet/releases/download/0.1.5/skinvestigator-lg.h5',
        'M-0015s': 'https://github.com/Thomasbehan/LesNet/releases/download/0.1.5/skinvestigator-sm.tflite',
        'M-0031': 'https://github.com/Thomasbehan/LesNet/releases/download/0.3.1/LesNet.keras',
        'M-0031s': 'https://github.com/Thomasbehan/LesNet/releases/download/0.3.1/LesNet.tflite',
    }
    LOG_DIR = "logs"
    IMG_SIZE = (160, 160)
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    CONV_LAYER_1 = 64
    BN_LAYER_1 = 64
    BN_LAYER_2 = 128
    BN_LAYER_3 = 256
    BN_LAYER_4 = 512
    STAGE_1_LAYERS = 3
    STAGE_2_LAYERS = 7
    STAGE_3_LAYERS = 35
    STAGE_4_LAYERS = 2
    DROPOUT_1 = 0.5
    DROPOUT_2 = 0.5
    RB_L2_LAYER_1 = 0.001
    L2_LAYER_1 = 0.001
    AUG_TOTAL = 50000
    CATEGORIES = 27
    EPOCHS = 100
    MIN_LR = 1e-6
    MIN_LR_DELTA = 1e-4
    LR_PATIENCE = 5
    LR_COOLDOWN = 3
    ES_PATIENCE = 20
    MAX_AUG_PER_IMAGE = 50000
    TRAIN_DIR = 'data/train'
    MODEL_TYPE = "KERAS"
    MODEL_NAME = "LesNet.keras"
    LABELS_NAME = "LesNet_labels.json"
