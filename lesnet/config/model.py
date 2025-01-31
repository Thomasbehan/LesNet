class ModelConfig(object):
    MODEL_DIRECTORY = 'models/'
    MODEL_URLS = {
        'M-0003':
            'https://github.com/Thomasbehan/LesNet/releases/download/0.0.3/skinvestigator_nano_40MB_91_38_acc.h5',
        'M-0015': 'https://github.com/Thomasbehan/LesNet/releases/download/0.1.5/skinvestigator-lg.h5',
        'M-0015s': 'https://github.com/Thomasbehan/LesNet/releases/download/0.1.5/skinvestigator-sm.tflite',
        'M-0031': 'https://github.com/Thomasbehan/LesNet/releases/download/0.3.1/LesNetM31.keras',
        'M-4': 'https://github.com/Thomasbehan/LesNet/releases/download/4.0.0/LesNet.M-4.keras',
    }
    LOG_DIR = "logs"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 1028
    LEARNING_RATE = 1e-4
    GLOBAL_WEIGHT_DECAY = 1e-6
    LAYER_1 = 128
    LAYER_2 = 64
    DROPOUT_1 = 0.3
    L2_LAYER_1 = 1e-5
    L2_LAYER_2 = 1e-4
    L2_LAYER_3 = 1e-4
    AUG_TOTAL = 10000
    CATEGORIES = 42
    EPOCHS = 300
    MIN_LR = 1e-7
    MIN_LR_DELTA = 1e-4
    LR_PATIENCE = 4
    LR_COOLDOWN = 3
    ES_PATIENCE = 6
    MAX_AUG_PER_IMAGE = 10
    BASE_LAYERS_TO_UNFREEZE = 10
    TRAIN_DIR = 'data/train'
    MODEL_TYPE = "KERAS"
    MODEL_NAME = "LesNet.keras"
    LABELS_NAME = "LesNet_labels.json"
    TPU_Train = False
    AUGMENTATION_ENABLED = False
