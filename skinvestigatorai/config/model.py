class ModelConfig(object):
    MODEL_DIRECTORY = 'models/'
    MODEL_URLS = {
        'M-0003':
            'https://github.com/Thomasbehan/SkinVestigatorAI/releases/download/0.0.3/skinvestigator_nano_40MB_91_38_acc.h5',
        'M-0015': 'https://github.com/Thomasbehan/SkinVestigatorAI/releases/download/0.1.5/skinvestigator-lg.h5',
        'M-0015s': 'https://github.com/Thomasbehan/SkinVestigatorAI/releases/download/0.1.5/skinvestigator-sm.tflite',
    }
    LOG_DIR = "logs"
    IMG_SIZE = (125, 125)
    BATCH_SIZE = 16
    AUG_TOTAL = 50000
    CATEGORIES = 28
    EPOCHS = 2
    MAX_AUG_PER_IMAGE = 10
    TRAIN_DIR = 'data/train'
    MODEL_TYPE = "KERAS"
    MODEL_NAME = "skinvestigator.keras"
    LABELS_NAME = "skinvestigator_labels.json"
