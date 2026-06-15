class ModelConfig(object):
    """Released-model registry used by the web app and the model downloader.

    The triage training/inference hyperparameters live in lesnet.ml.config.PipelineConfig.
    """
    MODEL_DIRECTORY = 'models/'
    MODEL_URLS = {
        'M-0003':
            'https://github.com/Thomasbehan/LesNet/releases/download/0.0.3/skinvestigator_nano_40MB_91_38_acc.h5',
        'M-0015': 'https://github.com/Thomasbehan/LesNet/releases/download/0.1.5/skinvestigator-lg.h5',
        'M-0015s': 'https://github.com/Thomasbehan/LesNet/releases/download/0.1.5/skinvestigator-sm.tflite',
        'M-0031': 'https://github.com/Thomasbehan/LesNet/releases/download/0.3.1/LesNetM31.keras',
        'M-4s': 'https://github.com/Thomasbehan/LesNet/releases/download/4.1.0/LesNet.M-4s.keras',
    }
