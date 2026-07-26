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
        # 4.5.0 family (uploaded to the v4.5.0 release by the model-build automation).
        'M4.5s': 'https://github.com/Thomasbehan/LesNet/releases/download/v4.5.0/LesNet.M4.5s.keras',
        'M4.5m': 'https://github.com/Thomasbehan/LesNet/releases/download/v4.5.0/LesNet.M4.5m.keras',
        'M4.5L': 'https://github.com/Thomasbehan/LesNet/releases/download/v4.5.0/LesNet.M4.5L.keras',
        'M4.5XL': 'https://github.com/Thomasbehan/LesNet/releases/download/v4.5.0/LesNet.M4.5XL.keras',
    }

    # JEPA family (DINOv2 backbones, Apache-2.0) — self-contained tarballs holding the served ONNX
    # encoder plus the fitted heads. Medium powers the live demo; Small is the edge default.
    JEPA_RELEASE = 'v4.6.0'
    JEPA_URLS = {
        'small': f'https://github.com/Thomasbehan/LesNet/releases/download/{JEPA_RELEASE}/lesnet-jepa-small.tar.gz',
        'medium': f'https://github.com/Thomasbehan/LesNet/releases/download/{JEPA_RELEASE}/lesnet-jepa-medium.tar.gz',
        'large': f'https://github.com/Thomasbehan/LesNet/releases/download/{JEPA_RELEASE}/lesnet-jepa-large.tar.gz',
    }
