import os
import requests

# Define model locations and URLs
MODEL_DIRECTORY = 'models/'
MODEL_URLS = {
    'M-0003':
        'https://github.com/Thomasbehan/SkinVestigatorAI/releases/download/0.0.3/skinvestigator_nano_40MB_91_38_acc.h5',
    'M-0015': 'https://github.com/Thomasbehan/SkinVestigatorAI/releases/download/0.1.5/skinvestigator-lg.h5',
    'M-0015s': 'https://github.com/Thomasbehan/SkinVestigatorAI/releases/download/0.1.5/skinvestigator-sm.tflite',
}


def downloader(model_name):
    """Download a model from a URL into the specified directory."""
    url = MODEL_URLS.get(model_name)
    if not url:
        print(f"URL for model '{model_name}' not found.")
        return False
    response = requests.get(url)
    if response.status_code == 200:
        if not os.path.exists(MODEL_DIRECTORY):
            os.makedirs(MODEL_DIRECTORY)
        model_path = os.path.join(MODEL_DIRECTORY, f"{model_name}.zip")
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {model_name} successfully.")
        return True
    else:
        print(f"Failed to download {model_name}.")
        return False
