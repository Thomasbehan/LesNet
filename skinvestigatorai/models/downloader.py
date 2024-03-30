import os
import requests
from tqdm import tqdm

MODEL_DIRECTORY = 'models/'
MODEL_URLS = {
    'M-0003':
        'https://github.com/Thomasbehan/SkinVestigatorAI/releases/download/0.0.3/skinvestigator_nano_40MB_91_38_acc.h5',
    'M-0015': 'https://github.com/Thomasbehan/SkinVestigatorAI/releases/download/0.1.5/skinvestigator-lg.h5',
    'M-0015s': 'https://github.com/Thomasbehan/SkinVestigatorAI/releases/download/0.1.5/skinvestigator-sm.tflite',
}


def downloader(model_name):
    url = MODEL_URLS.get(model_name)
    if not url:
        print(f"URL for model '{model_name}' not found.")
        return False

    response = requests.get(url, stream=True)

    if response.status_code == 200:
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kilobyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        model_path = os.path.join(MODEL_DIRECTORY, f"{model_name}")

        if not os.path.exists(MODEL_DIRECTORY):
            os.makedirs(MODEL_DIRECTORY)

        with open(model_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
            return False

        print(f"Downloaded {model_name} successfully.")
        return True
    else:
        print(f"Failed to download {model_name}.")
        return False
