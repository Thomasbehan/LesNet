import os
import requests
from tqdm import tqdm
from urllib.parse import urlparse, unquote
from skinvestigatorai.config.model import ModelConfig


def downloader(model_name):
    url = ModelConfig.MODEL_URLS.get(model_name)
    if not url:
        print(f"URL for model '{model_name}' not found.")
        return False

    response = requests.get(url, stream=True)

    if response.status_code == 200:
        parsed_url = urlparse(url)
        filename = os.path.basename(unquote(parsed_url.path))

        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kilobyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        model_path = os.path.join(ModelConfig.MODEL_DIRECTORY, filename)

        if not os.path.exists(ModelConfig.MODEL_DIRECTORY):
            os.makedirs(ModelConfig.MODEL_DIRECTORY)

        with open(model_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
            return False

        print(f"Downloaded {model_name} successfully as {filename}.")
        return True
    else:
        print(f"Failed to download {model_name}.")
        return False
