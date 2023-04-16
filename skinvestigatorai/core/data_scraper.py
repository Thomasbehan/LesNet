import os
import json
import requests
from tqdm import tqdm
import concurrent.futures
import mimetypes
import tensorflow.keras.preprocessing.image as image_utils
from PIL import UnidentifiedImageError


class DataScraper:
    def __init__(self, train_dir="data/train", val_dir="data/validation", test_dir="data/test"):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.base_url = "https://api.isic-archive.com/api/v2"
        self.image_list_url = f"{self.base_url}/images/?format=json"

    def _create_output_folders(self):
        # Delete the output folders if they exist
        if os.path.exists(self.train_dir):
            os.system("rm -r " + self.train_dir)
        if os.path.exists(self.val_dir):
            os.system("rm -r " + self.val_dir)
        if os.path.exists(self.test_dir):
            os.system("rm -r " + self.test_dir)

        # Create the output folders if they don't exist
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)

    def _image_safe_check(self, path):
        try:
            img = image_utils.load_img(path)
        except UnidentifiedImageError:
            print(f"Skipping corrupted image: {path}")
            return False
        return True

    def _download_and_save_image(self, image_metadata, output_folder):
        image_id = image_metadata["isic_id"]
        image_url = image_metadata["files"]["full"]["url"]
        response = requests.get(image_url)

        # Get the file extension based on the MIME type
        content_type = response.headers['content-type']
        ext = mimetypes.guess_extension(content_type)

        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        file_path = os.path.join(output_folder, f"{image_id}{ext}")
        with open(file_path, "wb") as f:
            f.write(response.content)

        if not self._image_safe_check(file_path):
            os.remove(file_path)

    def download_images(self, limit=-1):
        self._create_output_folders()

        next_url = self.image_list_url

        def process_image(idx, image_metadata, directory):
            self._download_and_save_image(image_metadata,
                                          directory + "/" + image_metadata["metadata"]["clinical"]["benign_malignant"])

        count = 0
        while next_url and (count < limit or limit == -1):
            print("CURRENT URL: ", next_url)
            response = requests.get(next_url)
            print("RESPONSE: ", response)
            response_data = json.loads(response.content.decode("utf-8"))
            next_url = response_data["next"]
            image_metadata_list = response_data["results"]

            total_images = len(image_metadata_list)
            train_size = int(0.7 * total_images)
            val_size = int(0.2 * total_images)

            print("Downloading " + str(total_images) + " images...")

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for idx, image_metadata in enumerate(image_metadata_list):
                    print("IMAGE: ", image_metadata)
                    if idx < train_size:
                        directory = self.train_dir
                    elif idx < train_size + val_size:
                        directory = self.val_dir
                    else:
                        directory = self.test_dir

                    futures.append(executor.submit(process_image, idx, image_metadata, directory))

                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error downloading image: {e}")

            count += 1
            print("Images downloaded and saved.")


if __name__ == "__main__":
    downloader = DataScraper()
    downloader.download_images()
