import os
import argparse
import json
import requests
import concurrent.futures
import mimetypes
import tensorflow.keras.preprocessing.image as image_utils
from PIL import UnidentifiedImageError
from collections import defaultdict


class DataScraper:
    def __init__(self, train_dir="data/train", val_dir="data/validation", test_dir="data/test"):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.base_url = "https://api.isic-archive.com/api/v2"
        self.image_list_url = f"{self.base_url}/images/?format=json"

    def _create_output_folders(self):
        # Create the output folders if they don't exist
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)

    def _image_safe_check(self, path):
        try:
            image_utils.load_img(path)
        except UnidentifiedImageError:
            print(f"Skipping corrupted image: {path}")
            return False
        return True

    def _download_and_save_image(self, image_metadata, output_folder):
        image_id = image_metadata["isic_id"]
        # image_url = image_metadata["files"]["full"]["url"]
        image_url = image_metadata["files"]["thumbnail_256"]["url"]
        response = requests.get(image_url)

        # Get the file extension based on the MIME type
        content_type = response.headers['content-type']
        ext = mimetypes.guess_extension(content_type)

        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        file_path = os.path.join(output_folder, f"{image_id}{ext}")
        # Skip download if file already exists
        if os.path.exists(file_path):
            print(f"File {file_path} already exists, skipping download.")
            return

        with open(file_path, "wb") as f:
            f.write(response.content)

        if not self._image_safe_check(file_path):
            os.remove(file_path)

    def download_images(self, limit=-1):
        self._create_output_folders()

        next_url = self.image_list_url

        def process_image(image_metadata, output_folder):
            if "benign_malignant" in image_metadata["metadata"]["clinical"]:
                self._download_and_save_image(image_metadata,
                                              output_folder + "/" + image_metadata["metadata"]["clinical"][
                                                  "benign_malignant"])
            else:
                print(f"Skipping image {image_metadata['isic_id']} due to missing category information.")

        count = 0
        image_metadata_dict = defaultdict(list)
        while next_url and (count < limit or limit == -1):
            print(str(count) + " CURRENT URL: ", next_url)
            response = requests.get(next_url)
            print("RESPONSE: ", response)
            response_data = json.loads(response.content.decode("utf-8"))
            next_url = response_data["next"]
            image_metadata_list = response_data["results"]

            # Grouping the images by their classification
            for image_metadata in image_metadata_list:
                if "benign_malignant" in image_metadata["metadata"]["clinical"]:
                    category = image_metadata["metadata"]["clinical"]["benign_malignant"]
                    image_metadata_dict[category].append(image_metadata)
                else:
                    print(f"Skipping image {image_metadata['isic_id']} due to missing category information.")

            count += 1

        # Achieving balance by taking the min number of images in each category
        min_images = min(len(image_metadata_dict["benign"]), len(image_metadata_dict["malignant"]))
        print("Achieving balance with " + str(min_images) + " images per category...")

        # Define distribution for train, validation, and test sets
        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1

        for category, image_metadata_list in image_metadata_dict.items():
            total_images = len(image_metadata_list)
            train_size = int(train_ratio * total_images)
            val_size = int(val_ratio * total_images)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for i in range(min_images):
                    if i < train_size:
                        directory = self.train_dir
                    elif i < train_size + val_size:
                        directory = self.val_dir
                    else:
                        directory = self.test_dir

                    futures.append(executor.submit(process_image, image_metadata_list[i], directory))

                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error downloading image: {e}")

        print("Images downloaded and saved.")


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--pages", help="Number of pages to download")
    args = argParser.parse_args()
    downloader = DataScraper()
    downloader.download_images(int(args.pages or -1))
