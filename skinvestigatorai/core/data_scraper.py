import os
import argparse
import requests
import shutil
from sklearn.model_selection import train_test_split

class DataScraper:
    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        self.api_url = "https://api.isic-archive.com/api/v2/images"
        self.failed_downloads_path = os.path.join(self.output_dir, "failed_downloads.txt")

    def _create_output_folders(self):
        for category in ["train", "test", "temp"]:
            for label in ["benign", "malignant"]:
                os.makedirs(os.path.join(self.output_dir, category, label), exist_ok=True)
        os.makedirs(os.path.dirname(self.failed_downloads_path), exist_ok=True)

    def _download_image(self, image_url, file_path):
        if os.path.exists(file_path):
            print(f"Already downloaded: {file_path}")
            return True
        try:
            response = requests.get(image_url, timeout=10)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded: {file_path}")
                return True
            else:
                print(f"Failed to download: {image_url}")
                with open(self.failed_downloads_path, 'a') as f:
                    f.write(f"{image_url}\n")
                return False
        except requests.RequestException as e:
            print(f"Error downloading {image_url}: {e}")
            with open(self.failed_downloads_path, 'a') as f:
                f.write(f"{image_url}\n")
            return False

    def _split_data(self, images, test_size=0.2):
        train, test = train_test_split(images, test_size=test_size, random_state=42)
        return train, test

    def _move_images(self, images, source_folder, dest_folder):
        for image in images:
            shutil.move(os.path.join(source_folder, image), os.path.join(dest_folder, image))

    def download_and_split_images(self):
        self._create_output_folders()
        temp_folder = os.path.join(self.output_dir, "temp")

        # Load failed downloads list to skip them
        failed_downloads = set()
        if os.path.exists(self.failed_downloads_path):
            with open(self.failed_downloads_path) as f:
                failed_downloads = {line.strip() for line in f}

        next_url = self.api_url
        params = {'limit': 100, 'offset': 0}
        total_images_downloaded = 0

        while next_url:
            response = requests.get(next_url, params=params)
            if response.status_code != 200:
                break

            data = response.json()
            next_url = data.get("next", None)

            for image in data['results']:
                isic_id = image['isic_id']
                image_url = image['files']['full']['url']
                benign_malignant = image['metadata']['clinical'].get('benign_malignant', 'unknown')
                if benign_malignant in ['benign', 'malignant'] and image_url not in failed_downloads:
                    file_path = os.path.join(temp_folder, benign_malignant, f"{isic_id}.jpg")
                    if self._download_image(image_url, file_path):
                        total_images_downloaded += 1

            print(f"Total images downloaded: {total_images_downloaded}")

        # Split and move images to train and test folders
        for label in ['benign', 'malignant']:
            images = [img for img in os.listdir(os.path.join(temp_folder, label)) if img.endswith(".jpg")]
            train_images, test_images = self._split_data(images)
            self._move_images(train_images, os.path.join(temp_folder, label), os.path.join(self.output_dir, "train", label))
            self._move_images(test_images, os.path.join(temp_folder, label), os.path.join(self.output_dir, "test", label))

        shutil.rmtree(temp_folder)
        print("Images split into train and test sets.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images from ISIC Archive and split into training and testing sets.")
    args = parser.parse_args()

    scraper = DataScraper()
    scraper.download_and_split_images()
