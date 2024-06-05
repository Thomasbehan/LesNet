import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm

from skinvestigatorai.config.data import DataConfig
from skinvestigatorai.services.data import Data


class DataScraper:
    def __init__(self, output_dir=DataConfig.OUTPUT_DIR, max_pages=-1):
        self.output_dir = output_dir
        self.api_url = DataConfig.API_URL
        self.failed_downloads_path = os.path.join(self.output_dir, DataConfig.FAILED_DOWNLOADS_LOG_PATH)
        self.max_pages = max_pages
        self.session = self._create_session()
        self.data_service = Data()

    def _create_session(self):
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session

    def _create_output_folders(self):
        for category in ["train"]:
            os.makedirs(os.path.join(self.output_dir, category), exist_ok=True)
        os.makedirs(os.path.dirname(self.failed_downloads_path), exist_ok=True)

    def _download_image(self, image_data):
        image_url, file_path = image_data
        if os.path.exists(file_path):
            return file_path, True
        try:
            response = self.session.get(image_url, timeout=60)
            if response.status_code == 200:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                return file_path, True
            else:
                return image_url, False
        except requests.RequestException:
            return image_url, False

    def balance_dataset_and_save(self):
        """Balance dataset by augmenting images for minority classes."""
        train_dir = os.path.join(self.output_dir, "train")
        categories = [name for name in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, name))]
        category_counts = {category: len([name for name in os.listdir(os.path.join(train_dir, category))
                                          if name.endswith(('.JPG', '.jpg', '.jpeg', '.png'))])
                           for category in categories}
        max_images = max(category_counts.values())

        for category in tqdm(categories, desc="Processing Categories", unit="category"):
            count = category_counts[category]
            if count < max_images:
                additional_images_needed = max_images - count
                category_path = os.path.join(train_dir, category)
                image_paths = [os.path.join(category_path, f) for f in os.listdir(category_path) if
                               f.endswith(('.JPG', '.jpg', '.jpeg', '.png'))]
                print(f"Balancing category: {category} by adding {additional_images_needed} augmented images.")
                try:
                    self.data_service.save_augmented_images(image_paths,
                                                            [category] * len(image_paths),
                                                            train_dir,
                                                            additional_images_needed)
                    print(f"Augmentation completed for {category}")
                except Exception as exc:
                    print(f"Error processing {category}: {exc}")

    def download_images(self):
        self._create_output_folders()
        temp_folder = os.path.join(self.output_dir, "train")
        failed_downloads = set()
        if os.path.exists(self.failed_downloads_path):
            with open(self.failed_downloads_path) as f:
                failed_downloads = {line.strip() for line in f}

        next_url = self.api_url
        params = {'limit': 100, 'offset': 0}
        total_images_downloaded = 0
        page_count = 0

        while next_url and (self.max_pages == -1 or page_count < self.max_pages):
            response = self.session.get(next_url, params=params)
            if response.status_code != 200:
                break

            data = response.json()
            next_url = data.get("next", None)
            page_count += 1

            download_tasks = []
            for image in data['results']:
                isic_id = image['isic_id']
                image_url = image['files']['thumbnail_256']['url']
                diagnosis = image['metadata']['clinical'].get('diagnosis', 'unknown')
                if diagnosis == 'unknown':
                    diagnosis = image['metadata']['clinical'].get('benign_malignant', 'unknown')
                    if diagnosis == 'unknown':
                        print(image['metadata']['clinical'])

                if image_url not in failed_downloads:
                    file_path = os.path.join(temp_folder, diagnosis, f"{isic_id}.jpg")
                    if diagnosis not in ['unknown', 'benign', 'malignant', 'other']:
                        download_tasks.append((image_url, file_path))

            with ThreadPoolExecutor(max_workers=200) as executor:
                future_to_url = {executor.submit(self._download_image, task): task for task in download_tasks}
                for future in as_completed(future_to_url):
                    url, success = future.result()
                    if success:
                        total_images_downloaded += 1
                    else:
                        with open(self.failed_downloads_path, 'a') as f:
                            f.write(f"{url}\n")

            print(f"Total images downloaded: {total_images_downloaded}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download images from ISIC Archive and split into training and testing sets.")
    parser.add_argument("-p", "--pages", type=int, default=-1,
                        help="Number of pages to download. Default is -1, which downloads all pages.")
    args = parser.parse_args()

    scraper = DataScraper(max_pages=args.pages)
    scraper.download_images()
    scraper.balance_dataset_and_save()
