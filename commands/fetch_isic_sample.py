"""Fetch a small REAL sample from the public ISIC v2 API for development/testing.

  python commands/fetch_isic_sample.py --out data/isic --max 400

Writes <out>/<isic_id>.jpg thumbnails + <out>/metadata.csv in the schema the ISIC loader
reads. Uses the diagnosis hierarchy (most specific available) as the label.
"""
import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

API_URL = "https://api.isic-archive.com/api/v2/images/"


def _collect(session, max_images, page_size):
    images = []
    url, params = API_URL, {'limit': page_size}
    while url and len(images) < max_images:
        response = session.get(url, params=params, timeout=60)
        params = None  # subsequent 'next' URLs already carry query params
        if response.status_code != 200:
            break
        payload = response.json()
        for image in payload['results']:
            clinical = image.get('metadata', {}).get('clinical', {})
            diagnosis = (clinical.get('diagnosis_3') or clinical.get('diagnosis_2')
                         or clinical.get('diagnosis_1'))
            thumbnail = image.get('files', {}).get('thumbnail_256', {}).get('url')
            if not diagnosis or not thumbnail:
                continue
            images.append({
                'isic_id': image['isic_id'], 'url': thumbnail, 'diagnosis': diagnosis,
                'age_approx': clinical.get('age_approx', ''), 'sex': clinical.get('sex', ''),
                'anatom_site_general': clinical.get('anatom_site_1', ''),
            })
            if len(images) >= max_images:
                break
        url = payload.get('next')
    return images


def main():
    parser = argparse.ArgumentParser(description="Download a small real ISIC sample.")
    parser.add_argument('--out', default='data/isic')
    parser.add_argument('--max', type=int, default=400)
    parser.add_argument('--page-size', type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    session = requests.Session()
    images = _collect(session, args.max, args.page_size)

    def _download(item):
        response = session.get(item['url'], timeout=60)
        if response.status_code == 200:
            with open(os.path.join(args.out, f"{item['isic_id']}.jpg"), 'wb') as handle:
                handle.write(response.content)
            return item
        return None

    downloaded = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(_download, item): item for item in images}
        for future in as_completed(futures):
            result = future.result()
            if result:
                downloaded.append(result)

    fields = ['isic_id', 'patient_id', 'diagnosis', 'age_approx', 'sex', 'anatom_site_general']
    with open(os.path.join(args.out, 'metadata.csv'), 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in downloaded:
            writer.writerow({
                'isic_id': item['isic_id'], 'patient_id': '', 'diagnosis': item['diagnosis'],
                'age_approx': item['age_approx'], 'sex': item['sex'],
                'anatom_site_general': item['anatom_site_general'],
            })
    print(f"Downloaded {len(downloaded)} real ISIC images to {args.out}")


if __name__ == '__main__':
    main()
