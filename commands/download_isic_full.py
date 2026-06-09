"""Download the ENTIRE ISIC archive (~800k images) — resumable and threaded.

This is built to run for a long time on a real machine (it is NOT expected to finish in a
sandbox). It:
  - pages the public ISIC v2 API following the `next` cursor,
  - persists that cursor to <out>/download_state.json so it resumes after interruption,
  - skips images already on disk,
  - appends rows to <out>/metadata.csv in the schema the ISIC loader reads,
  - downloads concurrently.

Layout is flat (<out>/<isic_id>.jpg + <out>/metadata.csv) so run_build_dataset.py can
consume it directly with --isic-root <out>.

  python commands/download_isic_full.py --out data/isic_full --workers 32
  # resolution: thumbnail_256 (default, ~tens of GB) or full (hundreds of GB+)
  # resume: just re-run the same command; pass --restart to start over.
"""
import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_URL = "https://api.isic-archive.com/api/v2/images/"
METADATA_FIELDS = ['isic_id', 'patient_id', 'diagnosis', 'diagnosis_1',
                   'age_approx', 'sex', 'anatom_site_general']


def _session(pool):
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries, pool_maxsize=pool)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def _specific_diagnosis(clinical):
    for key in ('diagnosis_3', 'diagnosis_2', 'diagnosis_1'):
        if clinical.get(key):
            return clinical[key]
    return clinical.get('diagnosis') or ''


def _records_from_page(payload, resolution):
    records = []
    for image in payload['results']:
        clinical = image.get('metadata', {}).get('clinical', {})
        files = image.get('files', {})
        chosen = files.get(resolution) or files.get('thumbnail_256') or files.get('full')
        url = (chosen or {}).get('url')
        if not url:
            continue
        records.append({
            'isic_id': image['isic_id'], 'url': url,
            'patient_id': clinical.get('patient_id') or image.get('patient_id') or '',
            'diagnosis': _specific_diagnosis(clinical),
            'diagnosis_1': clinical.get('diagnosis_1') or '',
            'age_approx': clinical.get('age_approx', ''),
            'sex': clinical.get('sex', ''),
            'anatom_site_general': clinical.get('anatom_site_1', ''),
        })
    return records


def main():
    parser = argparse.ArgumentParser(description="Resumable full ISIC archive downloader.")
    parser.add_argument('--out', default='data/isic_full')
    parser.add_argument('--resolution', default='thumbnail_256', choices=['thumbnail_256', 'full'])
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--page-size', type=int, default=100)
    parser.add_argument('--max', type=int, default=None, help="Optional cap (omit to fetch all).")
    parser.add_argument('--restart', action='store_true', help="Ignore saved state and start over.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    state_path = os.path.join(args.out, 'download_state.json')
    metadata_path = os.path.join(args.out, 'metadata.csv')
    session = _session(args.workers)

    next_url, params, downloaded, pages = API_URL, {'limit': args.page_size}, 0, 0
    if os.path.exists(state_path) and not args.restart:
        state = json.load(open(state_path, encoding='utf-8'))
        next_url = state.get('next_url') or API_URL
        params = None if state.get('next_url') else {'limit': args.page_size}
        downloaded, pages = state.get('downloaded', 0), state.get('pages', 0)
        print(f"Resuming: {downloaded} downloaded across {pages} pages.")

    writer_is_new = not os.path.exists(metadata_path)
    metadata_file = open(metadata_path, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(metadata_file, fieldnames=METADATA_FIELDS)
    if writer_is_new:
        writer.writeheader()

    def _fetch(record):
        path = os.path.join(args.out, f"{record['isic_id']}.jpg")
        if os.path.exists(path):
            return record, False  # already have it
        response = session.get(record['url'], timeout=120)
        if response.status_code == 200:
            with open(path, 'wb') as handle:
                handle.write(response.content)
            return record, True
        return record, None  # failed

    try:
        while next_url and (args.max is None or downloaded < args.max):
            response = session.get(next_url, params=params, timeout=120)
            params = None
            if response.status_code != 200:
                print(f"API error {response.status_code}; stopping.")
                break
            payload = response.json()
            records = _records_from_page(payload, args.resolution)

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(_fetch, record): record for record in records}
                for future in as_completed(futures):
                    record, is_new = future.result()
                    if is_new:
                        downloaded += 1
                        writer.writerow({field: record.get(field, '') for field in METADATA_FIELDS})

            pages += 1
            metadata_file.flush()
            next_url = payload.get('next')
            json.dump({'next_url': next_url, 'downloaded': downloaded, 'pages': pages},
                      open(state_path, 'w', encoding='utf-8'))
            print(f"page {pages}: {downloaded} images downloaded so far", flush=True)
    finally:
        metadata_file.close()

    print(f"Done. {downloaded} images in {args.out}")


if __name__ == '__main__':
    main()
