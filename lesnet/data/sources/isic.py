"""ISIC Archive v2 source — resumable, threaded metadata collection + selective image fetch.

Two phases so we can do "malignant-maximal" sourcing cheaply: page the whole archive's
metadata (no images), then download images only for a chosen subset (e.g. all malignant +
suspicious + a balanced benign sample). ``parse`` reads the written metadata.csv into records.
"""
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from lesnet.data.records import LesionRecord, read_csv_rows, to_float

API_URL = "https://api.isic-archive.com/api/v2/images/"
METADATA_FIELDS = ['isic_id', 'url', 'patient_id', 'diagnosis', 'diagnosis_1',
                   'age_approx', 'sex', 'anatom_site_general']


def make_session(pool=32):
    session = requests.Session()
    retries = Retry(
        total=8, connect=5, read=5, backoff_factor=2.0,
        status_forcelist=[429, 500, 502, 503, 504, 520, 522, 524],
        respect_retry_after_header=True, raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_maxsize=pool, pool_connections=pool)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def _specific_diagnosis(clinical):
    for key in ('diagnosis_3', 'diagnosis_2', 'diagnosis_1'):
        if clinical.get(key):
            return clinical[key]
    return clinical.get('diagnosis') or ''


def records_from_page(payload, resolution):
    """Extract metadata rows (incl. a downloadable image url) from one API page."""
    rows = []
    for image in payload['results']:
        clinical = image.get('metadata', {}).get('clinical', {})
        files = image.get('files', {})
        chosen = files.get(resolution) or files.get('full') or files.get('thumbnail_256')
        url = (chosen or {}).get('url')
        if not url:
            continue
        rows.append({
            'isic_id': image['isic_id'], 'url': url,
            'patient_id': clinical.get('patient_id') or image.get('patient_id') or '',
            'diagnosis': _specific_diagnosis(clinical),
            'diagnosis_1': clinical.get('diagnosis_1') or '',
            'age_approx': clinical.get('age_approx', ''),
            'sex': clinical.get('sex', ''),
            'anatom_site_general': clinical.get('anatom_site_1', ''),
        })
    return rows


def collect_metadata(root, resolution='full', page_size=100, limit=None, session=None):
    """Page the whole archive's metadata into <root>/metadata.csv; return the rows (with urls)."""
    os.makedirs(root, exist_ok=True)
    session = session or make_session()
    rows, next_url, params = [], API_URL, {'limit': page_size}
    while next_url and (limit is None or len(rows) < limit):
        response = session.get(next_url, params=params, timeout=120)
        params = None
        if response.status_code != 200:
            break
        payload = response.json()
        rows.extend(records_from_page(payload, resolution))
        next_url = payload.get('next')
    rows = rows[:limit] if limit else rows
    with open(os.path.join(root, 'metadata.csv'), 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in METADATA_FIELDS})
    return rows


def download_images(root, url_by_id, workers=32, session=None):
    """Fetch images for the given {isic_id: url} into <root>/<isic_id>.jpg (skip existing)."""
    os.makedirs(root, exist_ok=True)
    session = session or make_session(workers)
    downloaded = failed = 0

    def _fetch(item):
        isic_id, url = item
        path = os.path.join(root, f"{isic_id}.jpg")
        if os.path.exists(path):
            return True
        response = session.get(url, timeout=120)
        if response.status_code == 200:
            with open(path, 'wb') as handle:
                handle.write(response.content)
            return True
        return False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch, item): item for item in url_by_id.items()}
        for future in as_completed(futures):
            if future.result():
                downloaded += 1
            else:
                failed += 1
    return downloaded, failed


def download(root, limit=None, resolution='full', workers=32):
    """Convenience full download: collect metadata then fetch every image."""
    rows = collect_metadata(root, resolution=resolution, limit=limit)
    download_images(root, {row['isic_id']: row['url'] for row in rows}, workers=workers)


def parse(root, limit=None):
    rows = read_csv_rows(os.path.join(root, 'metadata.csv'))
    rows = rows[:limit] if limit else rows
    records = []
    for row in rows:
        isic_id = row.get('isic_id')
        if not isic_id:
            continue
        records.append(LesionRecord(
            image_path=os.path.join(root, f"{isic_id}.jpg"),
            source_dataset='isic',
            raw_label=row.get('diagnosis') or row.get('diagnosis_1') or 'unknown',
            group_id=row.get('patient_id') or isic_id,
            anatomical_site=row.get('anatom_site_general') or None,
            age=to_float(row.get('age_approx')),
            sex=row.get('sex') or None,
        ))
    return records
