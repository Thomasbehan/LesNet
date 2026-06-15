"""ISIC Archive v2 source — resumable, threaded metadata collection + selective image fetch.

Two phases so we can do "malignant-maximal" sourcing cheaply: page the whole archive's
metadata (no images), then download images only for a chosen subset (e.g. all malignant +
suspicious + a balanced benign sample). ``parse`` reads the written metadata.csv into records.
"""
import csv
import json
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


def url_map(root):
    """{isic_id: image url} read back from a written metadata.csv (for selective download)."""
    return {row['isic_id']: row['url']
            for row in read_csv_rows(os.path.join(root, 'metadata.csv'))
            if row.get('isic_id') and row.get('url')}


def collect_metadata(root, resolution='full', page_size=100, limit=None, session=None):
    """Page the archive's metadata into <root>/metadata.csv — incremental + resumable.

    Each page is appended immediately and the cursor saved to metadata_state.json, so a long
    crawl survives interruption (re-run resumes) and is observable while running. Returns the
    rows collected *this call*; use ``url_map(root)`` for the full id->url map after completion.
    """
    os.makedirs(root, exist_ok=True)
    session = session or make_session()
    metadata_path = os.path.join(root, 'metadata.csv')
    state_path = os.path.join(root, 'metadata_state.json')

    next_url, params, seen = API_URL, {'limit': page_size}, set()
    if os.path.exists(state_path) and os.path.exists(metadata_path):
        state = json.load(open(state_path, encoding='utf-8'))
        next_url = state.get('next_url') or API_URL
        params = None if state.get('next_url') else {'limit': page_size}
        seen = {row['isic_id'] for row in read_csv_rows(metadata_path)}

    handle = open(metadata_path, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
    if not seen:
        writer.writeheader()

    collected = []
    try:
        while next_url and (limit is None or len(seen) < limit):
            response = session.get(next_url, params=params, timeout=120)
            params = None
            if response.status_code != 200:
                break
            payload = response.json()
            for row in records_from_page(payload, resolution):
                if row['isic_id'] in seen:
                    continue
                writer.writerow({field: row.get(field, '') for field in METADATA_FIELDS})
                seen.add(row['isic_id'])
                collected.append(row)
            handle.flush()
            next_url = payload.get('next')
            json.dump({'next_url': next_url, 'collected': len(seen)},
                      open(state_path, 'w', encoding='utf-8'))
            print(f"isic metadata: {len(seen)} rows", flush=True)
    finally:
        handle.close()
    return collected


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
    collect_metadata(root, resolution=resolution, limit=limit)
    download_images(root, url_map(root), workers=workers)


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
