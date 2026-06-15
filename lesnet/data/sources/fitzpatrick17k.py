"""Fitzpatrick17k source — clinical images with Fitzpatrick skin-type labels.

The dataset ships a CSV (``fitzpatrick17k.csv``) with image URLs; ``download`` best-effort
fetches them. Many links rot, so missing images are tolerated and dropped later by the
quality gate.
"""
import os

import requests

from lesnet.data.records import LesionRecord, read_csv_rows, to_int


def download(root, limit=None, session=None):
    """Best-effort fetch of images referenced by a 'url' column into <root>/images/."""
    rows = read_csv_rows(os.path.join(root, 'fitzpatrick17k.csv'))
    rows = rows[:limit] if limit else rows
    images_dir = os.path.join(root, 'images')
    os.makedirs(images_dir, exist_ok=True)
    session = session or requests
    for row in rows:
        key, url = row.get('md5hash'), row.get('url')
        if not key or not url:
            continue
        path = os.path.join(images_dir, f"{key}.jpg")
        if os.path.exists(path):
            continue
        try:
            response = session.get(url, timeout=60)
        except Exception:  # noqa: BLE001 - rotten link; quality gate drops the missing image
            continue
        if response.status_code == 200:
            with open(path, 'wb') as handle:
                handle.write(response.content)


def parse(root, limit=None):
    rows = read_csv_rows(os.path.join(root, 'fitzpatrick17k.csv'))
    rows = rows[:limit] if limit else rows
    records = []
    for row in rows:
        key = row.get('md5hash')
        if not key:
            continue
        records.append(LesionRecord(
            image_path=os.path.join(root, 'images', f"{key}.jpg"),
            source_dataset='fitzpatrick17k',
            raw_label=row.get('label') or 'unknown',
            group_id=key,                                  # no patient grouping -> per-image
            fitzpatrick=to_int(row.get('fitzpatrick_scale') or row.get('fitzpatrick')),
        ))
    return records
