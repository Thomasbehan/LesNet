"""PAD-UFES-20 source (CC BY 4.0, Mendeley). Smartphone clinical images with Fitzpatrick."""
import io
import os
import zipfile

import requests

from lesnet.data.records import LesionRecord, read_csv_rows, to_float, to_int

MENDELEY_ZIP_URL = (
    "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip"
)


def download(root, limit=None, session=None):  # noqa: ARG001 - limit unused (small dataset)
    """Best-effort fetch + extract of the PAD-UFES-20 archive into ``root``."""
    os.makedirs(root, exist_ok=True)
    session = session or requests
    response = session.get(MENDELEY_ZIP_URL, timeout=600)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(root)


def parse(root, limit=None):
    rows = read_csv_rows(os.path.join(root, 'metadata.csv'))
    rows = rows[:limit] if limit else rows
    records = []
    for row in rows:
        image_id = row.get('img_id')
        if not image_id:
            continue
        records.append(LesionRecord(
            image_path=os.path.join(root, 'images', image_id),
            source_dataset='pad_ufes_20',
            raw_label=row.get('diagnostic') or 'unknown',
            group_id=row.get('patient_id') or row.get('lesion_id') or image_id,
            fitzpatrick=to_int(row.get('fitspatrick')),
            anatomical_site=row.get('region') or None,
            age=to_float(row.get('age')),
            sex=row.get('gender') or None,
        ))
    return records
