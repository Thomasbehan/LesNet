"""The canonical dataset record + manifest IO (paper §5.3).

``LesionRecord`` is the one row type every data source maps into. The manifest is a CSV of
these rows with grouped train/val/test splits already assigned; training/inference read it
back with ``load_manifest``. ``triage_bucket`` (benign/not_sure/malignant) and the canonical
``diagnosis`` are filled by the sorting stage and are optional so older manifests still load.
"""
import csv
from dataclasses import dataclass, fields
from typing import Optional


@dataclass
class LesionRecord:
    image_path: str
    source_dataset: str
    raw_label: str
    group_id: str                           # patient/lesion id — the grouped-split key
    fitzpatrick: Optional[int] = None
    anatomical_site: Optional[str] = None
    age: Optional[float] = None
    sex: Optional[str] = None
    split: Optional[str] = None
    triage_bucket: Optional[str] = None     # benign | not_sure | malignant (set by sorting)
    diagnosis: Optional[str] = None         # canonical diagnosis (set by sorting)


MANIFEST_FIELDS = [field.name for field in fields(LesionRecord)]


def read_csv_rows(path):
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def save_manifest(records, path):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({name: getattr(record, name) for name in MANIFEST_FIELDS})


def load_manifest(path):
    records = []
    for row in read_csv_rows(path):
        records.append(LesionRecord(
            image_path=row['image_path'],
            source_dataset=row['source_dataset'],
            raw_label=row['raw_label'],
            group_id=row['group_id'],
            fitzpatrick=to_int(row.get('fitzpatrick')),
            anatomical_site=row.get('anatomical_site') or None,
            age=to_float(row.get('age')),
            sex=row.get('sex') or None,
            split=row.get('split') or None,
            triage_bucket=row.get('triage_bucket') or None,
            diagnosis=row.get('diagnosis') or None,
        ))
    return records
