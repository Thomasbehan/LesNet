"""Unified multi-source dataset ingestion (paper §5.3).

Maps ISIC, PAD-UFES-20, Fitzpatrick17k, and DDI into one canonical manifest of
``LesionRecord`` rows, supports a fast ``sample`` mode and a ``full`` mode (so we can
iterate on a few hundred images and later flip to the whole download with one flag),
and assigns patient/lesion-grouped train/val/test splits with no leakage.

Loaders read each dataset's metadata CSV defensively; point the *_root paths at local
downloads. Images are not opened here — only paths and labels are collected.
"""
import csv
import os
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np

from lesnet.ml.splits import grouped_train_val_test

SAMPLE_MODE = 'sample'
FULL_MODE = 'full'


@dataclass
class DatasetConfig:
    mode: str = SAMPLE_MODE                 # 'sample' (fast iteration) or 'full'
    sample_size_per_dataset: int = 200
    datasets: tuple = ('isic',)             # any of: isic, pad_ufes_20, fitzpatrick17k, ddi
    isic_root: Optional[str] = None
    pad_ufes_root: Optional[str] = None
    fitzpatrick17k_root: Optional[str] = None
    ddi_root: Optional[str] = None
    image_size: tuple = (224, 224)
    test_size: float = 0.15
    val_size: float = 0.15
    seed: int = 42

    def sample_limit(self):
        return self.sample_size_per_dataset if self.mode == SAMPLE_MODE else None


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


MANIFEST_FIELDS = [field.name for field in fields(LesionRecord)]


def _read_csv(path):
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _ddi_skin_tone_to_fitzpatrick(skin_tone):
    """DDI encodes skin tone as 12/34/56 (Fitzpatrick pairs); map to a representative band."""
    return {'12': 1, '34': 3, '56': 5}.get(str(skin_tone).strip(), None)


def load_isic(root, sample_limit=None):
    rows = _read_csv(os.path.join(root, 'metadata.csv'))
    rows = rows[:sample_limit] if sample_limit else rows
    records = []
    for row in rows:
        isic_id = row.get('isic_id')
        if not isic_id:
            continue
        records.append(LesionRecord(
            image_path=os.path.join(root, f"{isic_id}.jpg"),
            source_dataset='isic',
            raw_label=row.get('diagnosis') or row.get('benign_malignant') or 'unknown',
            group_id=row.get('patient_id') or isic_id,
            anatomical_site=row.get('anatom_site_general') or None,
            age=_to_float(row.get('age_approx')),
            sex=row.get('sex') or None,
        ))
    return records


def load_pad_ufes(root, sample_limit=None):
    rows = _read_csv(os.path.join(root, 'metadata.csv'))
    rows = rows[:sample_limit] if sample_limit else rows
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
            fitzpatrick=_to_int(row.get('fitspatrick')),
            anatomical_site=row.get('region') or None,
            age=_to_float(row.get('age')),
            sex=row.get('gender') or None,
        ))
    return records


def load_fitzpatrick17k(root, sample_limit=None):
    rows = _read_csv(os.path.join(root, 'fitzpatrick17k.csv'))
    rows = rows[:sample_limit] if sample_limit else rows
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
            fitzpatrick=_to_int(row.get('fitzpatrick_scale') or row.get('fitzpatrick')),
        ))
    return records


def load_ddi(root, sample_limit=None):
    rows = _read_csv(os.path.join(root, 'ddi_metadata.csv'))
    rows = rows[:sample_limit] if sample_limit else rows
    records = []
    for row in rows:
        image_file = row.get('DDI_file')
        if not image_file:
            continue
        records.append(LesionRecord(
            image_path=os.path.join(root, image_file),
            source_dataset='ddi',
            raw_label=row.get('disease') or 'unknown',
            group_id=image_file,
            fitzpatrick=_ddi_skin_tone_to_fitzpatrick(row.get('skin_tone')),
        ))
    return records


_LOADERS = {
    'isic': ('isic_root', load_isic),
    'pad_ufes_20': ('pad_ufes_root', load_pad_ufes),
    'fitzpatrick17k': ('fitzpatrick17k_root', load_fitzpatrick17k),
    'ddi': ('ddi_root', load_ddi),
}


def build_manifest(config):
    sample_limit = config.sample_limit()
    records = []
    for name in config.datasets:
        if name not in _LOADERS:
            raise ValueError(f"Unknown dataset '{name}'. Known: {sorted(_LOADERS)}")
        root_attribute, loader = _LOADERS[name]
        root = getattr(config, root_attribute)
        if not root:
            raise ValueError(f"Dataset '{name}' enabled but '{root_attribute}' is not set.")
        records.extend(loader(root, sample_limit))
    return records


def assign_splits(records, config):
    groups = np.array([record.group_id for record in records])
    train_index, val_index, test_index = grouped_train_val_test(
        groups, config.test_size, config.val_size, config.seed)
    for index in train_index:
        records[index].split = 'train'
    for index in val_index:
        records[index].split = 'val'
    for index in test_index:
        records[index].split = 'test'
    return records


def save_manifest(records, path):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({name: getattr(record, name) for name in MANIFEST_FIELDS})


def load_manifest(path):
    records = []
    for row in _read_csv(path):
        records.append(LesionRecord(
            image_path=row['image_path'],
            source_dataset=row['source_dataset'],
            raw_label=row['raw_label'],
            group_id=row['group_id'],
            fitzpatrick=_to_int(row.get('fitzpatrick')),
            anatomical_site=row.get('anatomical_site') or None,
            age=_to_float(row.get('age')),
            sex=row.get('sex') or None,
            split=row.get('split') or None,
        ))
    return records
