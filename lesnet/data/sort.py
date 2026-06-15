"""Sort records into the clinical folder taxonomy (stage 1).

benign / not_sure / malignant  ->  <canonical_diagnosis>/  ->  image files.
Bucket comes from the triage taxonomy; the diagnosis subfolder from curated canonicalisation.
Records that map to no triage bucket are dropped (and counted).
"""
import os
import shutil

from lesnet.data.canonical import canonical_or_slug
from lesnet.data.taxonomy import BENIGN, MALIGNANT, SUSPICIOUS, triage_index

BUCKET_BY_INDEX = {BENIGN: 'benign', SUSPICIOUS: 'not_sure', MALIGNANT: 'malignant'}
BUCKETS = ('benign', 'not_sure', 'malignant')


def bucket_for(raw_label):
    """Clinical bucket name for a raw label, or None if unmappable."""
    index = triage_index(raw_label)
    return BUCKET_BY_INDEX.get(index) if index is not None else None


def annotate(records):
    """Set triage_bucket + canonical diagnosis on each record; drop unmappable ones.

    Returns (kept, dropped).
    """
    kept, dropped = [], []
    for record in records:
        bucket = bucket_for(record.raw_label)
        if bucket is None:
            dropped.append(record)
            continue
        record.triage_bucket = bucket
        record.diagnosis = canonical_or_slug(record.raw_label)
        kept.append(record)
    return kept, dropped


def _dest_filename(record):
    """Source-prefixed filename so ids from different datasets never collide."""
    return f"{record.source_dataset}_{os.path.basename(record.image_path)}"


def materialise(records, dest, link=False):
    """Copy (or symlink) each record's image into dest/<bucket>/<diagnosis>/ and repoint it.

    Records whose source image is missing are skipped (and counted). Returns (materialised, missing).
    """
    materialised, missing = [], []
    for record in records:
        if not os.path.exists(record.image_path):
            missing.append(record)
            continue
        folder = os.path.join(dest, record.triage_bucket, record.diagnosis)
        os.makedirs(folder, exist_ok=True)
        target = os.path.join(folder, _dest_filename(record))
        if link:
            if not os.path.exists(target):
                os.symlink(os.path.abspath(record.image_path), target)
        else:
            shutil.copy2(record.image_path, target)
        record.image_path = target
        materialised.append(record)
    return materialised, missing
