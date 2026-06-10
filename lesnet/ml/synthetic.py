"""Synthetic dataset generator for smoke runs and tests (no real data required)."""
import os

import numpy as np
from PIL import Image

from lesnet.ml.datasets import LesionRecord

_LABELS = ['nevus', 'melanoma', 'actinic keratosis', 'basal cell carcinoma']


def make_synthetic_records(root, per_class=12, size=(64, 64), seed=0):
    """Write synthetic JPEGs and return matching LesionRecords across triage groups."""
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(seed)
    records = []
    index = 0
    for label in _LABELS:
        # Give each class a different colour bias so a tiny model can learn *something*.
        bias = rng.uniform(0.2, 0.8, size=3)
        for position in range(per_class):
            pixels = (np.clip(rng.normal(bias, 0.15, size=(size[1], size[0], 3)), 0, 1) * 255).astype('uint8')
            path = os.path.join(root, f"img_{index}.jpg")
            Image.fromarray(pixels).save(path)
            records.append(LesionRecord(
                image_path=path,
                source_dataset='synthetic',
                raw_label=label,
                group_id=f"P{index // 2}",
                fitzpatrick=int(rng.integers(1, 7)),
                anatomical_site='torso',
                age=float(rng.integers(20, 80)),
                sex='male' if position % 2 else 'female',
            ))
            index += 1
    return records
