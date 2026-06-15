"""Manifest assembly: grouped leakage-safe splits + CSV write (stage 1)."""
import numpy as np

from lesnet.data.records import save_manifest
from lesnet.data.splits import assert_no_group_leakage, grouped_train_val_test


def assign_splits(records, test_size=0.15, val_size=0.15, seed=42):
    """Assign train/val/test with no patient/lesion group shared across splits."""
    if not records:
        return records
    groups = np.array([record.group_id for record in records])
    train_index, val_index, test_index = grouped_train_val_test(groups, test_size, val_size, seed)
    assert_no_group_leakage(groups, train_index, val_index, test_index)
    for index in train_index:
        records[index].split = 'train'
    for index in val_index:
        records[index].split = 'val'
    for index in test_index:
        records[index].split = 'test'
    return records


def write(records, path):
    save_manifest(records, path)
