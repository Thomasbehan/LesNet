"""Patient/lesion-grouped dataset splitting (paper §5.3, §6.1).

A group (e.g. patient_id or lesion_id) must never appear in more than one split, or the
same lesion leaks across train/test and inflates the reported metrics.
"""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def grouped_train_val_test(groups, test_size=0.15, val_size=0.15, seed=42):
    """Split sample indices into train/val/test with no group shared across splits."""
    groups = np.asarray(groups)
    indices = np.arange(len(groups))

    test_splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_index, test_index = next(test_splitter.split(indices, groups=groups))

    relative_val_size = val_size / (1.0 - test_size)
    val_splitter = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=seed)
    train_relative, val_relative = next(
        val_splitter.split(train_val_index, groups=groups[train_val_index]))

    train_index = train_val_index[train_relative]
    val_index = train_val_index[val_relative]
    return train_index, val_index, test_index


def assert_no_group_leakage(groups, train_index, val_index, test_index):
    """Raise if any group appears in more than one split."""
    groups = np.asarray(groups)
    train_groups = set(groups[train_index])
    val_groups = set(groups[val_index])
    test_groups = set(groups[test_index])
    if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
        raise AssertionError("Group leakage detected across splits.")
