import csv

import numpy as np

from lesnet.ml import preprocessing
from lesnet.ml.datasets import (
    DatasetConfig,
    assign_splits,
    build_manifest,
    load_manifest,
    save_manifest,
)


def _write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_isic(root, patients=10, per_patient=4):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for patient in range(patients):
        for image in range(per_patient):
            rows.append({
                'isic_id': f"ISIC_{patient}_{image}",
                'patient_id': f"P{patient}",
                'diagnosis': 'melanoma' if patient % 2 else 'nevus',
                'age_approx': '55', 'sex': 'male', 'anatom_site_general': 'torso',
            })
    _write_csv(root / 'metadata.csv',
               ['isic_id', 'patient_id', 'diagnosis', 'age_approx', 'sex', 'anatom_site_general'], rows)


def _make_pad_ufes(root, count=12):
    (root / 'images').mkdir(parents=True, exist_ok=True)
    rows = [{
        'img_id': f"PAT_{i}.png", 'patient_id': f"PAD{i // 2}", 'lesion_id': f"L{i}",
        'diagnostic': 'BCC' if i % 2 else 'NEV', 'fitspatrick': str((i % 6) + 1),
        'region': 'arm', 'age': '60', 'gender': 'female',
    } for i in range(count)]
    _write_csv(root / 'metadata.csv',
               ['img_id', 'patient_id', 'lesion_id', 'diagnostic', 'fitspatrick', 'region', 'age', 'gender'], rows)


def test_isic_manifest_and_grouped_split(tmp_path):
    isic_root = tmp_path / 'isic'
    _make_isic(isic_root)
    config = DatasetConfig(mode='sample', sample_size_per_dataset=100, datasets=('isic',),
                           isic_root=str(isic_root), test_size=0.2, val_size=0.2, seed=3)
    records = assign_splits(build_manifest(config), config)

    assert len(records) == 40
    assert all(record.split in {'train', 'val', 'test'} for record in records)
    # No patient appears in more than one split.
    split_of_group = {}
    for record in records:
        split_of_group.setdefault(record.group_id, record.split)
        assert split_of_group[record.group_id] == record.split


def test_pad_ufes_parses_fitzpatrick_and_sample_limit(tmp_path):
    pad_root = tmp_path / 'pad'
    _make_pad_ufes(pad_root, count=12)
    config = DatasetConfig(mode='sample', sample_size_per_dataset=5, datasets=('pad_ufes_20',),
                           pad_ufes_root=str(pad_root))
    records = build_manifest(config)
    assert len(records) == 5                         # sample limit applied
    assert all(record.source_dataset == 'pad_ufes_20' for record in records)
    assert all(record.fitzpatrick is not None for record in records)


def test_multi_dataset_manifest_roundtrip(tmp_path):
    isic_root, pad_root = tmp_path / 'isic', tmp_path / 'pad'
    _make_isic(isic_root, patients=6, per_patient=3)
    _make_pad_ufes(pad_root, count=8)
    config = DatasetConfig(mode='full', datasets=('isic', 'pad_ufes_20'),
                           isic_root=str(isic_root), pad_ufes_root=str(pad_root))
    records = assign_splits(build_manifest(config), config)
    assert {r.source_dataset for r in records} == {'isic', 'pad_ufes_20'}

    out = tmp_path / 'manifest.csv'
    save_manifest(records, out)
    reloaded = load_manifest(out)
    assert len(reloaded) == len(records)
    assert reloaded[0].image_path == records[0].image_path


def test_preprocessing_pipeline_outputs_unit_scaled_array():
    image = (np.random.default_rng(0).random((40, 30, 3)) * 255).astype('uint8')
    pipeline = preprocessing.PreprocessingPipeline(image_size=(24, 24), remove_hair=False)
    output = pipeline(image)
    assert output.shape == (24, 24, 3)
    assert output.dtype == np.float32
    assert output.min() >= 0.0 and output.max() <= 1.0
