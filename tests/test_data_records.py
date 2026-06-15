import csv

from lesnet.data.records import (
    LesionRecord,
    MANIFEST_FIELDS,
    load_manifest,
    read_csv_rows,
    save_manifest,
    to_float,
    to_int,
)


def test_to_float_and_to_int_edge_cases():
    assert to_float('3.5') == 3.5
    assert to_float(None) is None
    assert to_float('not-a-number') is None
    assert to_int('4') == 4
    assert to_int('4.9') == 4
    assert to_int(None) is None
    assert to_int('nope') is None


def test_manifest_fields_include_bucket_and_diagnosis():
    assert 'triage_bucket' in MANIFEST_FIELDS
    assert 'diagnosis' in MANIFEST_FIELDS


def test_save_and_load_manifest_roundtrip(tmp_path):
    records = [
        LesionRecord(image_path='a.jpg', source_dataset='isic', raw_label='melanoma',
                     group_id='P1', fitzpatrick=3, anatomical_site='torso', age=55.0,
                     sex='male', split='train', triage_bucket='malignant', diagnosis='melanoma'),
        LesionRecord(image_path='b.jpg', source_dataset='ddi', raw_label='nevus', group_id='P2'),
    ]
    path = tmp_path / 'manifest.csv'
    save_manifest(records, str(path))
    loaded = load_manifest(str(path))
    assert len(loaded) == 2
    assert loaded[0].triage_bucket == 'malignant'
    assert loaded[0].diagnosis == 'melanoma'
    assert loaded[0].fitzpatrick == 3
    assert loaded[0].age == 55.0
    assert loaded[1].fitzpatrick is None and loaded[1].split is None


def test_load_manifest_tolerates_missing_optional_columns(tmp_path):
    path = tmp_path / 'old_manifest.csv'
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['image_path', 'source_dataset', 'raw_label', 'group_id'])
        writer.writeheader()
        writer.writerow({'image_path': 'a.jpg', 'source_dataset': 'isic',
                         'raw_label': 'nevus', 'group_id': 'P1'})
    loaded = load_manifest(str(path))
    assert loaded[0].triage_bucket is None
    assert loaded[0].diagnosis is None
    assert read_csv_rows(str(path))[0]['raw_label'] == 'nevus'
