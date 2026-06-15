import os

import numpy as np
import pytest
from PIL import Image

from lesnet.data import balance, sort, splits
from lesnet.data.records import LesionRecord


def _record(group_id, raw_label='nevus', fitzpatrick=None, bucket=None, diagnosis=None, path='x.jpg'):
    return LesionRecord(image_path=path, source_dataset='isic', raw_label=raw_label,
                        group_id=group_id, fitzpatrick=fitzpatrick,
                        triage_bucket=bucket, diagnosis=diagnosis)


def _write_image(path):
    Image.fromarray((np.random.default_rng(0).random((40, 40, 3)) * 255).astype('uint8')).save(path)


# --- sort ---

def test_bucket_for():
    assert sort.bucket_for('melanoma') == 'malignant'
    assert sort.bucket_for('actinic keratosis') == 'not_sure'
    assert sort.bucket_for('nevus') == 'benign'
    assert sort.bucket_for('gibberish') is None


def test_annotate_sets_bucket_and_drops_unmappable():
    records = [_record('P1', 'melanoma'), _record('P2', 'naevus'), _record('P3', 'gibberish')]
    kept, dropped = sort.annotate(records)
    assert [r.triage_bucket for r in kept] == ['malignant', 'benign']
    assert kept[0].diagnosis == 'melanoma' and kept[1].diagnosis == 'nevus'
    assert [r.group_id for r in dropped] == ['P3']


def test_materialise_copy_link_and_missing(tmp_path):
    source = tmp_path / 'src.jpg'
    _write_image(source)
    present = _record('P1', bucket='malignant', diagnosis='melanoma', path=str(source))
    missing = _record('P2', bucket='benign', diagnosis='nevus', path=str(tmp_path / 'nope.jpg'))
    dest = tmp_path / 'out'

    materialised, missing_out = sort.materialise([present, missing], str(dest))
    assert len(materialised) == 1 and [r.group_id for r in missing_out] == ['P2']
    assert os.path.exists(present.image_path)
    assert present.image_path.endswith(os.path.join('malignant', 'melanoma', 'isic_src.jpg'))

    link_record = _record('P3', bucket='benign', diagnosis='nevus', path=str(source))
    sort.materialise([link_record], str(tmp_path / 'linked'), link=True)
    assert os.path.islink(link_record.image_path)


# --- balance ---

def test_bucket_counts():
    records = [_record('P1', bucket='benign'), _record('P2', bucket='benign'),
              _record('P3', bucket='malignant')]
    assert balance.bucket_counts(records) == {'benign': 2, 'malignant': 1}


def test_targets_branches():
    assert balance._targets({'benign': 100, 'malignant': 10}, 1.0) == {'benign': 10, 'malignant': 10}
    assert balance._targets({'benign': 5, 'malignant': 10}, 1.0) == {'benign': 5, 'malignant': 5}
    assert balance._targets({'benign': 0, 'malignant': 10}, 1.0) == {'benign': 0, 'malignant': 10}


def test_select_groups_is_group_safe_and_fairness_first():
    # 4 groups of 2; one (G3) has Fitzpatrick 6 -> retained first.
    records = []
    for group in range(4):
        for _image in range(2):
            records.append(_record(f"G{group}", fitzpatrick=6 if group == 3 else 1, bucket='benign'))
    selected = balance._select_groups(records, target=4, seed=1)
    assert len(selected) == 4                      # whole groups only (2 + 2)
    assert all(r.group_id == 'G3' for r in selected[:2]) or any(r.group_id == 'G3' for r in selected)
    assert any(r.group_id == 'G3' for r in selected)   # the dark-skin group is retained
    # under target -> everything kept
    assert len(balance._select_groups(records, target=99, seed=1)) == 8
    assert balance._select_groups(records, target=0, seed=1) == []


def _by_diagnosis(records):
    counts = {}
    for record in records:
        counts[record.diagnosis] = counts.get(record.diagnosis, 0) + 1
    return counts


def test_balance_bucket_single_diagnosis_reaches_target():
    # All one diagnosis: must still reach target (cap must not starve it).
    records = [_record(f"M{i}", diagnosis='melanoma', bucket='malignant') for i in range(8)]
    selected = balance._balance_bucket(records, target=6, cap_fraction=0.6, seed=1)
    assert len(selected) == 6


def test_balance_bucket_limits_dominance_when_diverse():
    records = [_record(f"A{i}", diagnosis='nevus', bucket='benign') for i in range(16)]
    records += [_record(f"B{i}", diagnosis='seborrheic keratosis', bucket='benign') for i in range(4)]
    selected = balance._balance_bucket(records, target=4, cap_fraction=0.5, seed=1)
    counts = _by_diagnosis(selected)
    assert counts['nevus'] <= 2 and counts['seborrheic keratosis'] <= 2   # neither dominates
    assert len(selected) == 4
    # under target -> all kept
    assert len(balance._balance_bucket(records, target=99, cap_fraction=0.5, seed=1)) == 20
    assert balance._balance_bucket(records, target=0, cap_fraction=0.5, seed=1) == []


def test_balance_bucket_trims_when_capped_diagnoses_overshoot():
    # 3 diagnoses x 3 records, cap=2 each -> 6 capped > target 4 -> trimmed back to 4.
    records = []
    for diagnosis in ('nevus', 'seborrheic keratosis', 'dermatofibroma'):
        for i in range(3):
            records.append(_record(f"{diagnosis}{i}", diagnosis=diagnosis, bucket='benign'))
    selected = balance._balance_bucket(records, target=4, cap_fraction=0.6, seed=1)
    assert len(selected) == 4


def test_assert_no_group_leakage_raises():
    groups = np.array(['P1', 'P1', 'P2', 'P3'])
    with pytest.raises(AssertionError):
        splits.assert_no_group_leakage(groups, [0], [1], [2])   # P1 in both train and val


def test_balance_ratio_cap_and_keeps_not_sure():
    records = []
    for i in range(16):
        records.append(_record(f"BN{i}", diagnosis='nevus', bucket='benign'))
    for i in range(4):
        records.append(_record(f"BS{i}", diagnosis='seborrheic keratosis', bucket='benign'))
    for i in range(4):
        records.append(_record(f"M{i}", diagnosis='melanoma', bucket='malignant'))
    for i in range(3):
        records.append(_record(f"NS{i}", diagnosis='actinic keratosis', bucket='not_sure'))

    balanced = balance.balance(records, ratio=1.0, per_diagnosis_cap_fraction=0.6,
                               buckets=('benign', 'malignant'), seed=1)
    counts = balance.bucket_counts(balanced)
    assert counts['malignant'] == 4
    assert counts['benign'] <= 4               # downsampled to the malignant count
    assert counts['not_sure'] == 3             # untouched
