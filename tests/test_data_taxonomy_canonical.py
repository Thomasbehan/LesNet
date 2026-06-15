from lesnet.data.canonical import (
    canonical_diagnosis,
    canonical_or_slug,
    slugify,
    unmapped_diagnoses,
)
from lesnet.data.records import LesionRecord
from lesnet.data.taxonomy import (
    BENIGN,
    MALIGNANT,
    SUSPICIOUS,
    build_fine_vocabulary,
    fine_index,
    triage_index,
)


def test_triage_index_buckets():
    assert triage_index('Melanoma, NOS') == MALIGNANT
    assert triage_index('basal cell carcinoma') == MALIGNANT
    assert triage_index('actinic keratosis') == SUSPICIOUS
    assert triage_index('melanocytic nevus') == BENIGN
    assert triage_index('') is None
    assert triage_index(None) is None
    assert triage_index('unknown') is None
    assert triage_index('something unmappable') is None


def test_build_fine_vocabulary_and_fine_index():
    records = [LesionRecord('a', 'isic', 'Melanoma', 'P1'),
              LesionRecord('b', 'isic', 'nevus', 'P2'),
              LesionRecord('c', 'isic', 'unmappable-thing', 'P3')]
    vocabulary = build_fine_vocabulary(records)
    assert set(vocabulary) == {'melanoma', 'nevus'}            # unmappable excluded
    assert fine_index('NEVUS', vocabulary) == vocabulary['nevus']
    assert fine_index('absent', vocabulary) is None


def test_canonical_diagnosis_merges_variants():
    assert canonical_diagnosis('Melanoma, NOS') == 'melanoma'
    assert canonical_diagnosis('lentigo maligna melanoma') == 'melanoma'   # melanoma wins over lentigo
    assert canonical_diagnosis('BCC') == 'basal cell carcinoma'
    assert canonical_diagnosis('naevus') == 'nevus'
    assert canonical_diagnosis('seborrhoeic keratosis') == 'seborrheic keratosis'
    assert canonical_diagnosis('haemangioma') == 'vascular lesion'
    assert canonical_diagnosis('unknown') is None
    assert canonical_diagnosis(None) is None
    assert canonical_diagnosis('mystery lesion') is None


def test_slugify_and_canonical_or_slug():
    assert slugify('Some Weird / Name!!') == 'some_weird_name'
    assert slugify('') == 'unknown'
    assert canonical_or_slug('melanoma') == 'melanoma'
    assert canonical_or_slug('mystery lesion') == 'mystery_lesion'


def test_unmapped_diagnoses_report():
    labels = ['melanoma', 'mystery one', 'mystery one', 'unknown', None, 'another mystery']
    assert unmapped_diagnoses(labels) == ['another mystery', 'mystery one']
