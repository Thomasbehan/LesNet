"""End-to-end smoke: synthetic data -> train -> calibrate -> save -> load -> infer.

Uses the tiny backbone, no pretrained weights, 64px, 1 epoch — runs on CPU in seconds.
Proves the full pipeline is wired; real training scales the same code on GPU + real data.
"""
import numpy as np

from lesnet.ml.config import PipelineConfig
from lesnet.ml.datasets import DatasetConfig, assign_splits
from lesnet.ml.inference import TriagePredictor
from lesnet.ml.synthetic import make_synthetic_records
from lesnet.ml.training import train


def _split(records):
    buckets = {'train': [], 'val': [], 'test': []}
    for record in records:
        buckets[record.split].append(record)
    return buckets


def test_train_and_infer_end_to_end(tmp_path):
    records = make_synthetic_records(str(tmp_path / 'images'), per_class=10, seed=1)
    records = assign_splits(records, DatasetConfig(test_size=0.2, val_size=0.2, seed=1))
    buckets = _split(records)

    config = PipelineConfig(
        image_size=(64, 64), backbone='tiny', pretrained=False, batch_size=8,
        epochs=1, artifacts_dir=str(tmp_path / 'artifacts'), smoke=True)

    model, bundle = train(config, buckets['train'], buckets['val'] or buckets['train'])
    assert bundle['n_fine'] >= 1
    assert 0.0 < bundle['calibration']['temperature'] < 100.0
    assert 'operating_threshold' in bundle['thresholds']

    predictor = TriagePredictor(str(tmp_path / 'artifacts'))
    query = buckets['test'][0] if buckets['test'] else buckets['train'][0]
    image = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype('uint8')
    result = predictor.predict(image, query)

    assert result['triage'] in {'reassure', 'refer', 'urgent', 'abstain'}
    if result['valid_image']:
        assert set(result['probabilities']) == {'benign', 'suspicious', 'malignant'}
        assert 0.0 <= result['p_malignant'] <= 1.0
