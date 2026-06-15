"""Smoke tests for the 4.5.0 model family: variants, distillation, quantisation, TFLite.

Tiny backbone + synthetic data so the whole teacher->student->quantise->infer path runs on
CPU in seconds and proves it is wired; real training scales the same code on GPU.
"""
import numpy as np
import pytest

from lesnet.data.manifest import assign_splits
from lesnet.ml import quantize, variants
from lesnet.ml.config import PipelineConfig
from lesnet.ml.features import METADATA_DIM
from lesnet.ml.model import build_triage_model
from lesnet.ml.synthetic import make_synthetic_records
from lesnet.ml.training import distill, train


def _data(tmp_path):
    records = assign_splits(make_synthetic_records(str(tmp_path / 'img'), per_class=8, seed=1),
                            test_size=0.2, val_size=0.2, seed=1)
    buckets = {'train': [], 'val': [], 'test': []}
    for record in records:
        buckets[record.split].append(record)
    return buckets['train'], buckets['val'] or buckets['train']


def _smoke_config(tmp_path, name):
    return PipelineConfig(
        image_size=(64, 64), backbone='tiny', pretrained=False, epochs=1, batch_size=8,
        shared_units=32, artifacts_dir=str(tmp_path / name), smoke=True,
        use_ema=False, tensorboard=False, label_smoothing=0.0)


def test_variants_registry_and_config():
    config = variants.config_for('M4.5s')
    assert config.backbone == 'efficientnetv2s' and config.image_size == (320, 320)
    assert variants.VARIANTS['M4.5XL'].is_teacher
    assert variants.VARIANTS['M4.5s'].quantize
    assert variants.TEACHER == 'M4.5XL' and 'M4.5s' in variants.STUDENTS


def test_distill_then_quantize_and_tflite_infer(tmp_path):
    train_records, val_records = _data(tmp_path)
    teacher, _ = train(_smoke_config(tmp_path, 'teacher'), train_records, val_records)
    student, bundle = distill(_smoke_config(tmp_path, 'student'), teacher, train_records, val_records)

    assert 0.0 < bundle['calibration']['temperature'] < 100.0
    assert 'operating_threshold' in bundle['thresholds']

    path = quantize.export_tflite(student, str(tmp_path / 'student.tflite'), mode='float16')
    assert quantize.model_size_mb(path) > 0
    logits = quantize.tflite_triage_logits(
        path, np.zeros((1, 64, 64, 3), 'float32'), np.zeros((1, METADATA_DIM), 'float32'))
    assert logits.shape[-1] == 3
    assert quantize.peak_rss_mb() > 0


def test_export_tflite_rejects_bad_modes(tmp_path):
    model = build_triage_model(_smoke_config(tmp_path, 'm'), n_fine=2, metadata_dim=METADATA_DIM)
    with pytest.raises(ValueError):
        quantize.export_tflite(model, str(tmp_path / 'x.tflite'), mode='int8')   # no dataset
    with pytest.raises(ValueError):
        quantize.export_tflite(model, str(tmp_path / 'x.tflite'), mode='bogus')
