import json
import os

import numpy as np
from PIL import Image

from lesnet.data import manifest, pipeline
from lesnet.data.config import SourcingConfig
from lesnet.data.records import LesionRecord
from lesnet.data.sources.base import SourceSpec


def _write_image(path, seed):
    Image.fromarray((np.random.default_rng(seed).random((80, 80, 3)) * 255).astype('uint8')).save(path)


def _dataset(tmp_path):
    """Synthetic on-disk records: benign + malignant + not_sure + one unmappable."""
    records = []
    plan = [('nevus', 8), ('melanoma', 6), ('actinic keratosis', 2), ('totally-unmappable', 1)]
    index = 0
    for label, count in plan:
        for _ in range(count):
            path = tmp_path / f"img_{index}.jpg"
            _write_image(path, index)
            records.append(LesionRecord(str(path), 'isic', label, f"P{index}", fitzpatrick=index % 6 + 1))
            index += 1
    return records


# --- config ---

def test_sourcing_config_defaults_independent():
    one, two = SourcingConfig(), SourcingConfig()
    one.roots['isic'] = '/tmp/x'
    assert two.roots == {}                       # default_factory gives each instance its own dict
    assert one.balance_ratio == 1.0 and one.balance_buckets == ('benign', 'malignant')


# --- manifest ---

def test_assign_splits_empty_and_normal():
    assert manifest.assign_splits([]) == []
    records = [LesionRecord('x', 'isic', 'nevus', f"P{i}") for i in range(20)]
    manifest.assign_splits(records, test_size=0.2, val_size=0.2, seed=1)
    assert {r.split for r in records} == {'train', 'val', 'test'}


def test_write_manifest(tmp_path):
    records = [LesionRecord('x', 'isic', 'nevus', 'P1', split='train')]
    out = tmp_path / 'm.csv'
    manifest.write(records, str(out))
    assert out.exists()


# --- pipeline helpers ---

def test_distribution_and_build_report():
    records = [LesionRecord('x', 'isic', 'nevus', 'P1', triage_bucket='benign', diagnosis='nevus',
                            split='train', fitzpatrick=2)]
    assert pipeline._distribution(records, 'diagnosis') == {'nevus': 1}
    report = pipeline.build_report({'ingested': 1}, records)
    assert report['final_total'] == 1
    assert report['benign_to_malignant_ratio'] is None    # no malignant -> None branch


# --- pipeline.process (end to end, no network) ---

def test_process_end_to_end(tmp_path):
    config = SourcingConfig(dest=str(tmp_path / 'out'), per_diagnosis_cap_fraction=1.0, seed=1)
    records = _dataset(tmp_path)
    materialised, report = pipeline.process(records, config)

    assert report['dropped_unmappable'] == 1
    assert report['by_bucket']['malignant'] == 6
    assert report['by_bucket']['benign'] <= 6              # downsampled to 1:1
    assert report['by_bucket'].get('not_sure') == 2
    assert os.path.exists(os.path.join(config.dest, 'manifest.csv'))
    report_on_disk = json.load(open(os.path.join(config.dest, 'report.json'), encoding='utf-8'))
    assert report_on_disk['final_total'] == len(materialised)
    unmapped = open(os.path.join(config.dest, 'unmapped_diagnoses.txt'), encoding='utf-8').read()
    assert 'totally-unmappable' in unmapped
    assert {r.split for r in materialised} <= {'train', 'val', 'test'}


def test_process_without_dedupe(tmp_path):
    config = SourcingConfig(dest=str(tmp_path / 'out2'), dedupe=False, per_diagnosis_cap_fraction=1.0)
    _records, report = pipeline.process(_dataset(tmp_path), config)
    assert report['dropped_duplicate'] == 0


# --- pipeline.ingest / run (mocked sources) ---

def _fake_spec(calls, downloadable=True):
    def parse(root, limit):           # noqa: ARG001
        return [LesionRecord(os.path.join(root, 'a.jpg'), 'fake', 'nevus', 'P1')]

    def download(root, limit):        # noqa: ARG001
        os.makedirs(root, exist_ok=True)
        open(os.path.join(root, 'marker'), 'w').close()
        calls.append('download')

    return SourceSpec('fake', parse=parse, download=download if downloadable else None)


def test_ingest_downloads_when_absent(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, 'get_source', lambda name: _fake_spec(calls))
    config = SourcingConfig(sources=('fake',), raw_dir=str(tmp_path / 'raw'))
    records = pipeline.ingest(config)
    assert calls == ['download'] and len(records) == 1


def test_ingest_skips_download_when_present(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, 'get_source', lambda name: _fake_spec(calls))
    root = tmp_path / 'raw' / 'fake'
    root.mkdir(parents=True)
    (root / 'existing').write_text('x')
    config = SourcingConfig(sources=('fake',), raw_dir=str(tmp_path / 'raw'))
    pipeline.ingest(config)
    assert calls == []                            # present -> no download


def test_ingest_manual_source_without_downloader(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'get_source', lambda name: _fake_spec([], downloadable=False))
    config = SourcingConfig(sources=('fake',), raw_dir=str(tmp_path / 'raw'))
    assert len(pipeline.ingest(config)) == 1      # parse-only, no crash


def test_ingest_skips_unavailable_source(tmp_path, monkeypatch):
    def boom_parse(root, limit):      # noqa: ARG001
        raise FileNotFoundError('metadata not on disk')

    monkeypatch.setattr(pipeline, 'get_source',
                        lambda name: SourceSpec('fake', parse=boom_parse, download=None))
    config = SourcingConfig(sources=('fake',), raw_dir=str(tmp_path / 'raw'))
    assert pipeline.ingest(config) == []      # unavailable source skipped, not fatal


def test_run(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'get_source', lambda name: _fake_spec([]))
    config = SourcingConfig(sources=('fake',), raw_dir=str(tmp_path / 'raw'),
                            dest=str(tmp_path / 'out'), dedupe=False)
    # parse points at a non-existent image -> materialise drops it; run still completes.
    _records, report = pipeline.run(config)
    assert report['ingested'] == 1
