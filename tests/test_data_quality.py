import numpy as np
from PIL import Image

from lesnet.data import quality
from lesnet.data.records import LesionRecord


def _write_image(path, width=80, height=80):
    pixels = (np.random.default_rng(abs(hash(str(path))) % 1000).random((height, width, 3)) * 255).astype('uint8')
    Image.fromarray(pixels).save(path)


def test_image_short_side_and_quality_gate(tmp_path):
    good = tmp_path / 'good.jpg'
    _write_image(good, width=100, height=80)
    small = tmp_path / 'small.jpg'
    _write_image(small, width=10, height=10)
    corrupt = tmp_path / 'corrupt.jpg'
    corrupt.write_text('definitely not an image')

    assert quality.image_short_side(str(good)) == 80
    assert quality.image_short_side(str(tmp_path / 'missing.jpg')) is None
    assert quality.image_short_side(str(corrupt)) is None
    assert quality.passes_quality(str(good), min_pixels=64) is True
    assert quality.passes_quality(str(small), min_pixels=64) is False


def test_average_hash_and_hamming(tmp_path):
    image = tmp_path / 'a.jpg'
    _write_image(image)
    digest = quality.average_hash(str(image))
    assert isinstance(digest, int)
    assert quality.average_hash(str(tmp_path / 'missing.jpg')) is None
    assert quality.hamming_distance(0b1010, 0b1001) == 2
    assert quality.hamming_distance(5, 5) == 0


def test_filter_quality_splits_records(tmp_path):
    good = tmp_path / 'good.jpg'
    _write_image(good, 80, 80)
    small = tmp_path / 'small.jpg'
    _write_image(small, 10, 10)
    records = [
        LesionRecord(str(good), 'isic', 'nevus', 'P1'),
        LesionRecord(str(small), 'isic', 'nevus', 'P2'),
        LesionRecord(str(tmp_path / 'missing.jpg'), 'isic', 'nevus', 'P3'),
    ]
    kept, dropped = quality.filter_quality(records, min_pixels=64)
    assert [r.group_id for r in kept] == ['P1']
    assert {r.group_id for r in dropped} == {'P2', 'P3'}


def test_dedupe_collapses_near_duplicates_keeps_unhashable():
    records = [LesionRecord(f"{i}.jpg", 'isic', 'nevus', f"P{i}") for i in range(4)]
    fake = {'0.jpg': 0, '1.jpg': 1, '2.jpg': 1000, '3.jpg': None}  # 1 is near-dup of 0; 3 unhashable

    kept, dropped = quality.dedupe(records, max_distance=4, hash_fn=lambda path: fake[path])
    kept_ids = {r.group_id for r in kept}
    assert kept_ids == {'P0', 'P2', 'P3'}      # P1 dropped (near-dup), P3 kept (unhashable)
    assert [r.group_id for r in dropped] == ['P1']
