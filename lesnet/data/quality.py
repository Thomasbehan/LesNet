"""Image quality gate + perceptual-hash near-duplicate removal (stage 1).

Bad/duplicate data is the fastest way to inflate validation scores: a corrupt image trains
on noise, and a near-duplicate that lands in both train and test leaks the answer. We drop
unreadable/too-small images and collapse perceptual near-duplicates (keeping one
representative) BEFORE splitting, so reported metrics are trustworthy.

The hash function is injectable so the logic is testable without real images.
"""
import numpy as np
from PIL import Image


def image_short_side(path):
    """Shorter side length in pixels, or None if the image can't be read."""
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        return min(width, height)
    except Exception:  # noqa: BLE001 - any decode/IO failure means "unusable image"
        return None


def passes_quality(path, min_pixels=64):
    short_side = image_short_side(path)
    return short_side is not None and short_side >= min_pixels


def average_hash(path, hash_size=8):
    """64-bit perceptual average-hash, or None if unreadable."""
    try:
        with Image.open(path) as image:
            grayscale = image.convert('L').resize((hash_size, hash_size), Image.BILINEAR)
        pixels = np.asarray(grayscale, dtype=np.float64)
    except Exception:  # noqa: BLE001 - unreadable image has no hash
        return None
    bits = pixels > pixels.mean()
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left, right):
    return bin(left ^ right).count('1')


def filter_quality(records, min_pixels=64):
    """Split records into (kept, dropped) by the readability/size gate."""
    kept, dropped = [], []
    for record in records:
        (kept if passes_quality(record.image_path, min_pixels) else dropped).append(record)
    return kept, dropped


def dedupe(records, max_distance=4, hash_fn=average_hash):
    """Greedily drop perceptual near-duplicates, keeping the first representative.

    Returns (kept, dropped). Records whose image can't be hashed are kept (the quality gate
    is responsible for unreadable images, not this step).
    """
    kept, dropped = [], []
    representatives = []  # list of kept hashes
    for record in records:
        digest = hash_fn(record.image_path)
        if digest is None:
            kept.append(record)
            continue
        if any(hamming_distance(digest, seen) <= max_distance for seen in representatives):
            dropped.append(record)
        else:
            representatives.append(digest)
            kept.append(record)
    return kept, dropped
