"""Configuration for the data-sourcing pipeline (stage 1)."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SourcingConfig:
    # which sources to use + where their raw downloads live
    sources: tuple = ('isic',)              # any of: isic, pad_ufes_20, fitzpatrick17k, ddi
    roots: dict = field(default_factory=dict)  # {source_name: root_path}
    raw_dir: str = 'data/raw'               # where auto-downloads land
    dest: str = 'data/dataset'              # sorted-folder + manifest output

    # download scope
    sample_limit: Optional[int] = None      # cap rows per source (None = all)
    full_resolution: bool = True            # ISIC: full-res images, not thumbnails

    # quality + dedup
    min_image_pixels: int = 64              # reject images smaller than this on the short side
    dedupe: bool = True                     # collapse perceptual near-duplicates
    phash_distance: int = 4                 # Hamming distance under which two images are "the same"

    # balancing (decision-critical buckets)
    balance_ratio: float = 1.0              # benign : malignant target (1.0 = 1:1)
    per_diagnosis_cap_fraction: float = 0.6  # no single diagnosis may exceed this share of its bucket
    balance_buckets: tuple = ('benign', 'malignant')  # 'not_sure' kept as-is

    # leakage-safe splits
    test_size: float = 0.15
    val_size: float = 0.15
    seed: int = 42
