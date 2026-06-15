"""LesNet data subsystem (4.5.0).

Single-responsibility modules that source raw dermatology data, canonicalise diagnosis
names, gate quality + remove near-duplicates, sort into clinical buckets
(benign / not_sure / malignant, per-diagnosis subfolders), balance to a fair benign:malignant
ratio, and emit a leakage-safe manifest. Pure-logic + IO — no TensorFlow.
"""
