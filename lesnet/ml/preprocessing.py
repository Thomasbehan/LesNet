"""Dermoscopy preprocessing (paper §5.3).

Shades-of-Gray colour constancy (Finlayson & Trezzi, 2004) neutralises device colour
casts, which are a known shortcut confounder in ISIC. Operates on an HxWx3 array.
"""
import numpy as np


def shades_of_gray(image, power=6, eps=1e-6):
    """Apply Shades-of-Gray colour constancy; returns an array of the same dtype range."""
    image = np.asarray(image, dtype=float)
    per_channel = np.power(np.mean(np.power(image, power), axis=(0, 1)), 1.0 / power)
    illuminant = per_channel / (np.linalg.norm(per_channel) + eps)
    correction = illuminant * np.sqrt(3.0)
    corrected = image / (correction + eps)
    return np.clip(corrected, 0.0, float(np.max(image)) if image.size else 0.0)
