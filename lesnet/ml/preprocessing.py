"""Dermoscopy preprocessing (paper §5.3).

A composable pipeline of medically-motivated transforms: hair removal, Shades-of-Gray
colour constancy (Finlayson & Trezzi, 2004) to neutralise device colour casts (a known
ISIC shortcut), an optional lesion-segmentation hook, resize, and unit scaling. All
transforms operate on HxWx3 numpy arrays.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image


def shades_of_gray(image, power=6, eps=1e-6):
    """Apply Shades-of-Gray colour constancy; returns an array in the input value range."""
    image = np.asarray(image, dtype=float)
    per_channel = np.power(np.mean(np.power(image, power), axis=(0, 1)), 1.0 / power)
    illuminant = per_channel / (np.linalg.norm(per_channel) + eps)
    correction = illuminant * np.sqrt(3.0)
    corrected = image / (correction + eps)
    return np.clip(corrected, 0.0, float(np.max(image)) if image.size else 0.0)


def dullrazor_hair_removal(image, kernel_size=9, inpaint_radius=3, threshold=10):
    """Remove dark hair via blackhat morphology + inpainting (DullRazor).

    Requires OpenCV; if unavailable this is a no-op so the pipeline still runs.
    """
    try:
        import cv2
    except Exception:  # noqa: BLE001 - optional dependency
        return np.asarray(image)
    array = np.asarray(image).astype('uint8')
    grayscale = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(array, hair_mask, inpaint_radius, cv2.INPAINT_TELEA)


def segment_lesion(image, segmenter=None):
    """Optional lesion-segmentation hook; default is identity until a segmenter is wired."""
    if segmenter is None:
        return np.asarray(image)
    return segmenter(image)


def resize_image(image, size):
    array = np.asarray(image)
    pillow_image = Image.fromarray(array.astype('uint8'))
    resized = pillow_image.resize((size[1], size[0]), Image.BILINEAR)
    return np.asarray(resized)


def scale_unit(image):
    return np.asarray(image, dtype=np.float32) / 255.0


@dataclass
class PreprocessingPipeline:
    image_size: tuple = (224, 224)
    remove_hair: bool = True
    apply_colour_constancy: bool = True
    segmenter: object = None

    def __call__(self, image):
        """uint8 HxWx3 image -> float32 [0,1] array at image_size."""
        array = np.asarray(image)
        if self.remove_hair:
            array = dullrazor_hair_removal(array)
        array = segment_lesion(array, self.segmenter)
        array = resize_image(array, self.image_size)
        if self.apply_colour_constancy:
            array = shades_of_gray(array)
        return scale_unit(array)
