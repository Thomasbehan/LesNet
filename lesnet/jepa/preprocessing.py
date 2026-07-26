"""Shared pretrain/transfer input contract for the JEPA world model (see docs/jepa-world-model.md).

Reuses the TF-free medical transforms from lesnet.ml.preprocessing (DullRazor hair removal +
Shades-of-Gray colour constancy) so the SSL encoder sees the SAME input distribution as the TF
triage stack — otherwise the encoder can bake in device colour cast (a known ISIC shortcut) and
hair, and transfer degrades. Defaults to [0,1] scaling (matching the triage `scale_unit`), not
ImageNet normalisation.
"""
import numpy as np
from PIL import Image
from torchvision import transforms

from lesnet.ml.preprocessing import dullrazor_hair_removal, shades_of_gray

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_BICUBIC = transforms.InterpolationMode.BICUBIC


class MedicalPreprocess:
    """PIL -> PIL: optional DullRazor hair removal + Shades-of-Gray colour constancy."""

    def __init__(self, remove_hair=True, colour_constancy=True):
        self.remove_hair = remove_hair
        self.colour_constancy = colour_constancy

    def __call__(self, image):
        array = np.asarray(image.convert('RGB'))
        if self.remove_hair:
            array = np.asarray(dullrazor_hair_removal(array))
        if self.colour_constancy:
            array = shades_of_gray(array)
        return Image.fromarray(np.clip(array, 0, 255).astype('uint8'))


def _tail(config):
    """ToTensor already scales to [0,1]; append ImageNet norm only if requested."""
    steps = [transforms.ToTensor()]
    if config.normalize == 'imagenet':
        steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return steps


def build_transform(config, train=True, image_size=None):
    """Torchvision transform implementing the shared input contract.

    `image_size` overrides config.image_size so the resolution schedule can train low-res first.
    """
    size = image_size or config.image_size
    steps = [MedicalPreprocess(config.remove_hair, config.colour_constancy)]
    if train:
        steps.append(transforms.RandomResizedCrop(
            size, scale=(config.rrc_min_scale, 1.0), interpolation=_BICUBIC))
        steps.append(transforms.RandomHorizontalFlip())
    else:
        steps.append(transforms.Resize(int(size * 1.15), interpolation=_BICUBIC))
        steps.append(transforms.CenterCrop(size))
    steps.extend(_tail(config))
    return transforms.Compose(steps)
