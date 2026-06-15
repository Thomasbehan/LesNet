"""The 4.5.0 model family (paper §5.1, §7).

One teacher (XL) distilled into three deployable students (S/M/L). S is int8-quantised for
the live demo under the strict 500 MB peak-inference budget; the larger variants stay
available for maximum accuracy. Bigger backbone + resolution = more accurate + heavier.

    name      backbone           input   role
    M4.5s     EfficientNetV2-S   320px   live demo (int8 TFLite, <500 MB), distilled
    M4.5m     EfficientNetV2-M   384px   balanced, distilled
    M4.5L     EfficientNetV2-L   448px   high accuracy, distilled
    M4.5XL    EfficientNetV2-L   512px   teacher (widest head, heaviest aug) — distillation source
"""
from dataclasses import dataclass, replace

from lesnet.ml.config import PipelineConfig


@dataclass(frozen=True)
class Variant:
    name: str
    backbone: str
    image_size: int
    shared_units: int
    quantize: bool
    role: str
    is_teacher: bool = False


VARIANTS = {
    'M4.5s': Variant('M4.5s', 'efficientnetv2s', 320, 256, True, 'live demo (int8 TFLite, <500 MB)'),
    'M4.5m': Variant('M4.5m', 'efficientnetv2m', 384, 384, False, 'balanced'),
    'M4.5L': Variant('M4.5L', 'efficientnetv2l', 448, 512, False, 'high accuracy'),
    'M4.5XL': Variant('M4.5XL', 'efficientnetv2l', 512, 768, False, 'teacher', is_teacher=True),
}

# Distillation target order: which students learn from the teacher.
STUDENTS = ('M4.5L', 'M4.5m', 'M4.5s')
TEACHER = 'M4.5XL'


def config_for(variant_name, base=None, **overrides):
    """Build a PipelineConfig for a named variant, on top of ``base`` (or defaults)."""
    variant = VARIANTS[variant_name]
    config = base or PipelineConfig()
    config = replace(
        config,
        backbone=variant.backbone,
        image_size=(variant.image_size, variant.image_size),
        shared_units=variant.shared_units,
    )
    return replace(config, **overrides) if overrides else config
