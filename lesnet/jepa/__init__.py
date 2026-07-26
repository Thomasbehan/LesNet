"""LesNet JEPA — self-supervised world-model pretraining for skin lesions.

Standalone PyTorch subsystem (see docs/jepa-world-model.md). Framework-isolated from the
TF/Keras triage stack; installed via the optional ``pip install -e ".[jepa]"`` extra.

Importing this package requires torch; keep it out of the TF web/inference import path.
"""
from lesnet.jepa.config import JEPAConfig

__all__ = ['JEPAConfig']
