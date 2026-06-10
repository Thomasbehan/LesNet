"""Central configuration for the triage training/inference pipeline (paper §4, §5)."""
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # image / data
    image_size: tuple = (384, 384)
    # model
    backbone: str = 'efficientnetv2s'
    pretrained: bool = True
    backbone_trainable_layers: int = 60      # unfreeze the last N backbone layers
    shared_units: int = 256
    dropout: float = 0.3
    aux_loss_weight: float = 0.3             # weight on the auxiliary fine-grained head
    # training
    batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 1e-4
    focal_gamma: float = 2.0
    malignant_cost: float = 4.0              # extra cost multiplier on the malignant class
    target_sensitivity: float = 0.97         # sensitivity-first operating point
    conformal_alpha: float = 0.1
    seed: int = 42
    # metric-gated training: keep training until every target is in the ideal range
    train_until_target: bool = False
    epochs_per_round: int = 5
    max_epochs: int = 200                     # hard safety budget for the gated loop
    target_specificity: float = 0.80
    max_ece: float = 0.05
    require_fairness_gate: bool = True
    tensorboard: bool = True
    cache_dataset: bool = False               # disk-cache preprocessed images across epochs
    remove_hair: bool = True                  # DullRazor hair removal (costly at large scale)
    # io
    artifacts_dir: str = 'artifacts'
    # smoke mode shrinks everything for a fast CPU end-to-end check
    smoke: bool = False
