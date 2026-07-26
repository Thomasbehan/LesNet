"""Central configuration for I-JEPA world-model pretraining (see docs/jepa-world-model.md).

Pure dataclass — no torch import, so it is safe to read from anywhere. Mirrors the style of
lesnet.ml.config.PipelineConfig.
"""
from dataclasses import dataclass


@dataclass
class JEPAConfig:
    # image / patching
    image_size: int = 224
    patch_size: int = 16
    in_chans: int = 3

    # preprocessing / transfer contract (must match the TF triage stack for encoder transfer:
    # DullRazor hair removal + Shades-of-Gray colour constancy + [0,1] scaling, NOT ImageNet norm)
    remove_hair: bool = True
    colour_constancy: bool = True
    normalize: str = 'unit'          # 'unit' -> [0,1] (matches triage) | 'imagenet'
    rrc_min_scale: float = 0.4       # RandomResizedCrop lower bound (keep the lesion in view)

    # encoder ('vit_tiny' | 'vit_small' | 'vit_base'); ViT-S/16 is the default sweet spot
    encoder: str = 'vit_small'
    # openly-licensed pretrained init, e.g. 'dinov2_vitb14' (Apache-2.0). Inheriting ~142M images
    # of pretraining beats any from-scratch recipe on our 553k. '' = random init.
    pretrained: str = ''

    # predictor (narrow ViT; discarded after pretraining, so size is unconstrained by deployment)
    predictor_embed_dim: int = 192
    predictor_depth: int = 6
    predictor_num_heads: int = 6

    # encoder regularisation (default off for tiny/small; useful for vit_base)
    drop_path_rate: float = 0.0
    layerscale_init: float = 0.0     # 0 disables LayerScale

    # multi-block masking (I-JEPA: 1 large context block, 4 small target blocks)
    enc_mask_scale: tuple = (0.85, 1.0)
    pred_mask_scale: tuple = (0.15, 0.2)
    aspect_ratio: tuple = (0.75, 1.5)
    num_enc_masks: int = 1
    num_pred_masks: int = 4
    min_keep: int = 10               # min patches kept per block (else resample)
    mask_size_multiple: int = 16     # quantise context-mask length so shapes repeat (1=off)
    allow_overlap: bool = False      # allow enc/pred blocks to overlap

    # target-encoder EMA schedule (momentum ramps start -> end over training)
    ema_momentum: tuple = (0.996, 1.0)

    # optimisation
    epochs: int = 300
    batch_size: int = 128
    warmup_epochs: int = 40
    start_lr: float = 2e-4
    lr: float = 1e-3
    final_lr: float = 1e-6
    weight_decay: float = 0.04
    final_weight_decay: float = 0.4
    grad_clip: float = 1.0
    mixed_precision: str = 'bf16'    # 'bf16' | 'fp16' | 'no'
    scale_lr: bool = True            # scale LR by batch_size / lr_reference_batch (linear rule)
    lr_reference_batch: int = 256
    grad_accum_steps: int = 1        # effective batch = batch_size * grad_accum_steps
    max_batch_size: int = 0          # cap on the low-res stage's scaled-up micro-batch (0=off)
    compile_model: bool = False      # torch.compile the encoders/predictor (slow first step)

    # resolution schedule: ((fraction_of_epochs, image_size), ...) — train cheap at low res, then
    # finish at full res. The sin-cos position grids regenerate analytically, so this is exact.
    resize_schedule: tuple = ()

    # training optimisation: `epochs` is the UPPER bound; early stopping ends it sooner on plateau
    early_stopping: bool = True
    early_stop_patience: int = 15    # epochs without loss improvement before stopping
    early_stop_min_delta: float = 1e-4
    max_train_seconds: int = 0       # 0 = unlimited; set on Kaggle to checkpoint before session kill

    # loss
    loss: str = 'smooth_l1'          # 'smooth_l1' | 'l2'

    # data / io
    data_dir: str = 'data/dataset'   # unlabeled dermoscopy root (labels ignored)
    hdf5_path: str = ''              # full-archive HDF5 bundle (id -> encoded JPEG); wins over data_dir
    exclude_manifest: str = ''       # manifest whose held-out split is kept OUT of HDF5 pretraining
    num_workers: int = 8
    artifacts_dir: str = 'artifacts/jepa'
    checkpoint_every: int = 10       # epochs
    tensorboard: bool = True         # write TensorBoard event files to <artifacts_dir>/tb
    log_every: int = 10              # steps between scalar logs / stdout lines
    seed: int = 42

    # linear probe (clinical read-out): sensitivity-first operating point on the held-out split
    probe_target_sensitivity: float = 0.95
    pretrain_splits: tuple = ('train', 'val')   # manifest splits used for SSL (excludes 'test')

    # in-training probe: SSL loss keeps falling after representation quality plateaus, so select
    # and early-stop on downstream probe ROC-AUC instead when a probe manifest is available
    probe_every: int = 0             # epochs between in-training probes (0 = off)
    probe_manifest: str = ''         # labelled manifest for the in-training probe
    probe_subset: int = 3000         # images per probe split (keeps the probe a few % of epoch cost)
    select_on: str = 'loss'          # 'loss' | 'probe' — checkpoint + early-stop criterion

    # smoke mode shrinks everything for a fast CPU end-to-end check
    smoke: bool = False

    def encoder_dims(self):
        """(embed_dim, depth, num_heads) for the configured encoder (accepts family aliases)."""
        aliases = {'tiny': 'vit_tiny', 'small': 'vit_small', 'medium': 'vit_base',
                   'large': 'vit_large', 'xlarge': 'vit_huge'}
        table = {
            'vit_tiny': (192, 12, 3),
            'vit_small': (384, 12, 6),
            'vit_base': (768, 12, 12),
            'vit_large': (1024, 24, 16),
            'vit_huge': (1280, 32, 16),
        }
        key = aliases.get(self.encoder, self.encoder)
        if key not in table:
            raise ValueError(f"Unknown encoder '{self.encoder}'. Known: {sorted(table)} or "
                             f"aliases {sorted(aliases)}.")
        return table[key]

    @classmethod
    def smoke_config(cls):
        """Tiny config for a fast CPU end-to-end check."""
        return cls(
            image_size=96, patch_size=16, encoder='vit_tiny',
            predictor_embed_dim=96, predictor_depth=2, predictor_num_heads=3,
            num_pred_masks=2, min_keep=2, epochs=1, batch_size=4, warmup_epochs=0,
            remove_hair=False, colour_constancy=False, scale_lr=False,
            checkpoint_every=0, log_every=1, num_workers=0, mixed_precision='no', smoke=True,
        )
