"""I-JEPA model wrapper: context encoder + EMA target encoder + predictor (design doc).

Ties the three sub-modules together and computes the latent predictive loss. The target
encoder is a gradient-free EMA copy of the context encoder; only the context encoder is kept
for downstream triage.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from lesnet.jepa.masking import apply_masks, repeat_interleave_batch
from lesnet.jepa.vision_transformer import build_encoder, build_predictor


@torch.no_grad()
def ema_update(target_module, online_module, momentum):
    """In-place EMA: target = momentum * target + (1 - momentum) * online."""
    for target_param, online_param in zip(target_module.parameters(), online_module.parameters()):
        target_param.mul_(momentum).add_(online_param.detach(), alpha=1.0 - momentum)


class IJEPA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.context_encoder = build_encoder(config)
        if getattr(config, 'pretrained', ''):
            from lesnet.jepa.pretrained import load_pretrained
            load_pretrained(self.context_encoder, config.pretrained)
        # deepcopy AFTER loading so the EMA target starts from the same pretrained weights
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.predictor = build_predictor(
            config, self.context_encoder.num_patches, self.context_encoder.embed_dim
        )
        self._loss = F.smooth_l1_loss if config.loss == 'smooth_l1' else F.mse_loss
        # Collapse stats for TensorBoard. Each one ends in .item(), which is a full GPU->CPU
        # sync *inside* forward — it stops the GPU running ahead of the CPU on every micro-batch.
        # The training loop only reads them on logging steps, so it toggles this off in between.
        self.last_stats = {}
        self.stats_enabled = True

    @torch.no_grad()
    def _targets(self, images, masks_enc, masks_pred):
        h = self.target_encoder(images)                  # (B, N, D), full image
        h = F.layer_norm(h, (h.size(-1),))               # normalise targets (I-JEPA)
        batch_size = len(h)
        h = apply_masks(h, masks_pred)                   # (n_pred * B, K, D)
        return repeat_interleave_batch(h, batch_size, repeat=len(masks_enc))

    def forward(self, images, masks_enc, masks_pred):
        targets = self._targets(images, masks_enc, masks_pred)
        context = self.context_encoder(images, masks_enc)          # (n_enc * B, K_enc, D)
        predictions = self.predictor(context, masks_enc, masks_pred)
        # Cross-sample std of mean-pooled embeddings: a standard I-JEPA collapse signal. Pooling
        # over tokens first isolates spread ACROSS samples (positional variance within a block
        # would otherwise set a non-zero floor that masks partial collapse); ->0 means degenerate.
        if self.stats_enabled:
            self.last_stats = {
                'target_std': targets.detach().mean(dim=1).std(dim=0).mean().item(),
                'pred_std': predictions.detach().mean(dim=1).std(dim=0).mean().item(),
            }
        return self._loss(predictions, targets)

    @torch.no_grad()
    def update_target(self, momentum):
        ema_update(self.target_encoder, self.context_encoder, momentum)
