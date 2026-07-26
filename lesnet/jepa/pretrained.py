"""Initialise the I-JEPA encoder from openly-licensed pretrained weights.

Rationale (docs/architecture-research-2026.md): a from-scratch SSL run on our ~553k images cannot
out-represent a model pretrained on hundreds of millions. DINOv2 contributes ~142M images of
pretraining for free, so we inherit it and spend our budget on *domain adaptation* instead — tens
of dollars rather than hundreds, for a better encoder.

**Licensing is a hard constraint here.** LesNet is MPL-2.0 and must stay usable commercially, so
only weights that are open source AND commercially licensed may be wired in:

  * DINOv2 (facebookresearch/dinov2) — code and weights Apache-2.0.       ALLOWED
  * timm ImageNet-21k ViTs — Apache-2.0.                                   ALLOWED
  * PanDerm / DermLIP — CC-BY-NC-ND (non-commercial AND no-derivatives).   REFUSED
  * DINOv3 — custom Meta licence, not OSI open source.                     REFUSED
  * I-JEPA released weights — CC-BY-NC.                                    REFUSED

`load_pretrained` refuses anything not on the allow-list rather than trusting the caller: a
licence breach here would propagate into every downstream redistribution of the model.
"""
import math
import re
import warnings

import torch

# name -> (timm model id, expected embed_dim, patch_size, licence). Apache-2.0 only, by policy.
ALLOWED = {
    'dinov2_vits14': ('vit_small_patch14_dinov2.lvd142m', 384, 14, 'Apache-2.0'),
    'dinov2_vitb14': ('vit_base_patch14_dinov2.lvd142m', 768, 14, 'Apache-2.0'),
    'dinov2_vitl14': ('vit_large_patch14_dinov2.lvd142m', 1024, 14, 'Apache-2.0'),
    'dinov2_vits14_reg': ('vit_small_patch14_reg4_dinov2.lvd142m', 384, 14, 'Apache-2.0'),
    'dinov2_vitb14_reg': ('vit_base_patch14_reg4_dinov2.lvd142m', 768, 14, 'Apache-2.0'),
    'dinov2_vitl14_reg': ('vit_large_patch14_reg4_dinov2.lvd142m', 1024, 14, 'Apache-2.0'),
}

# Weights we deliberately refuse, with the reason, so the error is educational rather than "unknown".
REFUSED = {
    'panderm': 'CC-BY-NC-ND 4.0 — non-commercial AND no-derivatives; fine-tuning it is a breach',
    'dermlip': 'CC-BY-NC-ND 4.0 — non-commercial AND no-derivatives',
    'dinov3': 'custom Meta licence — permits commercial use but is not OSI open source',
    'ijepa': 'CC-BY-NC — non-commercial',
    'celldino': 'FAIR Noncommercial Research License',
    'xraydino': 'FAIR Noncommercial Research License',
}


def _resize_pos_embed(pos_embed, target_grid):
    """Bicubically resample a (1, N, D) patch position embedding to target_grid**2 tokens."""
    tokens, dim = pos_embed.shape[1], pos_embed.shape[2]
    source = int(round(math.sqrt(tokens)))
    if source == target_grid:
        return pos_embed
    grid = pos_embed.reshape(1, source, source, dim).permute(0, 3, 1, 2)
    grid = torch.nn.functional.interpolate(grid, size=(target_grid, target_grid),
                                           mode='bicubic', align_corners=False)
    return grid.permute(0, 2, 3, 1).reshape(1, target_grid ** 2, dim)


def _strip_prefix_tokens(state, key, num_prefix):
    """Drop DINOv2's cls (and register) tokens from its position embedding."""
    pos = state[key]
    return pos[:, num_prefix:] if pos.shape[1] > num_prefix else pos


def load_pretrained(encoder, name, strict_dim=True):
    """Load `name` into our VisionTransformer in place. Returns a report dict.

    Our ViT deliberately mirrors timm's parameter naming (blocks.N.attn.qkv, ls1.gamma, ...), so
    the transformer blocks transfer verbatim. The two structural differences are handled here:
    DINOv2 carries a cls token (and optional registers) we do not use, and it learns its position
    embedding where we generate a fixed sin-cos grid — the learned one is kept and resampled to
    our grid, because the pretrained blocks were trained against it.
    """
    key = name.lower().replace('-', '_')
    for banned, reason in REFUSED.items():
        if banned in key.replace('_', ''):
            raise ValueError(
                f"Refusing to load '{name}': {reason}. LesNet is MPL-2.0 and must remain "
                f"commercially usable — see docs/architecture-research-2026.md. "
                f"Allowed: {sorted(ALLOWED)}")
    if key not in ALLOWED:
        raise ValueError(f"Unknown pretrained weights '{name}'. Allowed (open source + "
                         f"commercial): {sorted(ALLOWED)}")

    timm_id, embed_dim, patch_size, licence = ALLOWED[key]
    if strict_dim and encoder.embed_dim != embed_dim:
        raise ValueError(
            f"{name} is {embed_dim}-dim but the configured encoder is {encoder.embed_dim}-dim. "
            f"Pick the matching --encoder (vit_small/base/large) or pass strict_dim=False.")
    if encoder.patch_size != patch_size:
        raise ValueError(
            f"{name} uses patch {patch_size}; the encoder is built with patch "
            f"{encoder.patch_size}. Set --patch-size {patch_size} (at 224px that is a "
            f"{224 // patch_size}x{224 // patch_size} grid).")

    try:
        import timm
    except ImportError as error:                # timm is Apache-2.0 and only needed for this path
        raise ImportError("loading pretrained weights needs timm: pip install timm") from error

    source = timm.create_model(timm_id, pretrained=True, num_classes=0)
    state = source.state_dict()

    target = encoder.state_dict()
    mapped, skipped = {}, []
    for key_name, tensor in state.items():
        if key_name in ('cls_token', 'reg_token', 'mask_token'):
            skipped.append(key_name)
            continue
        if key_name == 'pos_embed':
            continue                            # handled separately below
        clean = re.sub(r'^backbone\.', '', key_name)
        if clean in target and target[clean].shape == tensor.shape:
            mapped[clean] = tensor
        else:
            skipped.append(key_name)

    missing = [k for k in target if k not in mapped]
    encoder.load_state_dict(mapped, strict=False)

    # position embedding: drop prefix tokens, resample to our grid, keep it as the encoder's buffer
    pos_used = False
    if 'pos_embed' in state:
        prefix = state['pos_embed'].shape[1] - (source.patch_embed.num_patches
                                                if hasattr(source, 'patch_embed') else 0)
        pos = _strip_prefix_tokens(state, 'pos_embed', max(prefix, 0))
        pos = _resize_pos_embed(pos.float(), encoder.grid_size)
        if pos.shape[1] == encoder.pos_embed.shape[1]:
            encoder.register_buffer('pos_embed', pos.to(encoder.pos_embed.device),
                                    persistent=True)   # must survive save/load — see ViT comment
            encoder.pos_embed_pretrained = True
            pos_used = True
        else:
            warnings.warn('pretrained position embedding did not match the grid; keeping sin-cos')

    report = {'name': name, 'timm_id': timm_id, 'licence': licence,
              'loaded_tensors': len(mapped), 'skipped': len(skipped),
              'missing_in_checkpoint': missing, 'pos_embed_from_checkpoint': pos_used}
    print(f"pretrained: {name} ({licence}) -> {len(mapped)} tensors loaded, "
          f"{len(missing)} left at init, pos_embed_pretrained={pos_used}", flush=True)
    return report
