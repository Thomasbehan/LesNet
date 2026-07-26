"""Vision Transformer encoder + predictor for I-JEPA (see docs/jepa-world-model.md).

Dependency-light implementation of the joint-embedding predictive architecture described in
Assran et al., CVPR 2023, built on the standard ViT of Dosovitskiy et al. (2020). The context
and target encoders share `VisionTransformer` (the target encoder is a gradient-free EMA copy);
`VisionTransformerPredictor` maps context-token representations to the latent representations of
the masked target blocks. Only the context encoder is kept for downstream triage.

Uses fixed 2-D sin-cos position embeddings and `F.scaled_dot_product_attention` (flash) — needs
torch >= 2.4.
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from lesnet.jepa.masking import apply_masks, repeat_interleave_batch


def build_2d_sincos_pos_embed(embed_dim, grid_size):
    """Fixed 2-D sin-cos position embedding, shape (grid_size**2, embed_dim)."""
    if embed_dim % 4 != 0:
        raise ValueError('embed_dim must be divisible by 4 for 2-D sin-cos position embedding.')
    coords = torch.arange(grid_size, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
    quarter = embed_dim // 4
    omega = 1.0 / (10000 ** (torch.arange(quarter, dtype=torch.float32) / quarter))
    out_x = torch.einsum('n,d->nd', grid_x.reshape(-1), omega)
    out_y = torch.einsum('n,d->nd', grid_y.reshape(-1), omega)
    return torch.cat(
        [torch.sin(out_x), torch.cos(out_x), torch.sin(out_y), torch.cos(out_y)], dim=1
    )


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim {dim} not divisible by num_heads {num_heads}.')
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # SDPA (flash) for training speed; eager path is used for portable ONNX export.
        self.use_sdpa = True

    def forward(self, x):
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        dropout_p = self.attn_drop if self.training else 0.0
        if self.use_sdpa:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            if dropout_p:
                attn = F.dropout(attn, p=dropout_p)
            x = attn @ v
        x = x.transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj_drop(self.proj(x))


def set_attention_impl(module, use_sdpa):
    """Toggle SDPA vs eager attention across a model (eager exports to ONNX portably)."""
    for submodule in module.modules():
        if isinstance(submodule, Attention):
            submodule.use_sdpa = use_sdpa


class DropPath(nn.Module):
    """Stochastic depth per sample (0.0 => identity)."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x / keep * mask


class LayerScale(nn.Module):
    def __init__(self, dim, init_value):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, layerscale_init=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.ls1 = LayerScale(dim, layerscale_init) if layerscale_init > 0 else nn.Identity()
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)
        self.ls2 = LayerScale(dim, layerscale_init) if layerscale_init > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class VisionTransformer(nn.Module):
    """I-JEPA context/target encoder (no class token; mean-pool patch tokens downstream)."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.0, layerscale_init=0.0):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        pos_embed = build_2d_sincos_pos_embed(embed_dim, self.grid_size).unsqueeze(0)
        # PERSISTENT on purpose. A sin-cos grid is regenerable, but a *learned* one loaded from a
        # pretrained checkpoint is not — and a non-persistent buffer is silently dropped from
        # state_dict(). That cost us a full debugging cycle: DINOv2's learned position embedding
        # vanished on save, the reload regenerated sin-cos, and the exported encoder produced
        # embeddings with cosine 0.085 to the model we thought we had exported.
        self.register_buffer('pos_embed', pos_embed, persistent=True)
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()  # stochastic-depth schedule
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                  drop_path=dpr[i], layerscale_init=layerscale_init, norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.pos_embed_pretrained = False   # set by jepa.pretrained when a checkpoint supplies one
        self.apply(_init_weights)

    def forward(self, x, masks=None):
        """Encode image patches. If `masks` (list of (B, K) index tensors) is given, keep only
        those context tokens; the output batch is repeated once per mask."""
        x = self.patch_embed(x)
        if x.shape[1] != self.pos_embed.shape[1]:
            raise ValueError(
                f'Got {x.shape[1]} patch tokens but the position embedding has '
                f'{self.pos_embed.shape[1]}. Rebuild the encoder at this resolution '
                f'(the sin-cos grid regenerates analytically).'
            )
        x = x + self.pos_embed
        if masks is not None:
            x = apply_masks(x, masks)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def set_image_size(module, image_size, patch_size):
    """Retarget every ViT/predictor in `module` to a new input resolution.

    Position embeddings are fixed 2-D sin-cos buffers regenerated analytically, and the patch-embed
    conv is resolution-agnostic — so this is exact, not an interpolation, and the weights carry
    over untouched. That is what makes the low-res-first resolution schedule safe.
    """
    grid = image_size // patch_size
    for sub in module.modules():
        if isinstance(sub, VisionTransformer):
            device = sub.pos_embed.device
            if getattr(sub, 'pos_embed_pretrained', False):
                # A loaded checkpoint's LEARNED position embedding must be resampled, not replaced
                # by a sin-cos grid — the pretrained blocks were trained against it.
                from lesnet.jepa.pretrained import _resize_pos_embed
                resized = _resize_pos_embed(sub.pos_embed.float().cpu(), grid).to(device)
                sub.register_buffer('pos_embed', resized, persistent=True)
            else:
                sub.register_buffer(
                    'pos_embed',
                    build_2d_sincos_pos_embed(sub.embed_dim, grid).unsqueeze(0).to(device),
                    persistent=True)
            sub.grid_size, sub.num_patches = grid, grid ** 2
        elif isinstance(sub, VisionTransformerPredictor):
            device = sub.pos_embed.device
            dim = sub.pos_embed.shape[-1]
            sub.register_buffer(
                'pos_embed', build_2d_sincos_pos_embed(dim, grid).unsqueeze(0).to(device),
                persistent=False)
    return grid ** 2


class VisionTransformerPredictor(nn.Module):
    """Predict target-block representations from context tokens + positional mask tokens."""

    def __init__(self, num_patches, encoder_embed_dim, predictor_embed_dim=192, depth=6,
                 num_heads=6, mlp_ratio=4.0, qkv_bias=True):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.predictor_embed = nn.Linear(encoder_embed_dim, predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        grid_size = int(round(math.sqrt(num_patches)))
        pos_embed = build_2d_sincos_pos_embed(predictor_embed_dim, grid_size).unsqueeze(0)
        self.register_buffer('pos_embed', pos_embed, persistent=False)
        self.blocks = nn.ModuleList([
            Block(predictor_embed_dim, num_heads, mlp_ratio, qkv_bias, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, encoder_embed_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(_init_weights)

    def forward(self, context_tokens, masks_enc, masks_pred):
        """context_tokens: (B * n_enc, K_enc, encoder_dim) from the context encoder.
        Returns predicted target reps (B * n_enc * n_pred, K_pred, encoder_dim)."""
        if not isinstance(masks_enc, list):
            masks_enc = [masks_enc]
        if not isinstance(masks_pred, list):
            masks_pred = [masks_pred]
        batch_size = len(context_tokens) // len(masks_enc)

        x = self.predictor_embed(context_tokens)
        x = x + apply_masks(self.pos_embed.repeat(batch_size, 1, 1), masks_enc)
        _, n_ctx, _ = x.shape

        # mask tokens carry the target positional embeddings
        pos_pred = apply_masks(self.pos_embed.repeat(batch_size, 1, 1), masks_pred)
        pos_pred = repeat_interleave_batch(pos_pred, batch_size, repeat=len(masks_enc))
        mask_tokens = self.mask_token.expand(pos_pred.size(0), pos_pred.size(1), -1) + pos_pred

        x = x.repeat(len(masks_pred), 1, 1)
        x = torch.cat([x, mask_tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.predictor_proj(x[:, n_ctx:])


# Standard ViT size ladder (params @ /16, 224px): Ti ~5.7M, S ~22M, B ~86M, L ~304M, H ~632M.
_ENCODER_TABLE = {
    'vit_tiny': dict(embed_dim=192, depth=12, num_heads=3),
    'vit_small': dict(embed_dim=384, depth=12, num_heads=6),
    'vit_base': dict(embed_dim=768, depth=12, num_heads=12),
    'vit_large': dict(embed_dim=1024, depth=24, num_heads=16),
    'vit_huge': dict(embed_dim=1280, depth=32, num_heads=16),
}

# Friendly family names -> ViT keys (Tiny / Small / Medium / Large / XLarge).
SIZE_ALIASES = {
    'tiny': 'vit_tiny', 'small': 'vit_small', 'medium': 'vit_base',
    'large': 'vit_large', 'xlarge': 'vit_huge',
}
ENCODER_CHOICES = sorted(_ENCODER_TABLE) + sorted(SIZE_ALIASES)


def resolve_encoder(name):
    """Map a friendly alias (tiny/small/medium/large/xlarge) to its vit_* key; pass through vit_*."""
    return SIZE_ALIASES.get(name, name)


def build_encoder(config):
    key = resolve_encoder(config.encoder)
    if key not in _ENCODER_TABLE:
        raise ValueError(
            f"Unknown encoder '{config.encoder}'. Known: {sorted(_ENCODER_TABLE)} or aliases "
            f"{sorted(SIZE_ALIASES)}."
        )
    return VisionTransformer(
        img_size=config.image_size, patch_size=config.patch_size, in_chans=config.in_chans,
        drop_path_rate=config.drop_path_rate, layerscale_init=config.layerscale_init,
        **_ENCODER_TABLE[key],
    )


def build_predictor(config, num_patches, encoder_embed_dim):
    return VisionTransformerPredictor(
        num_patches=num_patches, encoder_embed_dim=encoder_embed_dim,
        predictor_embed_dim=config.predictor_embed_dim, depth=config.predictor_depth,
        num_heads=config.predictor_num_heads,
    )
