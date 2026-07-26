"""Multi-block masking for the JEPA world model (see docs/jepa-world-model.md).

Independent implementation of the multi-block masking *strategy* described in Assran et al.,
"Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture", CVPR 2023.
Per batch it samples one large *context* block and several small *target* blocks; the encoder sees
only context patches and the predictor predicts the target patches. Masks are returned as lists of
(B, K) index tensors — one per context/target block — ready for `apply_masks`.

LICENCE NOTE: the authors' reference repository (facebookresearch/ijepa) is CC-BY-NC 4.0, which
would be incompatible with this MPL-2.0 project's commercial use. The published *method* is not
copyrightable and is implemented here from the paper's description; no code is copied from that
repository. Keep it that way — see docs/architecture-research-2026.md.
"""
import math

import torch


def apply_masks(x, masks):
    """Gather the tokens selected by each mask, concatenated along the batch dim.

    x:     (B, N, D) patch tokens.
    masks: list of (B, K) LongTensors of patch indices to keep.
    returns (B * len(masks), K, D).
    """
    gathered = []
    for mask in masks:
        index = mask.unsqueeze(-1).expand(-1, -1, x.size(-1))
        gathered.append(torch.gather(x, dim=1, index=index))
    return torch.cat(gathered, dim=0)


def repeat_interleave_batch(x, batch_size, repeat):
    """Repeat each per-image block `repeat` times, keeping images grouped (I-JEPA helper)."""
    n = len(x) // batch_size
    return torch.cat([
        torch.cat([x[i * batch_size:(i + 1) * batch_size] for _ in range(repeat)], dim=0)
        for i in range(n)
    ], dim=0)


class MultiBlockMaskCollator:
    """Collate a batch of images into (images, enc_masks, pred_masks).

    Mirrors the I-JEPA collator: sample block sizes once per batch (shared), then per-image
    sample `num_pred_masks` target blocks and `num_enc_masks` context blocks (context blocks
    avoid the target regions unless `allow_overlap`). Index tensors are truncated to the batch
    minimum so they collate into rectangular tensors.
    """

    def __init__(self, config, image_size=None):
        self.patch_size = config.patch_size
        grid = (image_size or config.image_size) // config.patch_size
        self.height = grid
        self.width = grid
        # min_keep is expressed for the configured resolution; under the low-res-first schedule the
        # grid shrinks (224px -> 196 tokens, 128px -> 64) while the mask *scales* stay fractional,
        # so an absolute floor would make every target block unsatisfiable. Scale it with the grid.
        reference = max(config.image_size // config.patch_size, 1)
        self.enc_mask_scale = config.enc_mask_scale
        self.pred_mask_scale = config.pred_mask_scale
        self.aspect_ratio = config.aspect_ratio
        self.nenc = config.num_enc_masks
        self.npred = config.num_pred_masks
        self.min_keep = max(round(config.min_keep * grid ** 2 / reference ** 2), 1)
        self.allow_overlap = config.allow_overlap
        self.size_multiple = max(getattr(config, 'mask_size_multiple', 1), 1)
        self.base_seed = config.seed
        self._calls = 0  # per-worker call counter; combined with worker id for a unique seed

    def _next_seed(self):
        """Deterministic, spawn-safe per-call seed (no shared multiprocessing state)."""
        info = torch.utils.data.get_worker_info()
        worker = info.id if info is not None else 0
        self._calls += 1
        return (self.base_seed + worker * 100003 + self._calls) % (2 ** 31)

    def _sample_block_size(self, generator, scale, aspect_ratio_range):
        rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = aspect_ratio_range
        aspect = min_ar + rand * (max_ar - min_ar)
        h = int(round(math.sqrt(max_keep * aspect)))
        w = int(round(math.sqrt(max_keep / aspect)))
        h = min(h, self.height - 1)
        w = min(w, self.width - 1)
        return max(h, 1), max(w, 1)

    def _sample_block_mask(self, block_size, generator, acceptable_regions=None):
        h, w = block_size

        def constrain(mask, tries):
            for k in range(max(len(acceptable_regions) - tries, 0)):
                mask *= acceptable_regions[k]

        tries, timeout, reset, attempts, max_attempts = 0, 20, 20, 0, 200
        while True:
            top = int(torch.randint(0, self.height - h + 1, (1,), generator=generator).item())
            left = int(torch.randint(0, self.width - w + 1, (1,), generator=generator).item())
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top + h, left:left + w] = 1
            if acceptable_regions is not None:
                constrain(mask, tries)
            indices = torch.nonzero(mask.flatten()).squeeze(-1)
            if len(indices) > self.min_keep:
                break
            attempts += 1
            if attempts >= max_attempts:  # a valid block is impossible for this config
                raise ValueError(
                    f'Could not sample a {block_size} block with > {self.min_keep} patches on a '
                    f'{self.height}x{self.width} grid. Lower min_keep or raise image_size / mask scale.'
                )
            timeout -= 1
            if timeout == 0:          # relax the overlap constraint and retry
                tries += 1
                timeout = reset
        complement = torch.ones((self.height, self.width), dtype=torch.int32)
        complement[top:top + h, left:left + w] = 0
        return indices, complement

    def __call__(self, batch):
        collated = torch.utils.data.default_collate(batch)
        generator = torch.Generator().manual_seed(self._next_seed())
        pred_size = self._sample_block_size(generator, self.pred_mask_scale, self.aspect_ratio)
        enc_size = self._sample_block_size(generator, self.enc_mask_scale, (1.0, 1.0))

        batch_size = len(batch)
        masks_pred, masks_enc = [], []
        min_keep_pred = min_keep_enc = self.height * self.width
        for _ in range(batch_size):
            per_pred, complements = [], []
            for _ in range(self.npred):
                indices, complement = self._sample_block_mask(pred_size, generator)
                per_pred.append(indices)
                complements.append(complement)
                min_keep_pred = min(min_keep_pred, len(indices))
            masks_pred.append(per_pred)

            regions = None if self.allow_overlap else complements
            per_enc = []
            for _ in range(self.nenc):
                indices, _ = self._sample_block_mask(enc_size, generator, acceptable_regions=regions)
                per_enc.append(indices)
                min_keep_enc = min(min_keep_enc, len(indices))
            masks_enc.append(per_enc)

        # The context block is the big block MINUS the target blocks, so its size varies per batch
        # while the target blocks (all sampled at one `pred_size`) do not. A shape that changes
        # every step defeats kernel selection and the caching allocator: measured 20.5 -> 74.1 img/s
        # for ViT-L once the shape stopped moving. Quantising the truncation length down to a
        # multiple collapses ~15 distinct shapes into 2-3 while only ever dropping a few tokens,
        # which is the same kind of truncation the batch-minimum rule already performs.
        if self.size_multiple > 1:
            quantised = (min_keep_enc // self.size_multiple) * self.size_multiple
            # Only quantise when it costs little. On a small grid the context block is already tiny
            # (a 64-patch grid leaves ~6-13 context patches) and rounding down to a multiple of 16
            # would throw away nearly all of it — measured context collapsing to 3/64 patches.
            if quantised >= 0.75 * min_keep_enc:
                min_keep_enc = quantised
        masks_pred = [[m[:min_keep_pred] for m in per] for per in masks_pred]
        masks_enc = [[m[:min_keep_enc] for m in per] for per in masks_enc]
        # default_collate turns the (B, nblocks) nesting into nblocks tensors of (B, K)
        masks_pred = torch.utils.data.default_collate(masks_pred)
        masks_enc = torch.utils.data.default_collate(masks_enc)
        return collated, masks_enc, masks_pred
