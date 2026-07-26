# LesNet JEPA World Model — design

> Authoritative design for the self-supervised **world model** of skin lesions. This is a
> **standalone subsystem** (`lesnet/jepa/`), framework-isolated from the TF/Keras triage
> stack. It learns lesion representations from the *entire unlabeled* dermoscopy archive,
> then hands a frozen encoder to the triage head. Nothing here imports TensorFlow.

## Why JEPA

The triage model learns only from the balanced, *labeled* manifest — a small, curated slice
of the ISIC archive. A Joint-Embedding Predictive Architecture (I-JEPA, Assran et al. 2023)
learns by predicting the **latent representations** of masked regions of an image from a
visible context region — no labels, no pixel reconstruction, no hand-crafted augmentations.
That lets the encoder learn the structure of dermoscopic images at the scale of the whole
archive, giving downstream triage a far richer prior ("a world model of lesions") than
supervised training on labels alone can provide.

## Architecture (I-JEPA)

Three modules, all plain PyTorch ViTs:

- **Context encoder** `f_θ` — encodes a visible *context* block of patch tokens.
- **Target encoder** `f_θ̄` — an **EMA** copy of the context encoder (no gradient). Encodes
  the full image; its outputs at the masked target positions are the prediction targets.
- **Predictor** `g_φ` — from the context-token representations + learned mask tokens (carrying
  the target positional embeddings), predicts the target representations. Loss is **smooth-L1
  / L2 in latent space**, target side stop-gradient.

Only the **context encoder** is retained for deployment; the target encoder and predictor are
pretraining-only and discarded.

### Model-size family — ViT-S/16 default

Standard ViT ladder, exposed under friendly aliases (`tiny`/`small`/`medium`/`large`/`xlarge`)
and the `vit_*` keys. int8 ONNX size + **measured** onnxruntime peak RSS at 224px (`intra_op=1`,
sequential; Windows build host — measure on the Linux target before you trust the deploy number):

| Family | ViT key | Params | int8 ONNX | peak RSS | Fits 512 MB? |
|--------|---------|--------|-----------|----------|--------------|
| Tiny    | vit_tiny  | 5.5M  | 6.2 MB   | 77 MB   | ✅ (edge default) |
| **Small** | **vit_small** | **21.6M** | **22.4 MB** | **101 MB** | ✅ **default — best quality/size** |
| Medium  | vit_base  | 85.6M | 86.9 MB  | 174 MB  | ✅ |
| Large   | vit_large | 303M  | 305 MB   | 415 MB  | ✅ (measured) |
| XLarge  | vit_huge  | 631M  | ~630 MB  | > 512 MB | ❌ (weights alone exceed budget) |

`scripts/train_jepa.sh` trains the whole family and records each variant's measured fit in
`family_summary.tsv`; it fails only if no tiny/small edge variant fits. int8 parity vs torch stays
small across the family (0.05 → 0.16, gated < 0.5).

Modern touches: `F.scaled_dot_product_attention` (flash) for training with an eager fallback for
portable ONNX export, fixed 2-D sin-cos position embeddings (regenerate analytically at any
resolution — no interpolation), multi-block masking (1 large context block, 4 small target
blocks), optional LayerScale + stochastic depth for ViT-B.

### 512 MB deployment — measured, not asserted

The real constraint is inference-time **RSS**, not weight size: the PyTorch CPU runtime alone is
~300–500 MB RSS before a single activation. So the deployable artifact is **ONNX** served by
**onnxruntime** (`CPUExecutionProvider`), *not* torch:

1. `export_onnx` — `torch.onnx.export` of the fp32 context encoder (eager attention, opset ≥ 17,
   dynamic batch), verified against torch outputs (`onnx_parity_max_abs_diff`).
2. `quantize_onnx_int8` — onnxruntime dynamic int8.
3. `measure_peak_rss` — spawns a fresh process that loads the onnxruntime session and runs one
   forward, sampling peak RSS. **`fits_budget` is set from this measurement**, and both weight
   size and peak RSS are reported. The torch `.pt` checkpoint is kept only for research/fine-tuning.

Expectation: onnxruntime int8 **ViT-Ti comfortably fits 512 MB** (the guaranteed-deployable
variant); **ViT-S is measured** by `scripts/train_jepa.sh`, which exits non-zero if it doesn't fit
and tells you to fall back to `ENCODER=vit_tiny`.

### Shared pretrain/transfer input contract

The encoder must see the **same input distribution** as the TF triage stack, or transfer degrades
and known ISIC shortcuts (device colour cast, hair) get baked in. So JEPA reuses the TF-free
medical transforms from `lesnet.ml.preprocessing` — **DullRazor hair removal + Shades-of-Gray
colour constancy** — and scales to **[0,1]** (matching triage `scale_unit`), not ImageNet
normalisation. The sin-cos encoder rebuilds at the triage resolution on transfer.

## Data flow

```
unlabeled dermoscopy archive
      │  (torch Dataset; labels ignored)
      ▼
MultiBlockMaskCollator ──► (images, enc_masks, pred_masks)
      │
      ▼
context encoder  ──►  predictor  ──►  predicted target reps
target encoder (EMA) ─────────────►  actual target reps        ── smooth-L1 loss
```

Pretrain → **linear/attentive probe** on the labeled manifest to track representation quality
→ export context encoder → int8 quantize → (later) fine-tune the triage head on top.

### Full-archive pretraining

SSL wants every image, not the balanced labelled subset. `--hdf5` reads an archive bundle (one
HDF5 dataset per image id holding encoded JPEG bytes) so the **whole ISIC archive (~500k images)**
is a single file rather than half a million small reads. `--exclude-manifest` keeps the labelled
manifest's held-out **test** ids out of pretraining, and the loader *refuses to start* if the
exclusion list matches no keys — a silently-empty exclusion would make every downstream number
transductive.

### Making the compute budget count

Four levers, each addressing a way a fixed-budget SSL run quietly wastes itself:

| Lever | Flag | Why |
|---|---|---|
| Schedule sized to the budget | `--epochs` / `--warmup-epochs` | A 300-epoch cosine with 40 warmup epochs that only runs 21 epochs **never leaves warmup** — LR climbs the whole time and nothing ever anneals. `train()` now rejects `warmup >= epochs/2`. |
| Select on the downstream probe | `--probe-every`, `--select-on probe` | The SSL loss *rises* during warmup, so loss-based selection saves the epoch-0 encoder. Probe ROC-AUC is the metric that tracks what we actually ship. |
| Low-res first | `--resize-schedule 0:128,0.6:224` | 128px is 64 tokens vs 196 — roughly 3x cheaper per image. Position grids are analytic sin-cos, so retargeting resolution is exact and the weights carry over untouched. |
| Real effective batch | `--grad-accum` | I-JEPA's LR is tuned for batch 256; a 32-image batch on one GPU without accumulation trains at a different operating point than the LR implies. |

## Module layout (`lesnet/jepa/`)

```
config.py             JEPAConfig — all hyperparameters (pure dataclass, no torch)
vision_transformer.py ViT encoder + predictor, sin-cos pos-embed, SDPA/eager, LayerScale/DropPath
masking.py            MultiBlockMaskCollator (spawn-safe, seeded) + apply_masks / repeat_interleave
preprocessing.py      shared input contract: DullRazor + Shades-of-Gray + [0,1] scaling
data.py               unlabeled Dataset + loader (glob / manifest / full-archive HDF5; test excluded)
modeling.py           IJEPA = context + EMA target + predictor + collapse stats
engine.py             pretraining loop: WD param groups, cosine LR/WD/EMA, AMP, TensorBoard, resume
probe.py              leakage-free clinical linear probe (grouped split, sensitivity, fairness)
export.py             ONNX + onnxruntime int8 + measured-RSS 512 MB gate + parity check
```

`commands/run_pretrain_jepa.py` drives pretraining (`--smoke` for a CPU end-to-end check).
`scripts/train_jepa.sh` is the fully-automatic pipeline: download ISIC → pretrain (TensorBoard) →
probe → export, gated on the measured 512 MB budget.

## Conventions

- **PyTorch, isolated.** This subsystem is the one deliberate exception to the repo's
  single-framework (TF/Keras 3) rule. It never imports TF; the triage stack never imports it.
  Installed via the optional `pip install -e ".[jepa]"` extra so TF-only users don't pull torch.
- Hyperparameters live as dataclass fields in `lesnet/jepa/config.py` (mirrors `PipelineConfig`).
- Same medical disclaimer applies: research/triage tool, not a diagnostic device.

## References

- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive
  Architecture (I-JEPA)*, CVPR 2023 — `facebookresearch/ijepa`.
