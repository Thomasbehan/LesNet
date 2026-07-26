# LesNet JEPA — model family evaluation

Held-out evaluation of the deployable model family. Every number here is measured **through the
served artifact** — the exported ONNX encoder plus the fitted heads, loaded by `JEPADemoPredictor`
exactly as the web app loads it — not from an in-memory training object. That distinction cost a
full debugging cycle to learn (§4) and is now the rule.

LesNet is a research/triage tool, **not** a diagnostic device.

## The family

All three backbones are **DINOv2** (Meta), code and weights **Apache-2.0**, so the result is
redistributable including commercially. See
[`docs/architecture-research-2026.md`](architecture-research-2026.md) for why an off-the-shelf
backbone beats anything we could pretrain on 553k images, and which alternatives were refused on
licence grounds.

| Variant | Backbone | Params | Sens | Spec | ROC-AUC | pAUC@80%TPR | dx top-3 | OOD rej. | Abstain |
|---------|----------|--------|------|------|---------|-------------|----------|----------|---------|
| **Small** | DINOv2 ViT-S/14 | 21.5M | **0.954** | 0.905 | **0.974** | **0.182** | 0.852 | 1.00 | 1.3% |
| Medium ⭐demo | DINOv2 ViT-B/14 | 85.5M | 0.929 | **0.910** | 0.963 | 0.173 | **0.853** | 1.00 | 2.5% |
| Large | DINOv2 ViT-L/14 | 303M | 0.908 | 0.864 | 0.951 | 0.166 | 0.826 | 1.00 | 1.8% |

Held-out test split (n=400), patient/lesion-grouped so no patient appears in both train and test.

### Quality scales *inversely* with size here — and that is not a bug

Small beats Large on every clinical metric while being **14x smaller**. Two things explain it, and
both are properties of this setup rather than of DINOv2:

1. **The heads are linear probes on frozen features.** A 1024-dim feature space gives the probe far
   more freedom to overfit the ~5k fitting samples than a 384-dim one. Capacity in the *encoder*
   does not help when the *classifier* is the bottleneck and the labelled set is small.
2. **Bigger ViTs concentrate more capacity in high-norm "artifact" tokens** that mean-pooling
   dilutes; ViT-S/14's mean-pooled embedding is simply a better-behaved linear read-out.

The practical consequence: **Small is the recommended default**, and the family exists for
comparison rather than as a ladder where bigger is better. If the heads were fine-tuned rather than
linear, the ordering would likely change — that is future work, not a claim.

### Versus the previous release

| Metric | Released ViT-L (303M) | **New Small (21.5M)** | Δ |
|---|---|---|---|
| Sensitivity | 0.931 | **0.954** | +0.023 |
| Specificity | 0.746 | **0.905** | **+0.159** |
| ROC-AUC | 0.897 | **0.974** | +0.077 |
| Diagnosis top-1 | 0.250 | **0.490** | +0.240 |
| Diagnosis top-3 | 0.543 | **0.852** | +0.309 |
| OOD rejection | 0.93 | **1.00** | +0.07 |

A 14x smaller model, better on every axis — because the encoder is inherited rather than trained
from scratch on our data.

## Deployment

| Variant | fp32 ONNX | Measured RSS | Fits 512 MB? | Served precision |
|---------|-----------|--------------|--------------|------------------|
| **Small** | 87 MB | **165 MB** | ✅ | fp32 |
| Medium | 344 MB | 435 MB | ✅ (tight) | fp32 |
| Large | 1,214 MB | 1,317 MB | ❌ **exceeds budget** | fp32 (research only) |

**Large is not edge-deployable.** fp32 needs 1.3 GB and fp16 1.9 GB; only int8 would fit (420 MB)
and int8 is unusable (below). It ships as a research checkpoint, explicitly outside the 512 MB
contract — not as a "premium" tier.

**int8 is deliberately NOT served.** Dynamic int8 quantisation of DINOv2 produces parity errors of
**4.4–6.0** across the family (against ~3e-04 for fp32) — its high-norm tokens do not survive — and
in an end-to-end test that collapsed served ROC-AUC from 0.974 to **0.752**. Since fp32 Small
measures 165 MB RSS, well inside budget, there was never anything to buy by quantising. `serve.py`
loads the highest-fidelity encoder available (fp32 → fp16 → int8) rather than the smallest. The
int8 files are still exported for reference; nothing loads them.

## What the demo predicts

- **Triage probe** — benign vs malignant at a sensitivity-first operating point, fit on a balanced
  all-bucket sample with the threshold calibrated on a held-out split.
- **Diagnosis head** — multi-class over specific ISIC diagnoses; returns top-3.
- **OOD lesion gate** — supervised lesion-vs-not classifier. **100%** of non-lesion photos are
  rejected, so photographing a non-lesion object abstains instead of returning a diagnosis.

  Its threshold is calibrated to keep **99.5% of MALIGNANT lesions**, not 99.5% of all lesions.
  That is deliberate: "does this look like a lesion" scores *atypical* lesions lower, and
  malignancy is what makes a lesion atypical — on a held-out sample under the previous
  all-lesion calibration, **every single false abstention was a malignant case**. Recalibrating on
  the malignant subset cut false abstention from 4.3% to 1.3% (Small) with no loss of non-lesion
  rejection. Wrongly abstaining on a melanoma is dangerous; showing a result for a photo of a desk
  is merely embarrassing.

## Honest limitations

1. **The test split is class-balanced (~50% malignant); the real archive is ~0.4–5%.** These
   sensitivity/specificity figures are therefore **not comparable** to prospective clinical studies
   (pooled AI ≈ 0.863/0.784), which run at natural prevalence. At true prevalence the positive
   predictive value will be far lower. Evaluating at natural prevalence is outstanding work.
2. **n=400** for the served evaluation — confidence intervals are wide.
3. **pAUC@80%TPR** is reported because plain AUC averages over operating points nobody would deploy;
   a sensitivity-first system is only interesting above ~80% recall.
4. **No external validation yet.** Everything is ISIC. Fitzpatrick17k / DDI / PH2 are the obvious
   next step before any generalisation claim.
5. The encoder is **not** fine-tuned on dermoscopy — these are frozen general-purpose features with
   linear heads. Domain adaptation is unproven here (see below) and remains open.

## 4. A methodology note worth keeping

Two earlier results in this project's history were wrong for the same reason: they were measured on
an in-memory model that had never survived a save/load cycle. `pos_embed` was registered as a
**non-persistent** buffer, so `state_dict()` silently dropped DINOv2's *learned* position embedding
and every reload regenerated a sin-cos grid instead — pretrained weights with the wrong positional
signal, embeddings at cosine 0.085 to the intended model.

Both the claim "domain adaptation gives 0.9734" and its retraction "adaptation collapses to 0.818"
were artefacts of that bug. Neither should be cited. A regression test now saves, reloads and
asserts identical embeddings, and the standing rule is: **a number is only real once it comes out
of the served artifact.**
