# LesNet next architecture — research and proposal (July 2026)

Where the field is, where LesNet can credibly be state of the art, and the architecture that gets
it there. Companion to [`docs/jepa-world-model.md`](jepa-world-model.md) (the current SSL system)
and [`docs/model-redesign.md`](model-redesign.md) (the triage contract).

**Research/triage tool, not a diagnostic device.** Nothing below changes that.

---

## 1. What the evidence says

### 1.1 Pretraining scale decides representation quality, and we cannot win it

[PanDerm](https://www.nature.com/articles/s41591-025-03747-y) (Nature Medicine, 2025) is
self-supervised on **>2M dermatological images** across 4 modalities and 11 institutions, and is
state of the art on 28 benchmarks — often beating supervised baselines using **10% of the labels**.

We have 552,792 images and a budget of tens of dollars. A from-scratch SSL run on that data will
not out-represent a 2M-image model, however good our recipe is. This is the single most important
strategic fact, and it says: **inherit pretraining, do not recreate it.**

**We measured this directly.** Probe AUC on our held-out split (n=1,500/1,500, sensitivity-first
threshold set on train), comparing everything we have trained against an *untouched* DINOv2:

| Encoder | ISIC training | Params | Probe AUC | Sens | Spec |
|---|---|---|---|---|---|
| ViT-S, random init | none | 21M | 0.864 | 0.961 | 0.300 |
| ViT-B, random init | none | 86M | 0.871 | 0.959 | 0.300 |
| **Our released ViT-L** | full family run | 303M | 0.897 | 0.942 | 0.686 |
| Our ViT-S | 3 epochs from scratch | 21M | 0.928 | — | — |
| **DINOv2 ViT-S/14** | **none** | **21M** | **0.9633** | 0.902 | **0.921** |
| **DINOv2 ViT-B/14** | **none** | 86M | 0.9632 | 0.863 | 0.937 |

An off-the-shelf DINOv2 ViT-S/14, having never seen an ISIC image, beats every encoder this
project has trained: **+0.066 AUC over our ViT-L and +0.235 specificity at comparable
sensitivity, with 1/14 the parameters.** A from-scratch run at I-JEPA-parity compute (~$340,
§6) would have bought a worse model than a free download.

At 21M parameters ViT-S/14 is also ~22 MB in int8 — the 512 MB budget stops being a constraint
at all, where ViT-L needed 443 MB of it.

### 1.2 But the strongest dermatology model is unusable to us

PanDerm's weights are **CC-BY-NC-ND 4.0** — non-commercial *and* no-derivatives. Fine-tuning it is
a derivative work. For an MPL-2.0 project it is a **benchmark to beat, not a building block**.

Licence-clean options actually available:

| Model | Licence | Usable here? |
|---|---|---|
| **DINOv2** (ViT-S/B/L/g) | **Apache-2.0** | ✅ fine-tune and redistribute freely |
| DINOv3 | Custom Meta licence; commercial allowed, registration required, licence must travel with derivatives | ⚠️ workable, adds friction |
| timm ImageNet-21k ViTs | Apache-2.0 | ✅ |
| PanDerm / DermLIP | CC-BY-NC-ND | ❌ |
| I-JEPA released weights | CC-BY-NC | ❌ for commercial |

**DINOv2 is the backbone.** It contributes ~142M images of pretraining for free, under a licence
that survives redistribution.

### 1.3 There is a "granularity gap" — one head cannot serve both tasks

A [hierarchical benchmark of ten foundation models](https://arxiv.org/abs/2601.12382) (2026) on
DERM12345 (40 subclasses) found general **medical** models lead binary malignancy screening
(MedImageInsights, 97.5% weighted F1) while **dermatology-specific** models lead fine-grained
40-class subtyping. General screening ability and fine-grained discrimination are separable.

This validates LesNet's existing two-head split (triage + auxiliary diagnosis) and argues against
collapsing them — and it explains our own numbers, where triage works (0.93 sensitivity) while
top-1 diagnosis over ~40 classes sits at 0.25.

### 1.4 What actually won ISIC 2024 was *patient context*, not a better image encoder

The [ISIC 2024 challenge](https://www.kaggle.com/competitions/isic-2024-challenge) ran on the same
3D-TBP tiles that dominate our archive. Winning solutions
([example write-up](https://arxiv.org/pdf/2506.03420)) combined ViT/CNN image features with a
**GBDT over engineered metadata and patient-specific relational features**, scored on
**pAUC above 80% TPR** — a sensitivity-first metric, exactly LesNet's operating philosophy.

The relational features encode the clinical **"ugly duckling" sign**: a lesion is suspicious
because it looks unlike the patient's *other* lesions.
[UDTR](https://link.springer.com/chapter/10.1007/978-3-030-87234-2_17) (MICCAI 2021) learns this
end-to-end with a transformer over all lesions of one patient plus a two-branch patient/lesion
head; [tiered quadruplet networks](https://arxiv.org/pdf/2309.09689) pursue the same signal.

**LesNet currently discards this entirely** — every lesion is scored independently.

### 1.5 The bar to beat

Pooled across recent meta-analyses of AI melanoma detection:

| System | Sensitivity | Specificity |
|---|---|---|
| Generalist clinicians | 0.646 | 0.728 |
| Dermatologists (prospective) | 0.786 | 0.752 |
| AI alone (pooled) | 0.863 | 0.784 |
| AI-assisted dermatologists | 0.919 | 0.837 |
| DermaSensor (FDA-cleared device) | 0.955 | 0.207–0.325 |
| **LesNet JEPA demo (current)** | **0.931** | **0.746** |

Our number is *not* comparable to those studies and must not be presented as if it were: it is
n=250 on a **class-balanced** curated split (~50% malignant), whereas the real archive is 0.43%
malignant among multi-lesion patients. Fixing that comparability is part of the work (§4).

---

## 2. Where LesNet can actually be state of the art

We cannot out-pretrain PanDerm. We can own an axis nobody has combined into one deployable system:

> **Patient-set-conditioned, calibrated, abstaining triage that runs in 512 MB — under a licence
> anyone can actually use.**

Each ingredient exists somewhere; none of them ship together:

- PanDerm has the representations but is NC-ND and single-image.
- ISIC 2024 winners have patient context but are giant non-deployable ensembles with no
  calibration, abstention or OOD handling.
- Clinical devices hit high sensitivity at 20–30% specificity, referring nearly everyone.
- Nobody targets a 512 MB edge budget with a measured RSS gate.

**And our data uniquely supports the patient-context axis:** 3,082 patients hold ≥10 lesions each,
covering **455,688 images — 82% of the archive** (mean 46 lesions/patient, max 9,184). This is the
regime where the ugly-duckling signal is strongest, and it is free signal we currently throw away.

---

## 3. Proposed architecture — patient-set triage

```
                 ┌─────────── per lesion ───────────┐
  image ──▶ DINOv2 ViT-B/14 (Apache-2.0, frozen-ish) ──▶ I-JEPA domain adaptation on 553k ISIC
                                │                              (our existing lesnet/jepa/ stack)
                                ▼
                        lesion embedding e_i  ────────────────┐
  age/sex/site/Fitzpatrick ──▶ metadata tokens ───────────────┤
                                                              ▼
                          ╔═══════════ PATIENT-SET TRANSFORMER ═══════════╗
                          ║  self-attention over {e_1..e_N} of ONE patient ║
                          ║  permutation-invariant; N = 1 .. hundreds      ║
                          ║  learned NO-CONTEXT token for the N=1 web case ║
                          ╚════════════════════════════════════════════════╝
                                    │                        │
                       lesion-relative score            patient-level risk
                                    │
                    ┌───────────────┴───────────────┐
              triage head (3-way)          fine-grained diagnosis head
                    │                                │
        temperature calibration + conformal set + OOD gate + abstention
```

**Stage 1 — inherit, then adapt.** Initialise from DINOv2 ViT-B/14 and run *our existing I-JEPA
code* as domain adaptation for 20–50 epochs on the 553k archive, instead of 700 epochs from
scratch. Estimated **$15–30 rather than $340**, and near-certainly a better encoder. This directly
answers "how do we see more images faster": we inherit 142M images rather than paying for them.

**Stage 2 — the differentiator.** A small transformer (4–6 layers) over the *set* of a patient's
lesion embeddings. It learns "unlike this patient's other lesions" as a representation rather than
as hand-engineered GBDT features, which is what ISIC-2024 winners had to do by hand. Trained with
patient-grouped batches — which our leakage-free `group_id` splits already guarantee.

**Stage 3 — deployment split.** The web demo receives one photo, so distil the set-model's
behaviour into a single-lesion student (the no-context path) for the 512 MB int8 build, and keep
the full set-model for the "upload a body map" flow. Distillation is also how the S/Ti family
members are produced — quantisation alone cannot shrink ViT-L to 77 MB.

**Unchanged and still correct:** sensitivity-first threshold, temperature calibration, split
conformal, Mahalanobis + supervised OOD gate, abstention, measured-RSS budget gate.

---

## 4. Evaluation changes needed to make claims honest

1. **Report at realistic prevalence.** Our balanced split flatters specificity. Add an evaluation
   at the archive's natural ~0.4–5% malignant rate.
2. **Adopt pAUC above 80% TPR** as the primary metric, matching ISIC 2024 and our sensitivity-first
   philosophy — plain AUC hides behaviour in the only region that matters.
3. **Patient-level splits already hold** (`lesnet.data.splits` groups by patient/lesion) — the
   set-transformer makes this non-negotiable rather than merely good practice.
4. **External validation** on a set never used in development (Fitzpatrick17k / DDI / PH2) to
   support any generalisation claim, plus the existing per-Fitzpatrick fairness gate.

---

## 5. Honest risks

- **Malignant lesions inside multi-lesion patients number only 1,944.** The set-transformer is
  data-limited on positives; it may help ranking far more than absolute detection.
- **The web demo gets one image**, so the differentiator does not apply to the current product
  surface without a body-map flow. It is worth building only if that flow is wanted.
- **We will not beat PanDerm on fine-grained diagnosis.** Our advantage is triage + deployability +
  licence, and we should claim exactly that and nothing more.
- Scaled-down sweep results transfer as an *ordering* of choices, not as absolute numbers.

---

## 6. Sequence

| Phase | Work | Cost |
|---|---|---|
| 0 | Finish the local recipe sweep (running) | free |
| 1 | DINOv2 init + I-JEPA domain adaptation, best recipe | $15–30 |
| 2 | Re-fit heads; evaluate at real prevalence with pAUC@80%TPR | free |
| 3 | Patient-set transformer + patient-grouped training | $10–20 |
| 4 | Distil to S/Ti; int8 export; 512 MB RSS gate | free–$10 |

Phases 1–2 alone should beat the current family decisively. Phase 3 is the part that could be
genuinely state of the art for triage.
