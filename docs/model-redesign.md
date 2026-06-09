# A Calibrated, Abstaining Triage System for Skin-Lesion Risk Assessment

**Working paper · LesNet · v0.1 (living document)**
Author: Thomas Behan. Status: design of record; to be revised as experiments are run.

> This is a *living* design paper. Each section states the intended method, the
> rationale, and the evaluation that will confirm or refute it. As we implement and
> measure, results and revisions are appended in-place (see §11, Changelog). Claims
> here are hypotheses until the evaluation harness (§6) reports otherwise.

---

## Abstract

We propose redesigning LesNet from a flat 42-class softmax image classifier into a
**calibrated, uncertainty-aware triage system** whose primary output is a clinical
risk decision (`benign` / `suspicious` / `malignant`) rather than a fine-grained
diagnosis. The system is optimised **sensitivity-first** (minimising missed
malignancies), produces **calibrated probabilities**, **abstains** on
out-of-distribution or low-confidence inputs ("inconclusive — see a clinician"), and
provides **conformal guarantees** that bound its error rate. We treat fairness across
skin tones as a release-blocking requirement, and evaluate on **patient-grouped** and
**external** data at **realistic disease prevalence**. This paper specifies the
problem formulation, the seven methodological components, and — critically — the
evaluation protocol that must precede and gate all modelling work.

---

## 1. Introduction

Automated skin-lesion assessment promises earlier detection of melanoma and other
skin cancers, the deadliness of which is dominated by *late* detection. A tool
intended for unrestricted public use ("anyone can use it") raises the bar on three
axes simultaneously: **clinical safety** (it must not give false reassurance),
**generalisation** (it must work outside the training distribution and across skin
tones), and **honest uncertainty** (it must know when it does not know).

The current LesNet model (branch `M-4`) is a conventional transfer-learning image
classifier. It is a reasonable engineering artefact but is **not a clinically valid
decision system**: it cannot abstain, its probabilities are uncalibrated, its training
distribution is distorted, its evaluation is leakage-prone and prevalence-unrealistic,
and it has no fairness or out-of-distribution safeguards. §3 details these
deficiencies against the code.

This paper reframes the task as **selective, calibrated triage under asymmetric
cost** and specifies a system designed for that objective.

## 2. Problem formulation

### 2.1 The decision, not the diagnosis

Let an input be a dermoscopic or clinical skin image \(x\) (optionally with metadata
\(m\): age, sex, anatomical site). The clinically load-bearing question is **not**
"which of 42 diagnoses is this?" but **"what should the user do?"** We therefore model
a triage decision \(d \in \{\textsf{reassure}, \textsf{refer}, \textsf{urgent},
\textsf{abstain}\}\), derived from a calibrated estimate of malignancy risk
\(p(\textsf{malignant}\mid x, m)\). A secondary fine-grained head predicts the
diagnosis as *explanation only*.

### 2.2 Asymmetric cost and the sensitivity/specificity trade-off

Define Positive = malignant. Then:

- **False negative (FN):** malignant classified benign → *missed cancer* (catastrophic).
- **False positive (FP):** benign classified malignant → *unnecessary referral/biopsy* (low harm).

These errors are **not** symmetric. Let \(c_{FN} \gg c_{FP}\) be their costs. The
optimal decision threshold \(\tau\) minimises expected cost

\[
\mathbb{E}[\text{cost}] = c_{FN}\,(1-\text{Sens})\,\pi + c_{FP}\,(1-\text{Spec})\,(1-\pi),
\]

where \(\pi = p(\textsf{malignant})\) is disease prevalence. Two consequences are
central to this project:

1. **You cannot simultaneously drive FP→0 and keep Sens high.** Sensitivity and
   specificity trade off along the ROC curve; demanding near-zero false positives
   forces missed cancers. The correct objective is **sensitivity-first**: fix a target
   sensitivity (e.g. Sens ≥ 0.97 for melanoma), then maximise specificity subject to it.
2. **Prevalence dominates positive predictive value.** In a public screening setting
   \(\pi\) is small, so even an excellent classifier yields modest PPV
   (\(\text{PPV} = \frac{\text{Sens}\,\pi}{\text{Sens}\,\pi + (1-\text{Spec})(1-\pi)}\)).
   Reporting accuracy on an artificially *balanced* test set (as the current system
   implicitly does) is therefore clinically misleading.

### 2.3 Selective prediction resolves the user's requirement

The stated desire — "highly accurate with a very low false-positive rate" — is best
served not by moving \(\tau\), but by **abstention**. A selective classifier
\((f, g)\) outputs \(f(x)\) when a gate \(g(x)=1\) and abstains otherwise. By routing
ambiguous and out-of-distribution inputs to "see a clinician," we reduce *confident
errors of both kinds* without sacrificing sensitivity on the cases we do answer. The
trade-off becomes **risk vs. coverage** (§5.3), which we can bound with conformal
methods (§5.4).

## 3. Limitations of the current system (M-4)

Evidenced against `lesnet/services/model.py`, `lesnet/services/data.py`, and the data
pipeline in `commands/`:

- **L1 — No clinical target.** Flat 42-way softmax with categorical cross-entropy; all
  misclassifications weighted equally, so a melanoma→nevus error costs the same as a
  harmless confusion. No malignant/benign decision is modelled. *(SRP/clinical.)*
- **L2 — No abstention / OOD gate.** Softmax always emits a confident class; non-skin
  or low-quality inputs receive confident labels. The embedding-similarity code is
  dead.
- **L3 — Uncalibrated confidence.** Raw softmax is systematically overconfident
  (Guo et al., 2017) and cannot be thresholded to a clinical operating point.
- **L4 — Distorted training distribution.** Minority classes are balanced by *offline
  augmented duplication* up to the majority count, then `class_weight` double-corrects.
  This injects near-duplicates, removes real prevalence, and damages calibration/PPV.
- **L5 — Leakage-prone splits.** Per-image (not patient/lesion-grouped) splitting leaks
  the same lesion across train/test, inflating metrics. (The augment-before-split leak
  was fixed; grouped splitting still does not exist.)
- **L6 — Shortcut-prone preprocessing.** Resize + `/255` only. No hair removal, no
  colour-constancy normalisation, no lesion segmentation, no artefact handling — so the
  model can learn device colour casts, rulers, and ink markings instead of lesion
  morphology (documented ISIC failure mode; Winkler et al., 2019).
- **L7 — Skin-tone bias unmeasured.** ISIC is predominantly Fitzpatrick I–III. No
  subgroup evaluation; likely unsafe for darker skin and acral/mucosal melanoma —
  incompatible with "anyone can use it."
- **L8 — Metadata discarded.** `metadata.csv` is used only to sort files.
- **L9 — Ad-hoc architecture.** ResNet152V2 with only 10 layers unfrozen, a bolted-on
  Conv2D, overlapping manual L2, 224 px input, no TTA/ensembling; backbone changed by
  trial and error.
- **L10 — Clinically meaningless metrics.** Reported aggregate accuracy/recall on a
  balanced set; no per-class sensitivity/specificity, ROC/PR-AUC, calibration,
  confusion matrix, subgroup, or external validation. Recall ≈ 0.80 ⇒ ~1 in 5 cancers
  missed.
- **L11 — No working edge/quantisation path; in-process model load.**
- **L12 — No safety/regulatory framing** appropriate to a public medical-adjacent tool.

## 4. Related work (anchors)

- **Dermatology CNN parity:** Esteva et al., *Nature* 2017; Tschandl et al. (HAM10000),
  *Sci. Data* 2018; ISIC challenges (Codella et al.; Combalia et al., 2019).
- **Calibration:** Guo et al., ICML 2017 (temperature scaling); Ovadia et al., 2019
  (calibration under shift).
- **Uncertainty / ensembles:** Lakshminarayanan et al., 2017 (deep ensembles); Gal &
  Ghahramani, 2016 (MC-dropout).
- **Selective prediction / conformal:** Geifman & El-Yaniv, 2017; Vovk et al., 2005;
  Angelopoulos & Bates, 2021 (conformal tutorial).
- **Fairness / skin tone:** Daneshjou et al., 2022 (DDI); Groh et al., 2021
  (Fitzpatrick17k); Pacheco et al., 2020 (PAD-UFES-20).
- **Shortcut learning:** Winkler et al., *JAMA Dermatol.* 2019 (surgical-ink bias);
  Geirhos et al., 2020.

## 5. Proposed method

The seven components below are designed to be implemented and validated in dependency
order (§10). Each maps to a deficiency in §3.

### 5.1 Clinical-hierarchy reframing (addresses L1, L8)
- Map curated ISIC diagnoses to a **clinical taxonomy** with a primary grouping
  \{benign, pre-malignant/uncertain, malignant\}.
- **Multi-task model:** a primary calibrated triage head + an auxiliary fine-grained
  head, sharing a backbone; fine labels regularise the triage head.
- **Metadata fusion:** age/sex/site embedded via a small MLP, concatenated with image
  features.

### 5.2 Sensitivity-first, cost-aware training (addresses L1, L4)
- **Loss:** class-weighted **focal loss** plus an explicit cost matrix making FN on
  malignant classes far more expensive than FP.
- **Imbalance:** handle via balanced sampling + loss weighting, **not** offline
  duplication. Preserve real prevalence in validation/test.
- **Operating point:** choose \(\tau\) on validation to achieve a target malignant
  sensitivity (configurable, default Sens ≥ 0.97), then report resulting Spec/PPV/NPV.

### 5.3 Honest data regimen (addresses L4, L5, L6, L7)
- **Patient/lesion-grouped splits** (GroupKFold by `patient_id`/`lesion_id`).
- **Real-prevalence evaluation;** never balance the test set.
- **Dermoscopy preprocessing:** hair removal (DullRazor-style), **colour constancy**
  (Shades-of-Gray, Finlayson & Trezzi 2004), optional lesion segmentation/crop,
  vignette/artefact handling.
- **Multi-source + skin tone:** combine ISIC with PAD-UFES-20, Fitzpatrick17k, DDI;
  record Fitzpatrick/skin-tone where available.

### 5.4 Architecture & robustness (addresses L9)
- Right-sized pretrained backbone (EfficientNetV2-S/M or a ViT; consider
  self-supervised/derm-pretrained init), **substantial fine-tuning** with discriminative
  learning rates, ~384 px input.
- **Deep ensemble** (K seeds/backbones) + **test-time augmentation** — improves accuracy
  *and* calibration.
- Medically-safe augmentation only (no transforms that destroy diagnostic colour/structure).

### 5.5 Uncertainty, calibration, abstention — the safety core (addresses L2, L3)
- **Calibration:** temperature scaling on a held-out set; report ECE.
- **OOD / quality gate:** reject non-skin / blurry / off-distribution images via an
  embedding-based score (Mahalanobis or energy) and/or a dedicated validity classifier,
  *before* triage.
- **Selective prediction:** abstain ("inconclusive — see a dermatologist") below a
  calibrated-confidence threshold; tune on the **risk–coverage curve**.
- **Conformal prediction:** produce prediction sets with a user-set guaranteed error
  rate \(\alpha\) — the rigorous form of "bounded false-negative/positive rate."

### 5.6 Evaluation that proves viability (addresses L10) — *built first*
See §6. This is the foundation; no modelling claim is accepted without it.

### 5.7 Productisation & safety (addresses L11, L12)
- **Triage-style output** ("monitor / see a clinician / urgent") with a referral bias;
  never a definitive diagnosis; disclaimers retained.
- **Edge + server parity:** working TFLite export (replace the broken
  `tf.quantization.quantize` path); load model once at startup/lazily (done).
- **Model card** per release; drift monitoring; human-in-the-loop.

## 6. Experimental design & evaluation protocol

**This precedes all modelling.** Until we can measure FN/FP honestly, every model
change is unverifiable.

### 6.1 Splitting
- GroupKFold by patient; held-out test set never touched during development.
- A separate **external** dataset (trained-never-seen) for generalisation.

### 6.2 Primary metrics (malignant vs. benign)
- **Sensitivity, Specificity** at the chosen operating point; **ROC-AUC**, **PR-AUC**.
- **Melanoma-specific sensitivity** at the fixed operating point.
- **PPV/NPV at realistic prevalence** \(\pi\) (report a \(\pi\)-sweep).
- **Confusion matrix**; per-fine-class sensitivity where support allows.

### 6.3 Calibration & selective metrics
- **Expected Calibration Error (ECE)** and reliability diagram.
- **Risk–coverage curve** and **AURC**; selective risk at target coverages.
- Conformal **empirical coverage** vs. nominal \(1-\alpha\).

### 6.4 Fairness (release-blocking)
- All primary metrics **stratified by Fitzpatrick group**, anatomical site, age band,
  and acquisition device.
- **Fairness gate:** release blocked if any adequately-supported subgroup's malignant
  sensitivity falls more than a pre-registered margin below the overall.

### 6.5 Reporting
- A versioned **model card** and an auto-generated evaluation report per run.
- Decision-curve / net-benefit analysis for the chosen operating point.

## 7. Fairness & generalisation

Skin-tone coverage and external validation are first-class requirements, not
afterthoughts. We pre-register subgroup gates (§6.4) and report external-dataset
performance alongside in-distribution results. Any deployment claim is scoped to the
populations and devices represented in the validated data.

## 8. Ethical & regulatory considerations

A publicly usable tool that influences health decisions may constitute Software as a
Medical Device (FDA SaMD / EU MDR). Pending any regulatory pathway, the system is
designed and described as **education + triage with a referral bias**, never as
diagnosis; it defaults to "see a clinician" under uncertainty, and it surfaces its
limitations to the user.

## 9. Limitations & threats to validity

- Public dermatology datasets under-represent darker skin and rare/acral melanoma;
  measured fairness is bounded by available data.
- ISIC labelling and selection biases propagate; curation choices are documented.
- Conformal/calibration guarantees hold only under the exchangeability/no-shift
  assumptions they assume; distribution shift in the wild degrades them — hence drift
  monitoring (§5.7).
- "Zero false positives" is not attainable jointly with high sensitivity; the project
  targets bounded, calibrated error with abstention instead.

## 10. Roadmap (dependency-ordered build)

0. **Evaluation harness (§6)** — grouped splits, clinical metrics, calibration,
   risk–coverage, subgroup reporting. *(Foundation; everything else is gated on it.)*
1. Data regimen (§5.3): grouped splits + preprocessing + multi-source ingestion.
2. Clinical-hierarchy labels + multi-task heads (§5.1).
3. Sensitivity-first training & operating-point selection (§5.2).
4. Architecture, ensembling, TTA (§5.4).
5. Calibration + OOD gate + selective/conformal layer (§5.5).
6. Fairness audit + external validation (§7).
7. Productisation, model card, edge parity (§5.7).

Training-dependent stages (1–6) require dataset access and GPU compute; the harness
(0) and the pure-logic components (metrics, splitting, calibration math, conformal,
preprocessing transforms) are buildable and unit-testable without either.

## 11. Changelog

- **v0.1** — Initial design of record. No experiments run yet.

## References

(Indicative; to be completed with full citations as results are incorporated.)
Esteva 2017; Tschandl 2018 (HAM10000); Codella 2018 / Combalia 2019 (ISIC);
Guo 2017; Ovadia 2019; Lakshminarayanan 2017; Gal & Ghahramani 2016;
Geifman & El-Yaniv 2017; Vovk 2005; Angelopoulos & Bates 2021;
Daneshjou 2022 (DDI); Groh 2021 (Fitzpatrick17k); Pacheco 2020 (PAD-UFES-20);
Finlayson & Trezzi 2004 (Shades-of-Gray); Winkler 2019; Geirhos 2020.

## Appendix A — Operating-point selection

```
Given calibrated p_mal on validation (grouped, real prevalence):
  for tau in sorted(unique(p_mal)):
     sens(tau), spec(tau) = confusion(p_mal >= tau, y_mal)
  tau* = min{ tau : sens(tau) >= SENS_TARGET }      # sensitivity-first
  report spec(tau*), PPV/NPV(tau*; pi) over a prevalence sweep
```

## Appendix B — Metric definitions

- Sensitivity (recall, TPR) = TP / (TP + FN); Specificity (TNR) = TN / (TN + FP).
- PPV = TP/(TP+FP); NPV = TN/(TN+FN), reported at stated prevalence \(\pi\).
- ECE = \(\sum_b \frac{|B_b|}{N}\,\lvert \text{acc}(B_b) - \text{conf}(B_b)\rvert\).
- Selective risk \(R(g)\) at coverage \(c(g)\); AURC = area under risk–coverage.
- Conformal: with nonconformity scores and level \(\alpha\), prediction sets satisfy
  \(P(y \in \hat C(x)) \ge 1-\alpha\) under exchangeability.
