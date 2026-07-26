"""JEPA-backed demo predictor for the LesNet web frontend — the 512 MB path.

Encoder = ONNX via onnxruntime (CPU), highest precision that fits the RSS budget;
preprocessing = numpy/PIL only (NO torch/torchvision),
so the served process stays light enough for a ~512 MB box. The diagnosis head AND the OOD
lesion-gate are FIT on the SAME ONNX embeddings that are served, so train == serve. A demo
(embedding probe), not the calibrated/conformal triage pipeline in lesnet.ml.inference.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

# Highest fidelity first: int8 is a last resort, not the default (see OnnxEncoder).
_PRECISION_ORDER = ('fp32', 'fp16', 'int8')

HEAD_FILE = 'diagnosis_head.joblib'
OOD_FILE = 'ood.joblib'
META_FILE = 'demo_meta.json'
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Generic bucket / non-diagnosis labels: excluded from the diagnosis head so they never surface as
# a "most likely diagnosis" (they aren't diagnoses).
EXCLUDE_DIAGNOSES = {
    'benign', 'malignant', 'benign_other', 'collision_at_least_one_malignant_proliferation',
}


def _load_config(artifacts_dir):
    from lesnet.jepa.config import JEPAConfig
    return JEPAConfig(**json.loads((Path(artifacts_dir) / 'jepa_config.json').read_text()))


def _eval_np(pil_image, config):
    """Torch-free eval preprocessing matching build_transform(train=False): medical transforms +
    shorter-side resize to 1.15x + centre crop + [0,1] (or ImageNet) scaling -> (3,H,W) float32."""
    from PIL import Image
    from lesnet.ml.preprocessing import dullrazor_hair_removal, shades_of_gray
    arr = np.asarray(pil_image.convert('RGB'))
    if config.remove_hair:
        arr = np.asarray(dullrazor_hair_removal(arr))
    if config.colour_constancy:
        arr = np.asarray(shades_of_gray(arr))
    image = Image.fromarray(np.clip(arr, 0, 255).astype('uint8'))
    size = config.image_size
    target = max(size, round(size * 1.15))
    w, h = image.size
    scale = target / max(1, min(w, h))
    image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    w, h = image.size
    left, top = (w - size) // 2, (h - size) // 2
    image = image.crop((left, top, left + size, top + size))
    x = np.asarray(image, dtype=np.float32) / 255.0
    if getattr(config, 'normalize', 'unit') == 'imagenet':
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1)


class OnnxEncoder:
    """ONNX encoder over onnxruntime CPU. `threads`=1 for the light serve process; more for
    fast batched fitting."""

    def __init__(self, artifacts_dir, threads=1):
        import onnxruntime as ort
        artifacts_dir = Path(artifacts_dir)
        self.config = _load_config(artifacts_dir)
        # Precision preference: best fidelity that still fits the budget, NOT smallest-wins.
        # Naive dynamic int8 mangles DINOv2 — measured parity error 1.086 (vs 2.9e-05 fp32,
        # 0.022 fp16), which collapsed served ROC-AUC from 0.973 to 0.752. fp32 ViT-S/14 measures
        # 165 MB RSS, well inside 512 MB, so there is nothing to buy by quantising it.
        onnx = next((artifacts_dir / 'export' / f'encoder_{tier}.onnx'
                     for tier in _PRECISION_ORDER
                     if (artifacts_dir / 'export' / f'encoder_{tier}.onnx').exists()), None)
        if onnx is None:
            raise FileNotFoundError(f'no exported encoder under {artifacts_dir / "export"}')
        self.precision = onnx.stem.rsplit('_', 1)[-1]
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(onnx), sess_options=options, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def embed_array(self, image_array):
        from PIL import Image
        x = _eval_np(Image.fromarray(np.asarray(image_array).astype('uint8')), self.config)[None]
        tokens = self.session.run(None, {self.input_name: x.astype(np.float32)})[0]
        return tokens[0].mean(axis=0, keepdims=True)             # (1, D)

    def embed_paths(self, paths, batch=16, progress=None):
        from PIL import Image
        out, buf, done = [], [], 0
        for path in paths:
            try:
                with Image.open(path) as image:
                    buf.append(_eval_np(image, self.config))
            except Exception:  # noqa: BLE001 - skip unreadable
                continue
            if len(buf) >= batch:
                out.append(self._run(np.stack(buf)))
                done += len(buf)
                buf = []
                if progress and done % (batch * 20) == 0:
                    print(f'  encoded {done}', flush=True)
        if buf:
            out.append(self._run(np.stack(buf)))
        return np.concatenate(out) if out else np.empty((0, 1), dtype=np.float32)

    def _run(self, batch_arr):
        tokens = self.session.run(None, {self.input_name: batch_arr.astype(np.float32)})[0]
        return tokens.mean(axis=1)                               # (B, D)


class TorchEncoder:
    """GPU torch encoder with OnnxEncoder's interface, for FITTING heads only.

    Serving stays on ONNX. This is safe because fp32 ONNX export parity is 1.7e-05 — the two
    produce the same embeddings — so "train == serve" still holds. It exists purely for speed:
    fitting a head needs ~10k images encoded, which is minutes on the GPU versus hours of
    single-threaded CPU ONNX for the larger variants. (It would NOT be safe against int8, whose
    parity error is 1.09; that is exactly why int8 is no longer served.)
    """

    def __init__(self, artifacts_dir, device='cuda'):
        import torch

        from lesnet.jepa.vision_transformer import build_encoder
        artifacts_dir = Path(artifacts_dir)
        self.config = _load_config(artifacts_dir)
        self.precision = 'fp32-torch'
        self.device = device if torch.cuda.is_available() else 'cpu'
        state = torch.load(artifacts_dir / 'context_encoder.pt', map_location='cpu',
                           weights_only=False)
        self.model = build_encoder(self.config)
        self.model.load_state_dict(state['state_dict'])
        self.model.eval().to(self.device)

    def _batch(self, arrays):
        import torch
        with torch.no_grad():
            x = torch.from_numpy(np.stack(arrays)).to(self.device)
            return self.model(x).mean(dim=1).float().cpu().numpy()

    def embed_array(self, image_array):
        from PIL import Image
        x = _eval_np(Image.fromarray(np.asarray(image_array).astype('uint8')), self.config)
        return self._batch([x])

    def embed_paths(self, paths, batch=32, progress=None):
        from PIL import Image
        out, buf, done = [], [], 0
        for path in paths:
            try:
                with Image.open(path) as image:
                    buf.append(_eval_np(image, self.config))
            except Exception:  # noqa: BLE001 - skip unreadable
                continue
            if len(buf) >= batch:
                out.append(self._batch(buf))
                done += len(buf)
                buf = []
                if progress and done % (batch * 20) == 0:
                    print(f'  encoded {done}', flush=True)
        if buf:
            out.append(self._batch(buf))
        return np.concatenate(out) if out else np.empty((0, 1), dtype=np.float32)


def _fit_encoder(artifacts_dir):
    """Encoder used to FIT heads. LESNET_FIT_BACKEND=torch swaps in the GPU path (see TorchEncoder)."""
    import os
    if os.environ.get('LESNET_FIT_BACKEND', '').lower() == 'torch':
        return TorchEncoder(artifacts_dir)
    return OnnxEncoder(artifacts_dir, threads=max(1, (os.cpu_count() or 4) - 1))


def build_demo_head(artifacts_dir, manifest, max_fit=3000, **_ignore):
    """Fit + persist the multi-class diagnosis head on the served ONNX embeddings."""
    import joblib
    from lesnet.data.records import load_manifest
    from sklearn.linear_model import LogisticRegression

    artifacts_dir = Path(artifacts_dir)
    enc = _fit_encoder(artifacts_dir)
    records = [r for r in load_manifest(manifest)
               if r.split in ('train', 'val') and r.diagnosis and r.diagnosis not in EXCLUDE_DIAGNOSES]
    if not records:
        raise ValueError('manifest has no train/val rows with a usable diagnosis label.')
    if max_fit and len(records) > max_fit:
        rng = np.random.default_rng(enc.config.seed)
        records = [records[i] for i in rng.choice(len(records), max_fit, replace=False)]

    bucket_of = {}
    for diag, bucket in Counter((r.diagnosis, r.triage_bucket) for r in records):
        bucket_of.setdefault(diag, bucket)

    print(f'encoding {len(records)} lesions (ONNX)', flush=True)
    feats = enc.embed_paths([r.image_path for r in records], progress=True)
    labels = [r.diagnosis for r in records][:len(feats)]

    head = LogisticRegression(max_iter=3000, class_weight='balanced').fit(feats, labels)
    # raw weights only (no sklearn at serve). The benign/malignant triage probe is fit separately
    # by build_malignant_probe (on a balanced all-bucket sample) and merged into this file.
    joblib.dump({'coef': head.coef_.astype(np.float32), 'intercept': head.intercept_.astype(np.float32),
                 'classes': [str(c) for c in head.classes_]}, artifacts_dir / HEAD_FILE)
    (artifacts_dir / META_FILE).write_text(json.dumps({
        'encoder': enc.config.encoder, 'image_size': enc.config.image_size,
        'classes': list(head.classes_), 'bucket_of': bucket_of, 'n_train': int(len(labels)),
        'diagnosis_counts': dict(Counter(labels)), 'backend': f'{enc.precision}-onnx',
    }, indent=2))
    return artifacts_dir / HEAD_FILE


def build_malignant_probe(artifacts_dir, manifest, per_class=2500, target_sensitivity=0.92, **_ignore):
    """Fit the benign-vs-malignant TRIAGE probe on a BALANCED all-bucket sample (benign incl. the
    generic bucket + malignant), with a proper sensitivity-first threshold picked on a held-out
    calibration split. Merged into the head file as mal_coef/mal_intercept/mal_refer/mal_urgent.
    The earlier version trained only on the specific-diagnosis subset -> learned no generic benign
    -> ~12% specificity; this fixes that."""
    import joblib
    from lesnet.data.records import load_manifest
    from sklearn.linear_model import LogisticRegression

    artifacts_dir = Path(artifacts_dir)
    enc = _fit_encoder(artifacts_dir)
    recs = [r for r in load_manifest(manifest)
            if r.split in ('train', 'val') and r.triage_bucket in ('benign', 'malignant')]
    rng = np.random.default_rng(enc.config.seed)
    by_bucket = {'benign': [r for r in recs if r.triage_bucket == 'benign'],
                 'malignant': [r for r in recs if r.triage_bucket == 'malignant']}
    sample = []
    for bucket, rs in by_bucket.items():
        idx = rng.choice(len(rs), min(per_class, len(rs)), replace=False)
        sample += [(rs[i], 1 if bucket == 'malignant' else 0) for i in idx]
    rng.shuffle(sample)
    print(f'encoding {len(sample)} lesions for the malignant probe (ONNX)', flush=True)
    feats = enc.embed_paths([r.image_path for r, _ in sample], progress=True)
    y = np.array([lbl for _, lbl in sample][:len(feats)])

    # 80/20 split so the threshold is calibrated on held-out predictions
    n = len(y)
    perm = rng.permutation(n)
    cut = int(n * 0.8)
    tr, cal = perm[:cut], perm[cut:]
    probe = LogisticRegression(max_iter=3000, class_weight='balanced').fit(feats[tr], y[tr])
    coef, intercept = probe.coef_[0], float(probe.intercept_[0])
    p_cal = 1.0 / (1.0 + np.exp(-(feats[cal] @ coef + intercept)))
    ycal = y[cal]

    # highest threshold achieving >= target sensitivity on the calibration split (maximises spec)
    cand = np.unique(np.concatenate([[0.0], p_cal, [1.0]]))
    best = 0.0
    for t in cand:
        sens = ((p_cal >= t) & (ycal == 1)).sum() / max((ycal == 1).sum(), 1)
        if sens >= target_sensitivity:
            best = float(t)
    spec = ((p_cal < best) & (ycal == 0)).sum() / max((ycal == 0).sum(), 1)
    urgent = float(np.percentile(p_cal[ycal == 1], 50))
    print(f'mal probe: refer={best:.3f} urgent={urgent:.3f} cal sens>={target_sensitivity} spec={spec:.2f}',
          flush=True)

    head = joblib.load(artifacts_dir / HEAD_FILE)
    head.update({'mal_coef': coef.astype(np.float32), 'mal_intercept': intercept,
                 'mal_refer': best, 'mal_urgent': max(urgent, best + 1e-3)})
    joblib.dump(head, artifacts_dir / HEAD_FILE)
    return best, spec


def build_ood_gate(artifacts_dir, manifest, n_pos=1500, n_neg=600, keep_lesions=0.995,
                   **_ignore):
    """Supervised lesion-vs-not gate. Positives = ISIC lesions; negatives = random natural photos.
    Below the lesion-probability threshold -> abstain (out of distribution).

    The threshold keeps `keep_lesions` of REAL lesions rather than maximising non-lesion rejection.
    That asymmetry is deliberate and clinical: wrongly abstaining on a melanoma is dangerous, while
    showing a result for a photo of a desk is merely embarrassing. An earlier version cut at the
    4th percentile of lesion scores, so it rejected 4% of genuine lesions *by construction* — and
    because that percentile is calibrated on ISIC's own distribution, real dermoscopy from outside
    it (different device, magnification, or a textbook image) was rejected far more often than 4%.
    """
    import io
    import urllib.request
    import joblib
    from lesnet.data.records import load_manifest
    from PIL import Image
    from sklearn.linear_model import LogisticRegression

    artifacts_dir = Path(artifacts_dir)
    enc = _fit_encoder(artifacts_dir)
    records = [r for r in load_manifest(manifest) if r.split in ('train', 'val')]
    rng = np.random.default_rng(enc.config.seed)
    if len(records) > n_pos:
        records = [records[i] for i in rng.choice(len(records), n_pos, replace=False)]
    pos_paths = [r.image_path for r in records]
    is_malignant = np.array([r.triage_bucket == 'malignant' for r in records])

    neg_dir = artifacts_dir / 'ood_neg'
    neg_dir.mkdir(exist_ok=True)
    neg_paths = []
    print(f'downloading {n_neg} negative photos', flush=True)
    for i in range(n_neg):
        dest = neg_dir / f'{i}.jpg'
        if not dest.exists():
            try:
                data = urllib.request.urlopen(f'https://picsum.photos/seed/neg{i}/256/256', timeout=20).read()
                Image.open(io.BytesIO(data)).convert('RGB').save(dest)
            except Exception:  # noqa: BLE001
                continue
        neg_paths.append(dest)

    print(f'encoding {len(pos_paths)} lesions + {len(neg_paths)} negatives (ONNX)', flush=True)
    pos = enc.embed_paths(pos_paths, progress=True)
    neg = enc.embed_paths(neg_paths, progress=True)
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    gate = LogisticRegression(max_iter=3000, class_weight='balanced').fit(x, y)

    p_pos = gate.predict_proba(pos)[:, 1]
    p_neg = gate.predict_proba(neg)[:, 1]
    # Calibrate on MALIGNANT lesions, not all lesions. A "does this look like a lesion" gate
    # scores atypical lesions lower, and malignancy is what makes a lesion atypical — so a
    # threshold set on the whole population rejects malignant cases preferentially. Measured: every
    # single false abstention on a held-out sample was a malignant lesion. The gate must be
    # calibrated on the class it is most dangerous to reject.
    reference = p_pos[is_malignant[:len(p_pos)]] if is_malignant[:len(p_pos)].any() else p_pos
    threshold = float(np.percentile(reference, (1.0 - keep_lesions) * 100.0))
    kept_all = float((p_pos >= threshold).mean())
    kept_mal = float((reference >= threshold).mean())
    print(f'gate threshold={threshold:.4f}  keeps {kept_mal:.1%} of MALIGNANT / {kept_all:.1%} of '
          f'all lesions  negatives rejected={(p_neg < threshold).mean():.1%}', flush=True)
    joblib.dump({'coef': gate.coef_[0].astype(np.float32), 'intercept': float(gate.intercept_[0]),
                 'threshold': threshold, 'keep_lesions': keep_lesions,
                 'kept_malignant': kept_mal, 'kept_all_lesions': kept_all,
                 'negatives_rejected': float((p_neg < threshold).mean())},
                artifacts_dir / OOD_FILE)
    return threshold


class JEPADemoPredictor:
    """Frontend triage contract with top-k diagnoses. Serves via int8 ONNX (torch-free, ~512 MB)."""

    def __init__(self, artifacts_dir):
        import joblib
        artifacts_dir = Path(artifacts_dir)
        self.meta = json.loads((artifacts_dir / META_FILE).read_text())
        head = joblib.load(artifacts_dir / HEAD_FILE)
        self.head_coef, self.head_intercept = np.asarray(head['coef']), np.asarray(head['intercept'])
        self.classes = np.asarray(head['classes'])
        self.bucket_of = self.meta['bucket_of']
        # calibrated benign/malignant probe for the triage decision (if present)
        if 'mal_coef' in head:
            self.mal_coef = np.asarray(head['mal_coef'])
            self.mal_intercept = float(head['mal_intercept'])
            self.mal_refer, self.mal_urgent = float(head['mal_refer']), float(head['mal_urgent'])
        else:
            self.mal_coef = None
        self.encoder = OnnxEncoder(artifacts_dir, threads=1)
        self.bundle = {'label_maps': {'fine_vocabulary': {
            name: i for i, name in enumerate(self.meta['classes'])}}}
        ood_path = artifacts_dir / OOD_FILE
        if ood_path.exists():
            ood = joblib.load(ood_path)
            self.ood_coef, self.ood_intercept = np.asarray(ood['coef']), float(ood['intercept'])
            self.ood_threshold = float(ood['threshold'])
        else:
            self.ood_coef = self.ood_threshold = None

    def _proba(self, embedding):
        """Multinomial logistic softmax from raw weights (no sklearn at serve time)."""
        scores = embedding @ self.head_coef.T + self.head_intercept       # (1, n_classes)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return (exp / exp.sum(axis=1, keepdims=True))[0]

    def _is_ood(self, embedding):
        if self.ood_coef is None:
            return False, None
        z = float(embedding[0] @ self.ood_coef + self.ood_intercept)
        p_lesion = 1.0 / (1.0 + np.exp(-z))
        return p_lesion < self.ood_threshold, p_lesion

    def predict(self, image_array, record=None, top_k=3):
        from lesnet.ml.triage import triage_decision
        embedding = self.encoder.embed_array(image_array)

        is_ood, p_lesion = self._is_ood(embedding)
        if is_ood:
            return {
                'triage': 'abstain', 'valid_image': False, 'reason': 'out_of_distribution',
                'lesion_probability': p_lesion,
                'model': f"JEPA {self.meta['encoder']} (diagnosis probe demo)",
                'disclaimer': 'Research demo: self-supervised embedding + linear diagnosis probe, not a '
                              'diagnostic device and not the calibrated triage pipeline.',
            }

        proba = self._proba(embedding)
        classes = self.classes
        order = np.argsort(proba)[::-1]
        fine = [{'label': str(classes[i]), 'probability': float(proba[i]) * 100.0} for i in order[:top_k]]
        if self.mal_coef is not None:
            # calibrated benign/malignant probe drives the triage decision (sensitivity-first)
            z = float(embedding[0] @ self.mal_coef + self.mal_intercept)
            p_malignant = 1.0 / (1.0 + np.exp(-z))
            p_benign = 1.0 - p_malignant
            p_suspicious = 0.0
            decision = triage_decision(p_malignant, valid_image=True, refer_threshold=self.mal_refer,
                                       urgent_threshold=self.mal_urgent, abstain_band=(0.0, 0.0))
        else:  # fallback: marginalise the diagnosis head (uncalibrated)
            p_malignant = float(sum(proba[i] for i in range(len(classes))
                                    if self.bucket_of.get(str(classes[i])) == 'malignant'))
            p_benign = float(sum(proba[i] for i in range(len(classes))
                                 if self.bucket_of.get(str(classes[i])) == 'benign'))
            p_suspicious = max(0.0, 1.0 - p_malignant - p_benign)
            decision = triage_decision(p_malignant, valid_image=True,
                                       refer_threshold=0.30, urgent_threshold=0.66, abstain_band=(0.0, 0.0))
        return {
            'triage': decision, 'valid_image': True, 'p_malignant': p_malignant,
            'probabilities': {'benign': p_benign, 'suspicious': p_suspicious, 'malignant': p_malignant},
            'conformal_set': [decision], 'lesion_type': fine[0]['label'] if fine else None,
            'fine_predictions': fine,
            'model': f"JEPA {self.meta['encoder']} (diagnosis probe demo)",
            'disclaimer': 'Research demo: self-supervised embedding + linear diagnosis probe, not a '
                          'diagnostic device and not the calibrated triage pipeline.',
        }
