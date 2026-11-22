"""Backbone bake-off: which openly-licensed, sub-512 MB encoder gives the best frozen triage features?

Every candidate is scored the same way, so the ranking means something:

  * the SAME leakage-free grouped manifest splits (fit on train, choose C + operating point on val,
    score on test — never the threshold-on-train shortcut);
  * the SAME input contract as the served demo (Shades-of-Gray colour constancy, no hair removal,
    resize x1.15 -> centre crop), with each model's OWN normalisation unless stated otherwise;
  * the SAME heads: standardised logistic regression for triage (benign vs malignant) and a
    multinomial one for the fine-grained diagnosis (top-1 / top-3).

Two test sets are reported:

  1. ``balanced`` — the manifest's whole held-out test split (~5.9k, ~40% malignant). Primary
     ranking metric: pAUC above 80% TPR (the ISIC 2024 metric), with a bootstrap 95% CI.
  2. ``natural`` — the same test malignants plus a pool of archive benigns from patients never seen
     in train/val, at ~10% prevalence, with PPV extrapolated to 5% and 1% prevalence. The pool is
     mostly 3D-TBP tiles, so read its AUC as optimistic; it is there for specificity/PPV.

Licence policy (docs/architecture-research-2026.md): only open-source, commercially usable WEIGHTS.

    python commands/run_backbone_bakeoff.py --stage all
    python commands/run_backbone_bakeoff.py --stage extract --backbones pe_core_s siglip2_b
    python commands/run_backbone_bakeoff.py --stage report

LesNet is a research/triage tool, **not** a diagnostic device.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

# name -> (timm id, input size, normalisation, weights licence). 'model' = the checkpoint's own
# mean/std from timm; 'unit' = raw [0,1], which is what the current demo feeds DINOv2.
BACKBONES = {
    'dinov2_s_unit': ('vit_small_patch14_dinov2.lvd142m', 224, 'unit', 'Apache-2.0'),
    'dinov2_s': ('vit_small_patch14_dinov2.lvd142m', 224, 'model', 'Apache-2.0'),
    'dinov2_s_reg': ('vit_small_patch14_reg4_dinov2.lvd142m', 224, 'model', 'Apache-2.0'),
    'dinov2_b': ('vit_base_patch14_dinov2.lvd142m', 224, 'model', 'Apache-2.0'),
    'dinov2_b_reg': ('vit_base_patch14_reg4_dinov2.lvd142m', 224, 'model', 'Apache-2.0'),
    'pe_core_t': ('vit_pe_core_tiny_patch16_384.fb', 384, 'model', 'Apache-2.0'),
    'pe_core_s': ('vit_pe_core_small_patch16_384.fb', 384, 'model', 'Apache-2.0'),
    'pe_core_b': ('vit_pe_core_base_patch16_224.fb', 224, 'model', 'Apache-2.0'),
    'siglip2_b': ('vit_base_patch16_siglip_224.v2_webli', 224, 'model', 'Apache-2.0'),
    'siglip2_b_384': ('vit_base_patch16_siglip_384.v2_webli', 384, 'model', 'Apache-2.0'),
}
DEFAULT_ORDER = list(BACKBONES)

RSS_BUDGET_MB = 512
RUNTIME_OVERHEAD_MB = 80          # onnxruntime + numpy + app; measured ~78 MB on the Small demo
TARGET_SENSITIVITY = 0.95
MIN_DX_TRAIN = 30                 # diagnosis classes rarer than this in train are left out
POOL_SIZE = 15000
SEED = 0


# ---------------------------------------------------------------------------------------- data
def _manifest_rows(manifest):
    with open(manifest, newline='', encoding='utf-8') as handle:
        rows = [r for r in csv.DictReader(handle) if r['triage_bucket'] in ('benign', 'malignant')]
    return [{'path': r['image_path'], 'split': r['split'], 'label': int(r['triage_bucket'] == 'malignant'),
             'diagnosis': r['diagnosis'], 'fitz': int(float(r['fitzpatrick'])) if r['fitzpatrick'] else -1,
             'group': r['group_id'], 'pool': 0} for r in rows]


def _benign_pool(rows, archive_dir, size, seed):
    """Archive benigns from patients absent from train/val, never in the manifest themselves.

    Images without a patient_id are skipped: they are grouped by image id in the manifest, so a
    second photo of a training lesion could otherwise slip into the test pool.
    """
    seen_groups = {r['group'] for r in rows if r['split'] in ('train', 'val')}
    seen_images = {Path(r['path']).stem.removeprefix('isic_') for r in rows}
    images_dir = Path(archive_dir) / 'images'
    with open(Path(archive_dir) / 'metadata.csv', newline='', encoding='utf-8') as handle:
        candidates = [m for m in csv.DictReader(handle)
                      if m['diagnosis_1'] == 'Benign' and m['patient_id']
                      and m['patient_id'] not in seen_groups and m['isic_id'] not in seen_images
                      and (images_dir / f"{m['isic_id']}.jpg").exists()]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidates), min(size, len(candidates)), replace=False)
    return [{'path': str(images_dir / f"{candidates[i]['isic_id']}.jpg"), 'split': 'natural_pool',
             'label': 0, 'diagnosis': '', 'fitz': -1, 'group': candidates[i]['patient_id'], 'pool': 1}
            for i in sorted(chosen)]


def build_index(manifest, archive_dir, cache):
    index_path = cache / 'index.json'
    if index_path.exists():
        return json.loads(index_path.read_text())
    rows = _manifest_rows(manifest)
    pool = _benign_pool(rows, archive_dir, POOL_SIZE, SEED) if archive_dir else []
    index = rows + pool
    cache.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index))
    print(f'index: {len(rows)} manifest rows + {len(pool)} natural-pool benigns', flush=True)
    return index


class _ImageDataset:
    """PIL -> tensor under the served input contract. Pool images (384 px archive) are first
    downscaled to the manifest's 256 px thumbnails so resolution cannot act as a label shortcut."""

    def __init__(self, index, size, mean, std):
        from torchvision import transforms
        self.index = index
        bicubic = transforms.InterpolationMode.BICUBIC
        self.transform = transforms.Compose([
            transforms.Resize(int(size * 1.15), interpolation=bicubic), transforms.CenterCrop(size),
            transforms.ToTensor(), transforms.Normalize(mean, std)])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        from PIL import Image

        from lesnet.ml.preprocessing import shades_of_gray
        row = self.index[i]
        with Image.open(row['path']) as image:
            image = image.convert('RGB')
            if row['pool']:
                image.thumbnail((256, 256), Image.LANCZOS)
            array = shades_of_gray(np.asarray(image))
        return self.transform(Image.fromarray(np.clip(array, 0, 255).astype('uint8')))


# ---------------------------------------------------------------------------------- extraction
def extract(name, index, cache, batch_size, workers):
    import timm
    import torch
    from torch.utils.data import DataLoader

    out = cache / f'{name}.npz'
    if out.exists():
        print(f'{name}: cached', flush=True)
        return
    timm_id, size, norm, _ = BACKBONES[name]
    kwargs = {'img_size': size} if 'dinov2' in timm_id else {}
    model = timm.create_model(timm_id, pretrained=True, num_classes=0, **kwargs).eval().cuda()
    params = sum(p.numel() for p in model.parameters())
    if norm == 'unit':
        mean, std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    else:
        cfg = timm.data.resolve_data_config({}, model=model)
        mean, std = cfg['mean'], cfg['std']

    loader = DataLoader(_ImageDataset(index, size, mean, std), batch_size=batch_size,
                        num_workers=workers, persistent_workers=workers > 0, pin_memory=True)
    pooled, patch_mean = [], []
    start, done = time.time(), 0
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        for images in loader:
            tokens = model.forward_features(images.cuda(non_blocking=True))
            pooled.append(model.forward_head(tokens, pre_logits=True).float().cpu().numpy())
            patch_mean.append(tokens[:, model.num_prefix_tokens:].mean(1).float().cpu().numpy())
            done += len(images)
            if done % (batch_size * 50) == 0:
                print(f'{name}: {done}/{len(index)} ({done / (time.time() - start):.0f} img/s)', flush=True)
    seconds = time.time() - start
    np.savez(out, pooled=np.concatenate(pooled).astype(np.float16),
             patch_mean=np.concatenate(patch_mean).astype(np.float16),
             params=params, seconds=seconds, size=size, norm=norm, mean=mean, std=std)
    print(f'{name}: {params / 1e6:.1f}M params, {len(index)} images in {seconds:.0f}s '
          f'({len(index) / seconds:.0f} img/s, end-to-end incl. CPU preprocessing)', flush=True)
    del model
    torch.cuda.empty_cache()


# -------------------------------------------------------------------------------------- probes
def partial_auc(labels, scores, min_tpr=0.80):
    """Unnormalised ROC area above `min_tpr` (max 0.2) — the ISIC 2024 metric, as in the family docs."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    mask = tpr >= min_tpr
    if not mask.any():
        return 0.0
    fpr_region = np.concatenate([[np.interp(min_tpr, tpr, fpr)], fpr[mask]])
    tpr_region = np.concatenate([[min_tpr], tpr[mask]])
    return float(np.trapezoid(tpr_region - min_tpr, fpr_region))


def _bootstrap(labels, scores, n=500, seed=SEED):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    aucs, paucs = [], []
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        if labels[idx].min() == labels[idx].max():
            continue
        aucs.append(roc_auc_score(labels[idx], scores[idx]))
        paucs.append(partial_auc(labels[idx], scores[idx]))
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]  # noqa: E731
    return q(aucs), q(paucs)


def _ppv(sens, spec, prevalence):
    denom = sens * prevalence + (1 - spec) * (1 - prevalence)
    return round(sens * prevalence / denom, 4) if denom else None


def _triage_probe(feats, index):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    from lesnet.jepa.probe import _fairness, _sens_spec, _threshold_for_sensitivity

    split = np.array([r['split'] for r in index])
    y = np.array([r['label'] for r in index])
    fitz = np.array([r['fitz'] for r in index])
    tr, va, te, pool = (split == 'train'), (split == 'val'), (split == 'test'), (split == 'natural_pool')
    scaler = StandardScaler().fit(feats[tr])
    x = scaler.transform(feats).astype(np.float32)

    best = None
    for c in (0.01, 0.03, 0.1, 0.3, 1.0):
        clf = LogisticRegression(C=c, max_iter=3000, class_weight='balanced').fit(x[tr], y[tr])
        val_pauc = partial_auc(y[va], clf.predict_proba(x[va])[:, 1])
        if best is None or val_pauc > best[0]:
            best = (val_pauc, c, clf)
    _, c, clf = best
    proba = clf.predict_proba(x)[:, 1]
    threshold = _threshold_for_sensitivity(y[va], proba[va], TARGET_SENSITIVITY)   # chosen on VAL

    sens, spec = _sens_spec(y[te], proba[te], threshold)
    auc_ci, pauc_ci = _bootstrap(y[te], proba[te])
    per_fitz, gap = _fairness(y[te], proba[te], fitz[te], threshold)
    result = {'C': c, 'threshold': round(threshold, 4),
              'balanced': {'n': int(te.sum()), 'malignant': int(y[te].sum()),
                           'roc_auc': round(float(roc_auc_score(y[te], proba[te])), 4), 'roc_auc_ci': auc_ci,
                           'pauc_80tpr': round(partial_auc(y[te], proba[te]), 4), 'pauc_ci': pauc_ci,
                           'sensitivity': round(sens, 4), 'specificity': round(spec, 4),
                           'fitzpatrick_sensitivity': per_fitz, 'worst_fitz_gap': gap}}
    if pool.any():
        nat = te | pool
        n_sens, n_spec = _sens_spec(y[nat], proba[nat], threshold)
        result['natural'] = {
            'n': int(nat.sum()), 'prevalence': round(float(y[nat].mean()), 4),
            'roc_auc': round(float(roc_auc_score(y[nat], proba[nat])), 4),
            'pauc_80tpr': round(partial_auc(y[nat], proba[nat]), 4),
            'sensitivity': round(n_sens, 4), 'specificity': round(n_spec, 4),
            'pool_specificity': round(float((proba[pool] < threshold).mean()), 4),
            'ppv_at_5pct': _ppv(n_sens, n_spec, 0.05), 'ppv_at_1pct': _ppv(n_sens, n_spec, 0.01)}
    return result


def _diagnosis_probe(feats, index):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from lesnet.jepa.serve import EXCLUDE_DIAGNOSES

    split = np.array([r['split'] for r in index])
    dx = np.array([r['diagnosis'] for r in index])
    usable = (split != 'natural_pool') & ~np.isin(dx, list(EXCLUDE_DIAGNOSES) + [''])
    names, counts = np.unique(dx[usable & (split == 'train')], return_counts=True)
    keep = set(names[counts >= MIN_DX_TRAIN])
    mask = usable & np.isin(dx, list(keep))
    tr, te = mask & np.isin(split, ['train', 'val']), mask & (split == 'test')
    scaler = StandardScaler().fit(feats[tr])
    clf = LogisticRegression(C=0.1, max_iter=2000, class_weight='balanced')
    clf.fit(scaler.transform(feats[tr]), dx[tr])
    proba = clf.predict_proba(scaler.transform(feats[te]))
    top3 = clf.classes_[np.argsort(-proba, axis=1)[:, :3]]
    return {'classes': len(clf.classes_), 'n_test': int(te.sum()),
            'top1': round(float((top3[:, 0] == dx[te]).mean()), 4),
            'top3': round(float((top3 == dx[te][:, None]).any(axis=1).mean()), 4)}


def probe(name, index, cache):
    data = np.load(cache / f'{name}.npz')
    params = int(data['params'])
    fp32_mb, fp16_mb = params * 4 / 1e6, params * 2 / 1e6
    result = {'backbone': name, 'timm_id': BACKBONES[name][0], 'licence': BACKBONES[name][3],
              'input_size': int(data['size']), 'normalisation': str(data['norm']),
              'params_m': round(params / 1e6, 1), 'fp32_mb': round(fp32_mb), 'fp16_mb': round(fp16_mb),
              'est_rss_fp32_mb': round(fp32_mb + RUNTIME_OVERHEAD_MB),
              'fits_512_fp32': fp32_mb + RUNTIME_OVERHEAD_MB < RSS_BUDGET_MB,
              'fits_512_fp16': fp16_mb + RUNTIME_OVERHEAD_MB < RSS_BUDGET_MB,
              'gpu_extract_img_s': round(len(index) / float(data['seconds'])), 'features': {}}
    for kind in ('pooled', 'patch_mean'):
        feats = data[kind].astype(np.float32)
        t0 = time.time()
        result['features'][kind] = {'dim': feats.shape[1], 'triage': _triage_probe(feats, index),
                                    'diagnosis': _diagnosis_probe(feats, index)}
        tri = result['features'][kind]['triage']['balanced']
        print(f"{name:14s} {kind:10s} AUC {tri['roc_auc']:.4f} pAUC {tri['pauc_80tpr']:.4f} "
              f"sens {tri['sensitivity']:.3f} spec {tri['specificity']:.3f} "
              f"dx-top3 {result['features'][kind]['diagnosis']['top3']:.3f} ({time.time() - t0:.0f}s)",
              flush=True)
    best = max(result['features'], key=lambda k: result['features'][k]['triage']['balanced']['pauc_80tpr'])
    result['best_feature'] = best
    (cache / f'{name}.probe.json').write_text(json.dumps(result, indent=2))
    return result


# -------------------------------------------------------------------------------------- report
def write_report(results, out):
    rows = sorted(results, key=lambda r: -r['features'][r['best_feature']]['triage']['balanced']['pauc_80tpr'])
    lines = ['# Backbone bake-off — frozen features, linear heads', '',
             'Generated by `commands/run_backbone_bakeoff.py`. Same splits, preprocessing and heads for every '
             'backbone; C and the 95%-sensitivity operating point are chosen on **val**, scores are on the '
             'held-out grouped **test** split. LesNet is a research/triage tool, **not** a diagnostic device.', '',
             '## Ranking (balanced test split, best of pooled / patch-mean features)', '',
             '| # | backbone | params | fits 512 MB | pAUC@80 [95% CI] | ROC-AUC [95% CI] | sens | spec | '
             'dx top-1 | dx top-3 |', '|---|---|---|---|---|---|---|---|---|---|']
    for i, r in enumerate(rows, 1):
        f = r['features'][r['best_feature']]
        b, d = f['triage']['balanced'], f['diagnosis']
        fits = 'fp32' if r['fits_512_fp32'] else ('fp16' if r['fits_512_fp16'] else 'no')
        lines.append(f"| {i} | {r['backbone']} ({r['best_feature']}) | {r['params_m']}M | {fits} | "
                     f"{b['pauc_80tpr']:.4f} [{b['pauc_ci'][0]:.4f}–{b['pauc_ci'][1]:.4f}] | "
                     f"{b['roc_auc']:.4f} [{b['roc_auc_ci'][0]:.4f}–{b['roc_auc_ci'][1]:.4f}] | "
                     f"{b['sensitivity']:.3f} | {b['specificity']:.3f} | {d['top1']:.3f} | {d['top3']:.3f} |")
    if any('natural' in r['features'][r['best_feature']]['triage'] for r in rows):
        lines += ['', '## Natural-prevalence view (same threshold)', '',
                  'Test malignants + archive benigns from unseen patients. The pool is mostly 3D-TBP tiles, so '
                  'AUC here is optimistic; read it for specificity and PPV.', '',
                  '| backbone | n | prevalence | ROC-AUC | sens | spec | PPV @5% | PPV @1% |',
                  '|---|---|---|---|---|---|---|---|']
        for r in rows:
            n = r['features'][r['best_feature']]['triage'].get('natural')
            if n:
                lines.append(f"| {r['backbone']} | {n['n']} | {n['prevalence']:.3f} | {n['roc_auc']:.4f} | "
                             f"{n['sensitivity']:.3f} | {n['specificity']:.3f} | {n['ppv_at_5pct']} | "
                             f"{n['ppv_at_1pct']} |")
    lines += ['', '## Every backbone × feature', '',
              '| backbone | input | norm | feature | dim | pAUC@80 | ROC-AUC | sens | spec | worst-Fitz gap | '
              'dx top-3 | extract img/s |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        for kind, f in r['features'].items():
            b = f['triage']['balanced']
            lines.append(f"| {r['backbone']} | {r['input_size']} | {r['normalisation']} | {kind} | {f['dim']} | "
                         f"{b['pauc_80tpr']:.4f} | {b['roc_auc']:.4f} | {b['sensitivity']:.3f} | "
                         f"{b['specificity']:.3f} | {b['worst_fitz_gap']} | {f['diagnosis']['top3']:.3f} | "
                         f"{r['gpu_extract_img_s']} |")
    lines += ['', 'Memory columns are **estimates** (4 bytes/param fp32 + ~80 MB runtime); finalists still need '
              'the measured-RSS gate after ONNX export.']
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    out.with_suffix('.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(f'wrote {out} (+ .json)', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Frozen-feature backbone bake-off for LesNet triage.')
    parser.add_argument('--manifest', default='data/isic_dx/manifest.csv')
    parser.add_argument('--archive', default='data/isic_384', help='archive dir for the natural-prevalence '
                        'benign pool (metadata.csv + images/); pass "" to skip it')
    parser.add_argument('--cache', default='data/bakeoff')
    parser.add_argument('--backbones', nargs='+', default=DEFAULT_ORDER, choices=list(BACKBONES))
    parser.add_argument('--stage', choices=('extract', 'probe', 'report', 'all'), default='all')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--limit', type=int, default=0, help='smoke: use only the first N rows per split')
    parser.add_argument('--out', default='docs/backbone-bakeoff.md')
    args = parser.parse_args()

    cache = Path(args.cache)
    index = build_index(args.manifest, args.archive or None, cache)
    if args.limit:
        by_split = {}
        for row in index:
            by_split.setdefault(row['split'], []).append(row)
        rng = np.random.default_rng(SEED)
        index = [rows[i] for rows in by_split.values()
                 for i in sorted(rng.choice(len(rows), min(args.limit, len(rows)), replace=False))]
        cache = cache / f'limit{args.limit}'
        cache.mkdir(parents=True, exist_ok=True)

    if args.stage in ('extract', 'all'):
        for name in args.backbones:
            extract(name, index, cache, args.batch_size, args.workers)
    if args.stage in ('probe', 'all'):
        for name in args.backbones:
            if (cache / f'{name}.npz').exists():
                probe(name, index, cache)
    if args.stage in ('report', 'probe', 'all'):
        results = [json.loads(p.read_text()) for p in sorted(cache.glob('*.probe.json'))]
        if results:
            write_report(results, Path(args.out) if not args.limit else cache / 'report.md')


if __name__ == '__main__':
    main()
