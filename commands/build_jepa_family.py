"""Build the deployable LesNet model family end to end, from openly-licensed DINOv2 backbones.

For each variant: materialise the encoder -> export fp32/int8 ONNX -> fit the triage probe,
diagnosis head and OOD gate on the SAME int8 embeddings that get served (train == serve) ->
evaluate on the held-out test split -> record measured inference RSS against the 512 MB budget.

Every backbone here is Apache-2.0 (DINOv2), so the result is redistributable commercially. See
docs/architecture-research-2026.md for why off-the-shelf DINOv2 beats our from-scratch encoders.

    python commands/build_jepa_family.py --out artifacts/family \
        --manifest data/isic_dx/manifest.csv --ood-neg <dir of non-lesion photos>
"""
import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

# family name -> (our encoder key, DINOv2 checkpoint). Params: 21M / 86M / 300M.
FAMILY = {
    'small': ('vit_small', 'dinov2_vits14'),
    'medium': ('vit_base', 'dinov2_vitb14'),
    'large': ('vit_large', 'dinov2_vitl14'),
}


def make_encoder_checkpoint(variant, out_dir, image_size, adapt_from=None):
    """Write a context_encoder.pt the export/serve path understands."""
    import torch

    from lesnet.jepa.config import JEPAConfig
    from lesnet.jepa.pretrained import load_pretrained
    from lesnet.jepa.vision_transformer import build_encoder

    encoder_key, checkpoint = FAMILY[variant]
    config = JEPAConfig(encoder=encoder_key, image_size=image_size, patch_size=14,
                        pretrained=checkpoint, layerscale_init=1e-4, remove_hair=False,
                        colour_constancy=True, artifacts_dir=str(out_dir))
    encoder = build_encoder(config)
    if adapt_from and Path(adapt_from).exists():
        state = torch.load(adapt_from, map_location='cpu', weights_only=False)
        encoder.load_state_dict(state['context_encoder'] if 'context_encoder' in state
                                else state['state_dict'])
        print(f'  loaded domain-adapted weights from {adapt_from}', flush=True)
    else:
        load_pretrained(encoder, checkpoint)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'context_encoder.pt'
    torch.save({'state_dict': encoder.state_dict(), 'config': asdict(config)}, path)
    (out_dir / 'jepa_config.json').write_text(json.dumps(asdict(config), indent=2))
    params = sum(p.numel() for p in encoder.parameters())
    return path, config, params


def main():
    parser = argparse.ArgumentParser(description="Build + evaluate the deployable model family.")
    parser.add_argument('--out', default='artifacts/family')
    parser.add_argument('--manifest', default='data/isic_dx/manifest.csv')
    parser.add_argument('--ood-neg', default='', help="Directory of non-lesion photos for the gate.")
    parser.add_argument('--variants', nargs='*', default=list(FAMILY))
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--adapt-from', default='', help="Optional domain-adapted training_state.pt.")
    parser.add_argument('--probe-per-class', type=int, default=2500)
    parser.add_argument('--head-max-fit', type=int, default=3000)
    parser.add_argument('--eval-n', type=int, default=600)
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()

    from lesnet.jepa.export import export_tiers
    from lesnet.jepa.serve import build_demo_head, build_malignant_probe, build_ood_gate

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / 'family.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    for variant in args.variants:
        out_dir = root / variant
        if args.skip_existing and variant in summary:
            print(f'== {variant}: cached ==', flush=True)
            continue
        print(f'\n===== {variant} ({FAMILY[variant][1]}, Apache-2.0) =====', flush=True)
        started = time.time()
        record = {'variant': variant, 'backbone': FAMILY[variant][1], 'licence': 'Apache-2.0'}
        try:
            checkpoint, _config, params = make_encoder_checkpoint(
                variant, out_dir, args.image_size, args.adapt_from or None)
            record['params_m'] = round(params / 1e6, 1)

            print('  exporting ONNX tiers + measuring RSS', flush=True)
            tiers = export_tiers(checkpoint, out_dir / 'export', measure_rss=True)
            record['export'] = tiers

            if args.ood_neg:
                target = out_dir / 'ood_neg'
                if not target.exists():
                    shutil.copytree(args.ood_neg, target)

            print('  fitting diagnosis head on int8 embeddings', flush=True)
            build_demo_head(out_dir, args.manifest, max_fit=args.head_max_fit)
            print('  fitting balanced malignant/triage probe', flush=True)
            build_malignant_probe(out_dir, args.manifest, per_class=args.probe_per_class)
            if args.ood_neg:
                print('  fitting OOD lesion gate', flush=True)
                build_ood_gate(out_dir, args.manifest)

            print('  evaluating on the held-out test split', flush=True)
            record['eval'] = evaluate(out_dir, args.manifest, args.eval_n)
            record['minutes'] = round((time.time() - started) / 60, 1)
        except Exception as error:                 # one variant failing must not sink the family
            record['error'] = repr(error)[:400]
            print(f'  FAILED: {error!r}', flush=True)
        summary[variant] = record
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f'  -> {json.dumps({k: v for k, v in record.items() if k != "export"})}', flush=True)

    write_report(summary, root)


def evaluate(artifacts_dir, manifest, limit):
    """Honest held-out evaluation of the served predictor: triage, diagnosis, OOD, abstention."""
    import numpy as np
    from PIL import Image

    from lesnet.data.records import load_manifest
    from lesnet.jepa.serve import JEPADemoPredictor

    predictor = JEPADemoPredictor(artifacts_dir)
    records = [r for r in load_manifest(manifest)
               if r.split == 'test' and r.triage_bucket in ('benign', 'malignant')]
    rng = np.random.default_rng(0)
    sample = [records[i] for i in rng.choice(len(records), min(limit, len(records)), replace=False)]

    def load(path):
        with Image.open(path) as image:
            return np.asarray(image.convert('RGB'))

    abstained = tp = fn = tn = fp = 0
    dx_top1 = dx_top3 = dx_n = 0
    scores, labels = [], []
    for record in sample:
        out = predictor.predict(load(record.image_path))
        if out.get('triage') == 'abstain':
            abstained += 1
            continue
        malignant = record.triage_bucket == 'malignant'
        referred = out.get('triage') in ('refer', 'urgent')
        tp += malignant and referred
        fn += malignant and not referred
        fp += (not malignant) and referred
        tn += (not malignant) and (not referred)
        if out.get('p_malignant') is not None:
            scores.append(float(out['p_malignant']))
            labels.append(int(malignant))
        truth = getattr(record, 'diagnosis', None)
        if truth and truth not in ('benign', 'malignant', 'benign_other'):
            names = [d['label'] for d in (out.get('fine_predictions') or [])]
            dx_n += 1
            dx_top1 += bool(names) and names[0] == truth
            dx_top3 += truth in names[:3]

    metrics = {
        'n': len(sample), 'abstain_rate': round(abstained / max(len(sample), 1), 4),
        'sensitivity': round(tp / max(tp + fn, 1), 4), 'specificity': round(tn / max(tn + fp, 1), 4),
        'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
        'dx_top1': round(dx_top1 / max(dx_n, 1), 4), 'dx_top3': round(dx_top3 / max(dx_n, 1), 4),
        'dx_n': dx_n,
    }
    if len(set(labels)) == 2:
        from sklearn.metrics import roc_auc_score
        metrics['roc_auc'] = round(float(roc_auc_score(labels, scores)), 4)
        metrics['pauc_80tpr'] = round(partial_auc(np.array(labels), np.array(scores), 0.80), 5)

    neg_dir = Path(artifacts_dir) / 'ood_neg'
    negatives = sorted(p for p in neg_dir.glob('*') if p.is_file())[:120] if neg_dir.exists() else []
    if negatives:
        rejected = sum(1 for p in negatives
                       if predictor.predict(load(p)).get('triage') == 'abstain')
        metrics['ood_rejection'] = round(rejected / len(negatives), 4)
    return metrics


def partial_auc(labels, scores, min_tpr):
    """Area under ROC above `min_tpr`, normalised to [0, 1-min_tpr] — the ISIC 2024 metric.

    Plain AUC averages over operating points nobody would deploy; a sensitivity-first triage system
    is only interesting above ~80% recall, so that is the region to score.
    """
    import numpy as np
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    mask = tpr >= min_tpr
    if not mask.any():
        return 0.0
    fpr_region = np.concatenate([[np.interp(min_tpr, tpr, fpr)], fpr[mask]])
    tpr_region = np.concatenate([[min_tpr], tpr[mask]])
    return float(np.trapezoid(tpr_region - min_tpr, fpr_region)) if hasattr(np, 'trapezoid') \
        else float(np.trapz(tpr_region - min_tpr, fpr_region))


def write_report(summary, root):
    rows = ['| variant | backbone | params | served | ONNX MB | RSS MB | fits 512MB | sens | spec '
            '| ROC-AUC | pAUC@80 | dx top-3 | OOD rej |',
            '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for variant, record in summary.items():
        if 'eval' not in record:
            rows.append(f"| {variant} | {record.get('backbone')} | — | — | — | — | — | "
                        f"FAILED: {record.get('error', '')[:40]} | | | | | |")
            continue
        # Report the tier the predictor will actually LOAD (highest fidelity present), not the
        # smallest one — int8 exists on disk but is never served (its parity error is ~5).
        tiers = {t['level']: t for t in record.get('export', {}).get('tiers', [])}
        served = next((tiers[level] for level in ('fp32', 'fp16', 'int8') if level in tiers), {})
        evaluation = record['eval']
        rows.append(
            f"| **{variant}** | {record['backbone']} | {record.get('params_m')}M | "
            f"{served.get('level', '—')} | {served.get('onnx_mb', '—')} | "
            f"{served.get('peak_rss_mb', '—')} | "
            f"{'yes' if served.get('fits_budget') else 'NO'} | "
            f"{evaluation.get('sensitivity')} | {evaluation.get('specificity')} | "
            f"{evaluation.get('roc_auc')} | {evaluation.get('pauc_80tpr')} | "
            f"{evaluation.get('dx_top3')} | {evaluation.get('ood_rejection')} |")
    (root / 'family.md').write_text('\n'.join(rows), encoding='utf-8')
    print('\n' + '\n'.join(rows), flush=True)


if __name__ == '__main__':
    main()
