"""Evaluate one or more JEPA encoders and write a report into docs/ (see docs/jepa-world-model.md).

Runs the leakage-free linear probe on the manifest's held-out grouped test split for each model
(sensitivity-first operating point + per-Fitzpatrick fairness), aggregates the deployment
quantisation tiers (size + measured 512 MB fit), and writes docs/jepa-evaluation.md + .json.

Usage:
  python commands/run_jepa_evaluation.py --manifest data/dataset/manifest.csv \
      --artifacts artifacts/jepa/vit_large  [more dirs...]
"""
import argparse
import json
from pathlib import Path

from lesnet.jepa.export import load_encoder
from lesnet.jepa.probe import run_probe

DISCLAIMER = ('Leakage-free linear-probe evaluation on the held-out grouped test split '
              '(patient/lesion `group_id`), sensitivity-first operating point. LesNet is a '
              'research/triage tool, **not** a diagnostic device.')


def evaluate_one(artifacts_dir, manifest, batch_size):
    artifacts_dir = Path(artifacts_dir)
    encoder, config = load_encoder(artifacts_dir / 'context_encoder.pt')
    if manifest:
        metrics = run_probe(encoder, manifest, config, batch_size=batch_size)
    else:  # aggregate mode: reuse the probe metrics computed at training time (e.g. on Kaggle)
        metrics_path = artifacts_dir / 'probe_metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    export = None
    for name in ('tiers.json', 'report.json'):
        candidate = artifacts_dir / 'export' / name
        if candidate.exists():
            export = json.loads(candidate.read_text())
            break
    return {'artifacts': str(artifacts_dir), 'encoder': config.encoder,
            'image_size': config.image_size, 'probe': metrics, 'export': export}


def to_markdown(results):
    lines = ['# LesNet JEPA — evaluation', '', DISCLAIMER, '',
             '## Representation quality (per model)', '',
             '| model | image | n_test | sensitivity | specificity | ROC-AUC | worst-Fitz gap |',
             '|---|---|---|---|---|---|---|']
    for r in results:
        p = r['probe']
        lines.append(f"| {r['encoder']} | {r['image_size']} | {p.get('n_test')} | "
                     f"{p.get('sensitivity')} | {p.get('specificity')} | {p.get('roc_auc')} | "
                     f"{p.get('worst_group_gap')} |")
    lines += ['', '## Deployment tiers (size + measured 512 MB fit)', '',
              '| model | tier | ONNX MB | peak RSS MB | fits 512 MB | parity |',
              '|---|---|---|---|---|---|']
    for r in results:
        export = r['export'] or {}
        if export.get('tiers'):
            for tier in export['tiers']:
                lines.append(f"| {r['encoder']} | {tier['level']} | {tier['onnx_mb']} | "
                             f"{tier.get('peak_rss_mb')} | {tier.get('fits_budget')} | "
                             f"{tier.get('parity_max_abs_diff')} |")
        elif export:
            lines.append(f"| {r['encoder']} | int8 | {export.get('int8_onnx_mb')} | "
                         f"{export.get('peak_rss_mb')} | {export.get('fits_budget')} | "
                         f"{export.get('int8_parity_max_abs_diff')} |")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description="Evaluate JEPA encoders and write docs/.")
    parser.add_argument('--manifest', help="manifest.csv with grouped splits; omit to aggregate "
                        "each artifact dir's existing probe_metrics.json (e.g. pulled from Kaggle).")
    parser.add_argument('--artifacts', nargs='+', required=True, help="One or more artifact dirs.")
    parser.add_argument('--out', default='docs/jepa-evaluation.md')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    results = [evaluate_one(a, args.manifest, args.batch_size) for a in args.artifacts]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_markdown(results), encoding='utf-8')
    out.with_suffix('.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f"Wrote {out} and {out.with_suffix('.json')} ({len(results)} model(s)).")


if __name__ == '__main__':
    main()
