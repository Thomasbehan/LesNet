"""Package the built model family into per-variant release archives + a manifest.

Model weights are gitignored (correctly — they are large binaries), so they ship as GitHub release
assets instead. Each archive is self-contained: the ONNX encoder actually served, the fitted heads,
the OOD gate, and the config/metadata needed to reconstruct the predictor.

    python commands/package_jepa_release.py --family artifacts/family --out dist/release
"""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

# What a consumer needs to run JEPADemoPredictor(<extracted dir>). The torch checkpoint is included
# only for the smaller variants — it is redundant for inference and dominates the archive size.
SERVE_FILES = ['jepa_config.json', 'demo_meta.json', 'diagnosis_head.joblib', 'ood.joblib']


def sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Package the model family for a GitHub release.")
    parser.add_argument('--family', default='artifacts/family')
    parser.add_argument('--out', default='dist/release')
    parser.add_argument('--include-torch', action='store_true',
                        help="Also ship context_encoder.pt (large; ONNX is enough for inference).")
    args = parser.parse_args()

    family_dir = Path(args.family)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((family_dir / 'family.json').read_text())

    assets = []
    for variant, record in summary.items():
        source = family_dir / variant
        if 'eval' not in record or not source.exists():
            print(f'skip {variant}: not built', flush=True)
            continue
        # ship the precision the predictor will actually load (highest fidelity that exists)
        onnx = next((source / 'export' / f'encoder_{tier}.onnx' for tier in ('fp32', 'fp16', 'int8')
                     if (source / 'export' / f'encoder_{tier}.onnx').exists()), None)
        if onnx is None:
            print(f'skip {variant}: no exported encoder', flush=True)
            continue

        archive = out_dir / f'lesnet-jepa-{variant}.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            tar.add(onnx, arcname=f'{variant}/export/{onnx.name}')
            # Models past ~2 GB of protobuf spill their tensors into a sibling .data file; the
            # .onnx references it BY NAME, so shipping the graph without it yields a file that
            # loads and then fails at inference.
            for extra in onnx.parent.glob(onnx.name + '.data'):
                tar.add(extra, arcname=f'{variant}/export/{extra.name}')
            for name in SERVE_FILES:
                if (source / name).exists():
                    tar.add(source / name, arcname=f'{variant}/{name}')
            if args.include_torch and (source / 'context_encoder.pt').exists():
                tar.add(source / 'context_encoder.pt', arcname=f'{variant}/context_encoder.pt')
        size_mb = round(archive.stat().st_size / 1e6, 1)
        assets.append({
            'variant': variant, 'file': archive.name, 'size_mb': size_mb,
            'sha256': sha256(archive), 'backbone': record.get('backbone'),
            'licence': record.get('licence'), 'params_m': record.get('params_m'),
            'precision': onnx.stem.rsplit('_', 1)[-1], 'metrics': record.get('eval'),
        })
        print(f'{variant}: {archive.name} {size_mb} MB', flush=True)

    (out_dir / 'assets.json').write_text(json.dumps(assets, indent=2))
    lines = ['# LesNet JEPA model family — release assets', '',
             '| asset | variant | backbone | params | precision | size | sha256 |',
             '|---|---|---|---|---|---|---|']
    for asset in assets:
        lines.append(f"| `{asset['file']}` | {asset['variant']} | {asset['backbone']} | "
                     f"{asset['params_m']}M | {asset['precision']} | {asset['size_mb']} MB | "
                     f"`{asset['sha256'][:16]}…` |")
    lines += ['', 'Extract and point the predictor at the directory:', '',
              '```bash', 'tar xzf lesnet-jepa-small.tar.gz',
              'export LESNET_JEPA_ARTIFACTS=$PWD/small', 'pserve development.ini', '```']
    (out_dir / 'RELEASE_ASSETS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
