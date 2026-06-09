"""Train the triage model end-to-end (paper §5).

Smoke (CPU, synthetic):  python commands/run_train_triage.py --smoke
Real (GPU, manifest):    python commands/run_train_triage.py --manifest data/manifest.csv \
                             --artifacts artifacts --epochs 30
"""
import argparse
import os
import tempfile

from lesnet.ml.config import PipelineConfig
from lesnet.ml.datasets import DatasetConfig, assign_splits, load_manifest, save_manifest
from lesnet.ml.synthetic import make_synthetic_records
from lesnet.ml.training import train


def _split_records(records):
    by_split = {'train': [], 'val': [], 'test': []}
    for record in records:
        by_split.get(record.split, by_split['train']).append(record)
    return by_split['train'], by_split['val'], by_split['test']


def main():
    parser = argparse.ArgumentParser(description="Train the LesNet triage model.")
    parser.add_argument('--manifest', help="Path to a manifest CSV from run_build_dataset.py.")
    parser.add_argument('--artifacts', default='artifacts')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--image-size', type=int, default=384)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--backbone', default='efficientnetv2s', choices=['efficientnetv2s', 'tiny'])
    parser.add_argument('--no-pretrained', action='store_true', help="Random init (faster, CPU).")
    parser.add_argument('--until-target', action='store_true',
                        help="Train until all metric targets are in range (up to --max-epochs).")
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--cache', action='store_true', help="Disk-cache preprocessed images across epochs.")
    parser.add_argument('--smoke', action='store_true', help="Tiny synthetic CPU end-to-end run.")
    args = parser.parse_args()

    if args.smoke:
        config = PipelineConfig(
            image_size=(64, 64), backbone='tiny', pretrained=False, batch_size=16,
            epochs=4, shared_units=64, artifacts_dir=args.artifacts, smoke=True)
        records = make_synthetic_records(tempfile.mkdtemp(prefix='lesnet_smoke_'), per_class=40)
        records = assign_splits(records, DatasetConfig(test_size=0.2, val_size=0.2, seed=config.seed))
        os.makedirs(args.artifacts, exist_ok=True)
        save_manifest(records, os.path.join(args.artifacts, 'smoke_manifest.csv'))
    else:
        if not args.manifest:
            parser.error("--manifest is required unless --smoke is set.")
        config = PipelineConfig(
            image_size=(args.image_size, args.image_size), batch_size=args.batch_size,
            epochs=args.epochs, artifacts_dir=args.artifacts,
            backbone=args.backbone, pretrained=not args.no_pretrained,
            shared_units=64 if args.backbone == 'tiny' else 256,
            train_until_target=args.until_target, max_epochs=args.max_epochs,
            cache_dataset=args.cache)
        records = load_manifest(args.manifest)

    train_records, val_records, test_records = _split_records(records)
    if not val_records:
        val_records = train_records
    _, bundle = train(config, train_records, val_records)
    print(f"Trained. Artifacts in {config.artifacts_dir}.")
    print(f"  operating threshold: {bundle['thresholds']['operating_threshold']:.4f}")
    print(f"  temperature: {bundle['calibration']['temperature']:.4f}")
    print(f"  fine classes: {bundle['n_fine']}")


if __name__ == '__main__':
    main()
