"""Build the unified, grouped-split dataset manifest (paper §5.3, stage 1).

Fast iteration:   python commands/run_build_dataset.py --mode sample --sample-size 200 \
                      --datasets isic pad_ufes_20 --isic-root data/isic --pad-ufes-root data/pad_ufes
Full training:    python commands/run_build_dataset.py --mode full --datasets isic pad_ufes_20 fitzpatrick17k ddi ...
"""
import argparse
from collections import Counter

from lesnet.ml.datasets import DatasetConfig, assign_splits, build_manifest, save_manifest


def main():
    parser = argparse.ArgumentParser(description="Build the LesNet dataset manifest with grouped splits.")
    parser.add_argument('--mode', choices=['sample', 'full'], default='sample')
    parser.add_argument('--sample-size', type=int, default=200, help="Rows per dataset in sample mode.")
    parser.add_argument('--datasets', nargs='+', default=['isic'],
                        choices=['isic', 'pad_ufes_20', 'fitzpatrick17k', 'ddi'])
    parser.add_argument('--isic-root')
    parser.add_argument('--pad-ufes-root')
    parser.add_argument('--fitzpatrick17k-root')
    parser.add_argument('--ddi-root')
    parser.add_argument('--output', default='data/manifest.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = DatasetConfig(
        mode=args.mode,
        sample_size_per_dataset=args.sample_size,
        datasets=tuple(args.datasets),
        isic_root=args.isic_root,
        pad_ufes_root=args.pad_ufes_root,
        fitzpatrick17k_root=args.fitzpatrick17k_root,
        ddi_root=args.ddi_root,
        seed=args.seed,
    )

    records = assign_splits(build_manifest(config), config)
    save_manifest(records, args.output)

    by_split = Counter(record.split for record in records)
    by_dataset = Counter(record.source_dataset for record in records)
    print(f"Wrote {len(records)} records to {args.output} (mode={args.mode}).")
    print(f"  by split:   {dict(by_split)}")
    print(f"  by dataset: {dict(by_dataset)}")


if __name__ == '__main__':
    main()
