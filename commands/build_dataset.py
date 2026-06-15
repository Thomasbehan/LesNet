"""Stage 1 — source, canonicalise, quality-gate, balance, and sort a training dataset.

  # malignant-maximal balanced build from ISIC (+ any sources present on disk):
  python commands/build_dataset.py --sources isic pad_ufes_20 fitzpatrick17k ddi --dest data/dataset

ISIC is sourced malignant-maximally: page the whole archive's metadata (cheap), then download
images only for all malignant + suspicious + a benign sample matched to the target ratio.
Other sources auto-download where the licence allows (DDI must be placed on disk manually).
"""
import argparse
import json
import os

from lesnet.data import balance, sort
from lesnet.data.config import SourcingConfig
from lesnet.data.pipeline import run
from lesnet.data.sources import isic


def _isic_malignant_maximal_download(root, config):
    """Populate ``root`` with ISIC metadata + a malignant-maximal, ratio-matched image set."""
    resolution = 'full' if config.full_resolution else 'thumbnail_256'
    isic.collect_metadata(root, resolution=resolution, limit=config.sample_limit)
    url_by_id = isic.url_map(root)

    annotated, _ = sort.annotate(isic.parse(root, config.sample_limit))
    by_bucket = {'benign': [], 'not_sure': [], 'malignant': []}
    for record in annotated:
        by_bucket[record.triage_bucket].append(record)

    benign_target = round(config.balance_ratio * len(by_bucket['malignant']))
    benign_keep = balance._select_groups(by_bucket['benign'], benign_target, config.seed)
    selected = by_bucket['malignant'] + by_bucket['not_sure'] + benign_keep
    selected_ids = {os.path.splitext(os.path.basename(r.image_path))[0] for r in selected}

    download_set = {isic_id: url_by_id[isic_id] for isic_id in selected_ids if isic_id in url_by_id}
    print(f"ISIC: {len(annotated)} labelled · malignant={len(by_bucket['malignant'])} "
          f"not_sure={len(by_bucket['not_sure'])} benign(kept)={len(benign_keep)} "
          f"-> downloading {len(download_set)} images")
    isic.download_images(root, download_set)


def main():
    parser = argparse.ArgumentParser(description="Build the LesNet triage dataset (stage 1).")
    parser.add_argument('--sources', nargs='+', default=['isic'],
                        choices=['isic', 'pad_ufes_20', 'fitzpatrick17k', 'ddi'])
    parser.add_argument('--dest', default='data/dataset')
    parser.add_argument('--raw-dir', default='data/raw')
    parser.add_argument('--isic-root')
    parser.add_argument('--pad-ufes-root')
    parser.add_argument('--fitzpatrick17k-root')
    parser.add_argument('--ddi-root')
    parser.add_argument('--sample-limit', type=int, default=None)
    parser.add_argument('--ratio', type=float, default=1.0, help="benign:malignant target.")
    parser.add_argument('--no-dedupe', action='store_true')
    parser.add_argument('--thumbnails', action='store_true', help="ISIC thumbnails (faster, lower-res).")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    root_arg = {'isic': args.isic_root, 'pad_ufes_20': args.pad_ufes_root,
                'fitzpatrick17k': args.fitzpatrick17k_root, 'ddi': args.ddi_root}
    roots = {name: root_arg[name] for name in args.sources if root_arg.get(name)}
    config = SourcingConfig(
        sources=tuple(args.sources), roots=roots, raw_dir=args.raw_dir, dest=args.dest,
        sample_limit=args.sample_limit, full_resolution=not args.thumbnails,
        dedupe=not args.no_dedupe, balance_ratio=args.ratio, seed=args.seed)

    if 'isic' in config.sources:
        isic_root = config.roots.get('isic') or os.path.join(config.raw_dir, 'isic')
        if not (os.path.isdir(isic_root) and os.listdir(isic_root)):
            _isic_malignant_maximal_download(isic_root, config)

    _records, report = run(config)
    print(json.dumps({key: report[key] for key in
                      ('final_total', 'by_bucket', 'benign_to_malignant_ratio', 'by_split')}, indent=2))
    print(f"Dataset + manifest written to {config.dest}")


if __name__ == '__main__':
    main()
