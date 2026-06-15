"""Stage-1 orchestrator: raw sources -> clean, balanced, sorted, leakage-safe dataset.

process():  records (images on disk) -> annotate -> quality gate -> dedupe -> balance ->
            materialise into folders -> grouped splits -> manifest + report. (No network.)
ingest():   per source, ensure raw present (auto-download where allowed) and parse to records.
run():      ingest then process.
"""
import json
import os

from lesnet.data import balance as balance_module
from lesnet.data import manifest as manifest_module
from lesnet.data import quality, sort
from lesnet.data.canonical import unmapped_diagnoses
from lesnet.data.sources.registry import get_source


def _distribution(records, attribute):
    counts = {}
    for record in records:
        key = str(getattr(record, attribute))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def build_report(stats, kept):
    report = dict(stats)
    report['final_total'] = len(kept)
    report['by_bucket'] = balance_module.bucket_counts(kept)
    report['by_diagnosis'] = _distribution(kept, 'diagnosis')
    report['by_source'] = _distribution(kept, 'source_dataset')
    report['by_fitzpatrick'] = _distribution(kept, 'fitzpatrick')
    report['by_split'] = _distribution(kept, 'split')
    benign, malignant = report['by_bucket'].get('benign', 0), report['by_bucket'].get('malignant', 0)
    report['benign_to_malignant_ratio'] = round(benign / malignant, 3) if malignant else None
    return report


def process(records, config):
    os.makedirs(config.dest, exist_ok=True)
    stats = {'ingested': len(records)}

    annotated, unmappable = sort.annotate(records)
    stats['dropped_unmappable'] = len(unmappable)

    # Separate "image not on disk" (expected when a source is fetched selectively) from
    # "downloaded but corrupt/too small" so the report is honest about each.
    present = [record for record in annotated if os.path.exists(record.image_path)]
    stats['dropped_no_image'] = len(annotated) - len(present)
    quality_kept, low_quality = quality.filter_quality(present, config.min_image_pixels)
    stats['dropped_low_quality'] = len(low_quality)

    if config.dedupe:
        deduped, duplicates = quality.dedupe(quality_kept, config.phash_distance)
        stats['dropped_duplicate'] = len(duplicates)
    else:
        deduped, stats['dropped_duplicate'] = quality_kept, 0

    balanced = balance_module.balance(
        deduped, config.balance_ratio, config.per_diagnosis_cap_fraction,
        config.balance_buckets, config.seed)

    materialised, missing = sort.materialise(balanced, config.dest)
    stats['dropped_missing_image'] = len(missing)

    manifest_module.assign_splits(materialised, config.test_size, config.val_size, config.seed)
    manifest_module.write(materialised, os.path.join(config.dest, 'manifest.csv'))

    report = build_report(stats, materialised)
    with open(os.path.join(config.dest, 'report.json'), 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    with open(os.path.join(config.dest, 'unmapped_diagnoses.txt'), 'w', encoding='utf-8') as handle:
        handle.write("\n".join(unmapped_diagnoses(record.raw_label for record in records)) + "\n")
    return materialised, report


def ingest(config):
    """Ensure each source's raw data is present (auto-download where possible) and parse it.

    A source whose data is unavailable (licence-gated, not downloaded, or a failed fetch) is
    skipped with a warning rather than aborting the whole build.
    """
    records = []
    for name in config.sources:
        spec = get_source(name)
        root = config.roots.get(name) or os.path.join(config.raw_dir, name)
        try:
            metadata_present = os.path.isdir(root) and os.listdir(root)
            if not metadata_present and spec.download is not None:
                spec.download(root, config.sample_limit)
            source_records = spec.parse(root, config.sample_limit)
        except Exception as error:  # noqa: BLE001 - one unavailable source must not kill the build
            print(f"Skipping source '{name}': {error}")
            continue
        records.extend(source_records)
    return records


def run(config):
    return process(ingest(config), config)
