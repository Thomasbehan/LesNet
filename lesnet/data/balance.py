"""Fair, group-safe balancing of the decision-critical buckets (stage 1).

Goal: a ~1:1 benign:malignant set where no single diagnosis dominates its bucket. Two
invariants make the metrics trustworthy and the model fair:
  * group-safe — we drop whole patient/lesion groups, never split one across the cut;
  * fairness-aware — when downsampling we retain scarcer high-Fitzpatrick (darker-skin)
    groups first, so balancing doesn't wash out skin-tone diversity.
'not_sure'/suspicious is kept as-is (not forced into the ratio).
"""
import zlib


def bucket_counts(records):
    counts = {}
    for record in records:
        counts[record.triage_bucket] = counts.get(record.triage_bucket, 0) + 1
    return counts


def _group_order_key(seed, group_id, records):
    """Sort key: retain higher max-Fitzpatrick groups first; deterministic tie-break."""
    max_fitzpatrick = max((record.fitzpatrick or 0) for record in records)
    tiebreak = zlib.crc32(f"{seed}:{group_id}".encode())
    return (-max_fitzpatrick, tiebreak)


def _select_groups(records, target, seed):
    """Keep whole groups (no splitting) up to ``target`` rows, fairness-prioritised."""
    if target <= 0:
        return []
    if len(records) <= target:
        return list(records)
    groups = {}
    for record in records:
        groups.setdefault(record.group_id, []).append(record)
    ordered = sorted(groups.items(), key=lambda item: _group_order_key(seed, item[0], item[1]))
    kept = []
    for _group_id, group_records in ordered:
        if len(kept) + len(group_records) <= target:
            kept.extend(group_records)
    return kept


def _balance_bucket(records, target, cap_fraction, seed):
    """Pick ~``target`` rows, limiting any single diagnosis to ``cap_fraction`` of the bucket
    while there's diversity to spare, then water-filling from the rest to reach target.

    So a bucket with one diagnosis (e.g. all-melanoma malignant) still reaches target, but a
    mixed bucket (benign) won't let one diagnosis (nevus) dominate.
    """
    if target <= 0:
        return []
    if len(records) <= target:
        return list(records)
    by_diagnosis = {}
    for record in records:
        by_diagnosis.setdefault(record.diagnosis, []).append(record)
    cap = max(1, int(cap_fraction * target))

    selected = []
    for diagnosis_records in by_diagnosis.values():
        selected.extend(_select_groups(diagnosis_records, min(cap, target), seed))
    if len(selected) < target:
        chosen = {id(record) for record in selected}
        leftovers = [record for record in records if id(record) not in chosen]
        selected.extend(_select_groups(leftovers, target - len(selected), seed))
    if len(selected) > target:
        selected = _select_groups(selected, target, seed)
    return selected


def _targets(counts, ratio):
    """benign/malignant keep-targets for the desired benign:malignant ratio."""
    benign, malignant = counts.get('benign', 0), counts.get('malignant', 0)
    if benign == 0 or malignant == 0:
        return {'benign': benign, 'malignant': malignant}
    desired_benign = round(ratio * malignant)
    if benign > desired_benign:
        return {'benign': desired_benign, 'malignant': malignant}
    return {'benign': benign, 'malignant': round(benign / ratio)}


def balance(records, ratio=1.0, per_diagnosis_cap_fraction=0.6,
            buckets=('benign', 'malignant'), seed=42):
    by_bucket = {}
    for record in records:
        by_bucket.setdefault(record.triage_bucket, []).append(record)

    result = [record for bucket, recs in by_bucket.items()
              if bucket not in buckets for record in recs]   # untouched buckets (e.g. not_sure)

    targets = _targets(bucket_counts(records), ratio)
    for bucket in buckets:
        recs = by_bucket.get(bucket, [])
        target = targets.get(bucket, len(recs))
        result.extend(_balance_bucket(recs, target, per_diagnosis_cap_fraction, seed))
    return result
