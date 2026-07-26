"""Linear probe: freeze the encoder, mean-pool patch features, fit a logistic head on labels.

The sole encoder-quality signal, so it must not lie. Manifest mode (preferred) reads the same
leakage-free grouped splits as the triage stack (patient/lesion group_id), evaluates ONLY on the
held-out 'test' split (excluded from SSL pretraining), and reports clinical read-outs:
malignant sensitivity/specificity at a sensitivity-first operating point, ROC-AUC, and
per-Fitzpatrick sensitivity + worst-group gap. Class-folder mode is a leakage-uncontrolled fallback.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from lesnet.jepa.data import IMAGE_EXTENSIONS
from lesnet.jepa.preprocessing import build_transform


class _RecordDataset(Dataset):
    """(tensor, malignant_label, fitzpatrick) from LesionRecords via the eval transform."""

    def __init__(self, records, config, image_size=None):
        self.records = records
        self.transform = build_transform(config, train=False, image_size=image_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        label = 1 if record.triage_bucket == 'malignant' else 0
        fitz = record.fitzpatrick if record.fitzpatrick is not None else -1
        with Image.open(record.image_path) as image:
            return self.transform(image), label, int(fitz)


class ClassFolderDataset(Dataset):
    """Fallback: images under root/<class>/...; malignant label from folder name."""

    def __init__(self, root, config):
        root = Path(root)
        classes = sorted(p.name for p in root.iterdir() if p.is_dir())
        if not classes:
            raise FileNotFoundError(f'No class subfolders under {root}.')
        self.samples = []
        for name in classes:
            label = 1 if 'malignant' in name.lower() else 0
            for path in (root / name).rglob('*'):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, label))
        if not self.samples:
            raise FileNotFoundError(f'No images under {root}/<class>/.')
        self.transform = build_transform(config, train=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image), label, -1


@torch.no_grad()
def _encode(encoder, dataset, device, batch_size, num_workers):
    encoder.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    features, labels, fitz = [], [], []
    for images, batch_labels, batch_fitz in loader:
        tokens = encoder(images.to(device))                 # (B, N, D)
        features.append(tokens.mean(dim=1).cpu().numpy())   # mean-pool
        labels.append(np.asarray(batch_labels))
        fitz.append(np.asarray(batch_fitz))
    return np.concatenate(features), np.concatenate(labels), np.concatenate(fitz)


def _sens_spec(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float('nan')
    spec = tn / (tn + fp) if (tn + fp) else float('nan')
    return sens, spec


def _threshold_for_sensitivity(y_true, proba, target):
    """Highest threshold that still achieves >= target sensitivity (maximises specificity)."""
    candidates = np.concatenate([[0.0], np.unique(proba), [1.0]])
    valid = [t for t in candidates if _sens_spec(y_true, proba, t)[0] >= target]
    return float(max(valid)) if valid else 0.0


def _fairness(y_true, proba, fitz, threshold, min_positives=3):
    """Per-Fitzpatrick sensitivity at the operating point + worst-group gap."""
    per_group = {}
    for group in sorted(set(int(f) for f in fitz)):
        if group < 0:
            continue
        mask = fitz == group
        if int(y_true[mask].sum()) < min_positives:
            continue
        sens, _ = _sens_spec(y_true[mask], proba[mask], threshold)
        per_group[str(group)] = round(float(sens), 4)
    gap = round(max(per_group.values()) - min(per_group.values()), 4) if len(per_group) > 1 else None
    return per_group, gap


def linear_probe(train_feats, train_labels, test_feats, test_labels, test_fitz, target_sensitivity):
    classifier = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced')
    classifier.fit(train_feats, train_labels)
    train_proba = classifier.predict_proba(train_feats)[:, 1]
    test_proba = classifier.predict_proba(test_feats)[:, 1]

    threshold = _threshold_for_sensitivity(train_labels, train_proba, target_sensitivity)  # set on TRAIN
    sens, spec = _sens_spec(test_labels, test_proba, threshold)
    per_group, gap = _fairness(test_labels, test_proba, test_fitz, threshold)
    metrics = {
        'n_train': int(len(train_labels)), 'n_test': int(len(test_labels)),
        'accuracy': round(float(accuracy_score(test_labels, (test_proba >= threshold))), 4),
        'operating_threshold': round(threshold, 4),
        'target_sensitivity': target_sensitivity,
        'sensitivity': round(float(sens), 4), 'specificity': round(float(spec), 4),
        'fitzpatrick_sensitivity': per_group, 'worst_group_gap': gap,
    }
    if len(np.unique(test_labels)) == 2:
        metrics['roc_auc'] = round(float(roc_auc_score(test_labels, test_proba)), 4)
    else:
        metrics['roc_auc'] = None
    return metrics


def quick_probe(encoder, manifest_path, config, device=None, image_size=None, subset=3000,
                batch_size=64, num_workers=0):
    """Cheap in-training encoder-quality read-out (ROC-AUC + sens/spec) on a fixed random subset.

    Used as the model-selection and early-stopping signal during pretraining: the SSL loss keeps
    creeping down long after the representation stops getting more useful, so selecting on loss
    quietly ships a worse encoder. Same leakage discipline as run_probe — fit on train/val rows,
    score on held-out test rows.
    """
    from lesnet.data.records import load_manifest
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    records = load_manifest(manifest_path)
    rng = np.random.default_rng(config.seed)

    def sample(rows):
        rows = [r for r in rows if r.triage_bucket in ('benign', 'malignant')]
        if len(rows) <= subset:
            return rows
        return [rows[i] for i in rng.choice(len(rows), subset, replace=False)]

    train_rows = sample([r for r in records if r.split in ('train', 'val')])
    test_rows = sample([r for r in records if r.split == 'test'])
    if not train_rows or not test_rows:
        raise ValueError(f'{manifest_path} has no usable train/test rows for the in-training probe.')

    was_training = encoder.training
    train_ds = _RecordDataset(train_rows, config, image_size=image_size)
    test_ds = _RecordDataset(test_rows, config, image_size=image_size)
    train_feats, train_labels, _ = _encode(encoder, train_ds, device, batch_size, num_workers)
    test_feats, test_labels, test_fitz = _encode(encoder, test_ds, device, batch_size, num_workers)
    encoder.train(was_training)
    return linear_probe(train_feats, train_labels, test_feats, test_labels, test_fitz,
                        config.probe_target_sensitivity)


def run_probe(encoder, source, config, device=None, batch_size=64, num_workers=0):
    """source: a manifest.csv path (preferred, leakage-free) or a class-subfolder directory."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    target = config.probe_target_sensitivity

    if str(source).endswith('.csv'):
        from lesnet.data.records import load_manifest
        from lesnet.data.splits import assert_no_group_leakage, grouped_train_val_test
        records = load_manifest(source)
        splits = [r.split for r in records]
        if all(s in ('train', 'val', 'test') for s in splits) and 'test' in splits:
            train_records = [r for r in records if r.split in ('train', 'val')]
            test_records = [r for r in records if r.split == 'test']
        else:  # no usable splits in the manifest: derive a group-safe split now
            groups = [r.group_id for r in records]
            tr, va, te = grouped_train_val_test(groups)
            assert_no_group_leakage(groups, tr, va, te)
            train_records = [records[i] for i in list(tr) + list(va)]
            test_records = [records[i] for i in te]
        train_ds, test_ds = _RecordDataset(train_records, config), _RecordDataset(test_records, config)
    else:
        full = ClassFolderDataset(source, config)
        n_test = max(int(len(full) * 0.3), 1)
        generator = torch.Generator().manual_seed(config.seed)
        perm = torch.randperm(len(full), generator=generator).tolist()
        test_idx, train_idx = set(perm[:n_test]), set(perm[n_test:])
        train_ds = torch.utils.data.Subset(full, sorted(train_idx))
        test_ds = torch.utils.data.Subset(full, sorted(test_idx))

    train_feats, train_labels, _ = _encode(encoder, train_ds, device, batch_size, num_workers)
    test_feats, test_labels, test_fitz = _encode(encoder, test_ds, device, batch_size, num_workers)
    metrics = linear_probe(train_feats, train_labels, test_feats, test_labels, test_fitz, target)
    metrics['source'] = 'manifest' if str(source).endswith('.csv') else 'class_folder'
    return metrics
