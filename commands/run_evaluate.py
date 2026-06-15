"""Evaluate a trained triage model and write a report + model card (paper §6).

  python commands/run_evaluate.py --manifest data/manifest.csv --artifacts artifacts
"""
import argparse
import json


from lesnet.ml import artifacts as artifact_io
from lesnet.ml.calibration import softmax
from lesnet.ml.config import PipelineConfig
from lesnet.ml.data_loader import filter_valid, make_dataset
from lesnet.ml.evaluation import build_report, write_model_card
from lesnet.ml.features import normalize_site
from lesnet.ml.model import triage_logits_model
from lesnet.data.records import load_manifest

import tensorflow as tf


def _age_band(age):
    if not age:
        return 'unknown'
    return f"{int(age) // 20 * 20}s"


def _subgroups(records):
    return {
        'fitzpatrick': [str(record.fitzpatrick) if record.fitzpatrick else 'unknown' for record in records],
        'site': [normalize_site(record.anatomical_site) for record in records],
        'age_band': [_age_band(record.age) for record in records],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the LesNet triage model.")
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--artifacts', default='artifacts')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    args = parser.parse_args()

    bundle = artifact_io.load_bundle(args.artifacts)
    model = tf.keras.models.load_model(artifact_io.model_path(args.artifacts), compile=False)
    config = PipelineConfig(**{key: bundle['config'][key] for key in PipelineConfig().__dict__})

    records = [record for record in load_manifest(args.manifest) if record.split == args.split]
    records = filter_valid(records)
    fine_vocabulary = bundle['label_maps']['fine_vocabulary']
    dataset, _, y_triage = make_dataset(records, config, fine_vocabulary, training=False)

    logits = triage_logits_model(model).predict(dataset, verbose=0)
    probabilities = softmax(logits / bundle['calibration']['temperature'])

    report = build_report(
        y_triage, probabilities, subgroups=_subgroups(records),
        target_sensitivity=bundle['thresholds']['target_sensitivity'],
        q_hat=bundle['conformal']['q_hat'])

    with open(f"{args.artifacts}/evaluation_report.json", 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    write_model_card(report, bundle, f"{args.artifacts}/model_card.md")

    print(f"Sensitivity {report['sensitivity']:.3f} · Specificity {report['specificity']:.3f} "
          f"· ROC-AUC {report['roc_auc']:.3f} · ECE {report['ece']:.3f}")
    print(f"Fairness gate passed: {report['fairness_gate']['passed']}")


if __name__ == '__main__':
    main()
