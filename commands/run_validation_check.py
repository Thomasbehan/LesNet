"""Spot-check inference on the held-out validation set: true vs predicted triage."""
import argparse
import csv

import numpy as np
from PIL import Image

from lesnet.ml.datasets import LesionRecord
from lesnet.ml.inference import TriagePredictor


def main():
    parser = argparse.ArgumentParser(description="Run inference over a labelled validation folder.")
    parser.add_argument('--artifacts', default='models/triage')
    parser.add_argument('--labels', default='validation_samples/labels.csv')
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.labels, encoding='utf-8')))
    predictor = TriagePredictor(args.artifacts)

    header = f"{'fine_label':30} {'true':10} {'predicted':10} {'p_mal':6} valid"
    print(header)
    print('-' * len(header))

    agree = 0
    for row in rows:
        image = np.asarray(Image.open(row['image']).convert('RGB'))
        record = LesionRecord(
            image_path=row['image'], source_dataset='val', raw_label='unknown', group_id='val',
            fitzpatrick=None, anatomical_site=row['site'] or None,
            age=float(row['age']) if row['age'] else None, sex=row['sex'] or None)
        result = predictor.predict(image, record)
        p_malignant = result.get('p_malignant')
        p_text = f"{p_malignant:.2f}" if p_malignant is not None else '   -'
        flagged = result['triage'] in ('refer', 'urgent')
        is_malignant = row['triage'] == 'malignant'
        agree += int(flagged == is_malignant)
        print(f"{row['fine_label'][:30]:30} {row['triage']:10} {result['triage']:10} "
              f"{p_text:6} {result['valid_image']}")

    print('-' * len(header))
    print(f"malignant-vs-benign agreement: {agree}/{len(rows)}")


if __name__ == '__main__':
    main()
