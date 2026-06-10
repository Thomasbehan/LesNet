"""Run triage inference on a single image (paper §7).

  python commands/run_triage_inference.py --artifacts artifacts --image path/to/lesion.jpg \
      --age 60 --sex male --site torso
"""
import argparse
import json

import numpy as np
from PIL import Image

from lesnet.ml.datasets import LesionRecord
from lesnet.ml.inference import TriagePredictor


def main():
    parser = argparse.ArgumentParser(description="Triage inference for one image.")
    parser.add_argument('--artifacts', default='artifacts')
    parser.add_argument('--image', required=True)
    parser.add_argument('--age', type=float)
    parser.add_argument('--sex')
    parser.add_argument('--site')
    parser.add_argument('--fitzpatrick', type=int)
    parser.add_argument('--tta', action='store_true')
    args = parser.parse_args()

    record = LesionRecord(
        image_path=args.image, source_dataset='query', raw_label='unknown', group_id='query',
        fitzpatrick=args.fitzpatrick, anatomical_site=args.site, age=args.age, sex=args.sex)
    image = np.asarray(Image.open(args.image).convert('RGB'))

    predictor = TriagePredictor(args.artifacts, use_test_time_augmentation=args.tta)
    print(json.dumps(predictor.predict(image, record), indent=2))


if __name__ == '__main__':
    main()
