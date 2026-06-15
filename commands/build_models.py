"""Train the full 4.5.0 model family end to end (paper §5, §7).

  python commands/build_models.py --manifest data/dataset/manifest.csv --out models/4.5.0 --until-target

1. Train the teacher M4.5XL (EfficientNetV2-L @ 512).
2. Distil students M4.5L / M4.5m / M4.5s from the teacher (cross-resolution: the teacher is
   rebuilt at each student's input size and its convolutional weights transferred).
3. int8-quantise M4.5s to TFLite for the <500 MB live demo.
Each model's artifacts (model.keras + artifacts.json) land in models/4.5.0/<name>/; package
them for release with commands/package_model.py.
"""
import argparse
import os
from dataclasses import replace

from lesnet.data.records import load_manifest
from lesnet.data.taxonomy import build_fine_vocabulary
from lesnet.ml import quantize, variants
from lesnet.ml.config import PipelineConfig
from lesnet.ml.data_loader import filter_valid, make_dataset
from lesnet.ml.features import METADATA_DIM
from lesnet.ml.model import build_triage_model
from lesnet.ml.training import distill, train


def _split(records):
    buckets = {'train': [], 'val': [], 'test': []}
    for record in records:
        buckets.get(record.split, buckets['train']).append(record)
    return buckets['train'], buckets['val'] or buckets['train']


def main():
    parser = argparse.ArgumentParser(description="Train the LesNet 4.5.0 model family.")
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', default='models/4.5.0')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--until-target', action='store_true', help="Metric-gated training to targets.")
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--no-pretrained', action='store_true')
    args = parser.parse_args()

    train_records, val_records = _split(load_manifest(args.manifest))
    fine_vocabulary = build_fine_vocabulary(train_records)
    n_fine = max(len(fine_vocabulary), 1)
    base = PipelineConfig(epochs=args.epochs, train_until_target=args.until_target,
                          max_epochs=args.max_epochs, pretrained=not args.no_pretrained)

    print("=== Teacher: M4.5XL ===")
    teacher_config = variants.config_for(variants.TEACHER, base,
                                         artifacts_dir=os.path.join(args.out, variants.TEACHER))
    teacher_model, _ = train(teacher_config, train_records, val_records)
    teacher_weights = os.path.join(args.out, 'teacher.weights.h5')
    teacher_model.save_weights(teacher_weights)

    for name in variants.STUDENTS:
        print(f"=== Student: {name} (distilling from {variants.TEACHER}) ===")
        student_config = variants.config_for(name, base, artifacts_dir=os.path.join(args.out, name))
        # Teacher rebuilt at the student's resolution; conv weights transfer across input sizes.
        teacher_at_resolution = build_triage_model(
            replace(teacher_config, image_size=student_config.image_size), n_fine, METADATA_DIM)
        teacher_at_resolution.load_weights(teacher_weights)
        student_model, _bundle = distill(student_config, teacher_at_resolution, train_records, val_records)

        if variants.VARIANTS[name].quantize:
            print(f"=== Quantising {name} to int8 TFLite (live-demo model) ===")
            representative, _, _ = make_dataset(
                filter_valid(train_records), student_config, fine_vocabulary, training=False)
            tflite_path = os.path.join(args.out, name, 'model_int8.tflite')
            quantize.export_tflite(student_model, tflite_path, dataset=representative, mode='int8')
            print(f"  {name} int8 TFLite: {quantize.model_size_mb(tflite_path):.1f} MB")

    print(f"4.5.0 family complete in {args.out}: {variants.TEACHER} + {', '.join(variants.STUDENTS)}")


if __name__ == '__main__':
    main()
