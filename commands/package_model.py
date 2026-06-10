"""Package a trained artifacts directory into release assets with standard naming.

  python commands/package_model.py --artifacts models/triage_gpu40k --model-id M-4s

Produces in release_assets/:
  LesNet.<id>.keras            the model
  LesNet.<id>.artifacts.json   the calibration/conformal/OOD/threshold bundle
  LesNet.<id>_labels.json      the fine-grained label list (index order)
  LesNet.<id>.zip              all three, for drop-in use
"""
import argparse
import json
import os
import shutil
import zipfile


def main():
    parser = argparse.ArgumentParser(description="Package a trained model as release assets.")
    parser.add_argument('--artifacts', required=True)
    parser.add_argument('--model-id', required=True, help="e.g. M-4s or M-4")
    parser.add_argument('--out', default='release_assets')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    keras_name = f"LesNet.{args.model_id}.keras"
    bundle_name = f"LesNet.{args.model_id}.artifacts.json"
    labels_name = f"LesNet.{args.model_id}_labels.json"
    zip_name = f"LesNet.{args.model_id}.zip"

    shutil.copy(os.path.join(args.artifacts, 'triage_model.keras'), os.path.join(args.out, keras_name))
    shutil.copy(os.path.join(args.artifacts, 'artifacts.json'), os.path.join(args.out, bundle_name))

    bundle = json.load(open(os.path.join(args.artifacts, 'artifacts.json'), encoding='utf-8'))
    fine_vocabulary = bundle['label_maps']['fine_vocabulary']
    labels = [None] * len(fine_vocabulary)
    for label, index in fine_vocabulary.items():
        labels[index] = label
    json.dump(labels, open(os.path.join(args.out, labels_name), 'w', encoding='utf-8'), indent=2)

    with zipfile.ZipFile(os.path.join(args.out, zip_name), 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in (keras_name, bundle_name, labels_name):
            archive.write(os.path.join(args.out, name), name)

    size_mb = os.path.getsize(os.path.join(args.out, keras_name)) / 1e6
    print(f"Packaged {args.model_id}: {keras_name} ({size_mb:.1f} MB), input {bundle['config']['image_size']}, "
          f"fine-classes={len(labels)}, triage={bundle['label_maps']['triage_classes']}")


if __name__ == '__main__':
    main()
