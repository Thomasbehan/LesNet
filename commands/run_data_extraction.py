import os
import shutil

import pandas as pd

from lesnet.config.model import ModelConfig


def resolve_diagnosis(row):
    """Pick the most specific diagnosis available, matching the original precedence."""
    diagnosis = row['diagnosis_1'] if pd.notna(row['diagnosis_1']) else "unknown"
    if pd.notna(row['diagnosis_2']):
        diagnosis = row['diagnosis_2']
    if pd.notna(row['diagnosis_3']):
        diagnosis = row['diagnosis_3']
    if pd.notna(row['diagnosis']):
        diagnosis = row['diagnosis']
    return diagnosis


def main():
    metadata = pd.read_csv(os.path.join(ModelConfig.TRAIN_DIR, 'metadata.csv'))
    for column in ['diagnosis', 'diagnosis_1', 'diagnosis_2', 'diagnosis_3']:
        metadata[column] = metadata[column].astype(str).replace('None', pd.NA)

    failed_count = 0
    moved_count = 0

    for _, row in metadata.iterrows():
        isic_id = row['isic_id']
        diagnosis = resolve_diagnosis(row)

        image_file = os.path.join(ModelConfig.TRAIN_DIR, f"{isic_id}.jpg")
        diagnosis_folder = os.path.join(ModelConfig.TRAIN_DIR, diagnosis)
        os.makedirs(diagnosis_folder, exist_ok=True)

        if os.path.isfile(image_file):
            shutil.move(image_file, os.path.join(diagnosis_folder, f"{isic_id}.jpg"))
            moved_count += 1
        else:
            print(f"Image {image_file} not found. Diagnosis: {diagnosis}.")
            failed_count += 1

    print(f"Moved Count: {moved_count}. Failed Count: {failed_count}.")


if __name__ == '__main__':
    main()
