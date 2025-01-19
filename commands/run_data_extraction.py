import os
import pandas as pd
import shutil
from lesnet.config.model import ModelConfig

# Load the metadata.csv file
metadata = pd.read_csv(os.path.join(ModelConfig.TRAIN_DIR, 'metadata.csv'))
# Ensure 'diagnosis_*' is treated as a string and handle NaN values
metadata['diagnosis'] = metadata['diagnosis'].astype(str).replace('None', pd.NA)
metadata['diagnosis_1'] = metadata['diagnosis_1'].astype(str).replace('None', pd.NA)
metadata['diagnosis_2'] = metadata['diagnosis_2'].astype(str).replace('None', pd.NA)
metadata['diagnosis_3'] = metadata['diagnosis_3'].astype(str).replace('None', pd.NA)

# Extract the relevant columns
isic_id_diagnosis = metadata[['isic_id', 'diagnosis',  'diagnosis_1', 'diagnosis_2', 'diagnosis_3']]
failed_count = 0
moved_count = 0

# Iterate through each row in the DataFrame
for index, row in isic_id_diagnosis.iterrows():
    isic_id = row['isic_id']
    diagnosis = row['diagnosis_1'] if row['diagnosis_1'] is not pd.NA else "unknown"
    if row['diagnosis_2'] is not pd.NA:
        diagnosis = row['diagnosis_2']
    if row['diagnosis_3'] is not pd.NA:
        diagnosis = row['diagnosis_3']
    if row['diagnosis'] is not pd.NA:
        diagnosis = row['diagnosis']

    # Construct the image file path
    image_file = os.path.join(ModelConfig.TRAIN_DIR, f"{isic_id}.jpg")  # Assuming images are in .jpg format

    # Create a new folder for the diagnosis if it doesn't exist
    diagnosis_folder = os.path.join(ModelConfig.TRAIN_DIR, diagnosis)
    os.makedirs(diagnosis_folder, exist_ok=True)

    # Check if the image file exists and move it to the corresponding folder
    if os.path.isfile(image_file):
        shutil.move(image_file, os.path.join(diagnosis_folder, f"{isic_id}.jpg"))
        moved_count = moved_count + 1
    else:
        print(f"Image {image_file} not found. Diagnosis: {diagnosis}.")
        failed_count = failed_count + 1

print(f"Moved Count: {moved_count}. Failed Count: {failed_count}.")
