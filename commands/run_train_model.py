import os
from skinvestigatorai.services.detector_service import SkinCancerDetector
from __config import train_dir, val_dir, test_dir
from skinvestigatorai.services.data_scaper_service import DataScraper
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def calculate_class_weights(train_dir):
    classes = [0, 1]  # 0 for benign, 1 for malignant

    num_benign = len(os.listdir(os.path.join(train_dir, 'benign')))
    num_malignant = len(os.listdir(os.path.join(train_dir, 'malignant')))
    # Calculate class weights for balanced training
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=np.array([0] * num_benign + [1] * num_malignant)
    )
    return dict(zip(classes, class_weights))


def main(filename='models/skinvestigator.h5'):
    # check if data is downloaded and if not download it
    if not os.path.exists(train_dir):
        print('Downloading data...')
        downloader = DataScraper()
        downloader.download_and_split_images()
        print('Done downloading data')

    # Print count of files in each directory
    print('Train:', len(os.listdir(train_dir + '/benign')), 'benign,', len(os.listdir(train_dir + '/malignant')),
          'malignant')
    # Print count of files in each directory
    print('Test:', len(os.listdir(test_dir + '/benign')), 'benign,', len(os.listdir(test_dir + '/malignant')),
          'malignant')

    detector = SkinCancerDetector(train_dir, val_dir, test_dir)
    train_generator, val_generator, test_datagen = detector.preprocess_data()
    detector.build_model()

    class_weights = calculate_class_weights(train_dir)

    detector.train_model(train_generator, val_generator, class_weights=class_weights)
    detector.evaluate_model(test_datagen)
    detector.save_model(filename)


if __name__ == '__main__':
    main()
