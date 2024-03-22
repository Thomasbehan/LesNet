import os
from skinvestigatorai.core.ai.detector import SkinCancerDetector
from skinvestigatorai.core.ai.config import train_dir, val_dir, test_dir
from skinvestigatorai.core.data_scraper import DataScraper


def calculate_class_weights(train_dir):
    # Count the number of benign and malignant cases in the training set
    num_benign = len(os.listdir(os.path.join(train_dir, 'benign')))
    num_malignant = len(os.listdir(os.path.join(train_dir, 'malignant')))
    total = num_benign + num_malignant

    # Calculate class weights
    weight_for_benign = (1 / num_benign) * (total / 2.0)
    weight_for_malignant = (1 / num_malignant) * (total / 2.0)

    class_weights = {0: weight_for_benign, 1: weight_for_malignant}

    return class_weights


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
    detector.build_model(num_classes=len(train_generator.class_indices))

    # Calculate class weights for balanced training
    class_weights = calculate_class_weights(train_dir)

    detector.train_model(train_generator, val_generator, class_weights=class_weights)
    detector.evaluate_model(test_datagen)
    detector.save_model(filename)


if __name__ == '__main__':
    main('skin_cancer_detection_model_all_GPU.h5')
