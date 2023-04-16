import os
from skinvestigatorai.core.ai.detector import SkinCancerDetector
from skinvestigatorai.core.ai.config import train_dir, val_dir, test_dir
import skinvestigatorai.core.data_scraper as data_scraper


def main(filename='skin_cancer_detection_model.h5'):
    # Print count of files in each directory
    print('Train:', len(os.listdir(train_dir + '/benign')), 'benign,', len(os.listdir(train_dir + '/malignant')),
          'malignant')

    detector = SkinCancerDetector(train_dir, val_dir, test_dir)
    train_generator, val_generator, test_datagen = detector.preprocess_data()
    detector.build_model(num_classes=len(train_generator.class_indices))
    detector.train_model(train_generator, val_generator)
    detector.evaluate_model(test_datagen)
    detector.save_model(filename)


if __name__ == '__main__':

    # downloader = data_scraper.DataScraper()
    print('Done training models')
    print('Training model with all data')
    # downloader.download_images()
    main('skin_cancer_detection_model_all_GPU.h5')
