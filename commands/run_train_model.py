import os

from run_data_scraper import main as DownloadData
from skinvestigatorai.config.model import ModelConfig
from skinvestigatorai.services.data import Data
from skinvestigatorai.services.model import SVModel


def main(filename='models/skinvestigator.keras'):
    if not os.path.exists(ModelConfig.TRAIN_DIR):
        print('Missing Training Data: Downloading Dataset.')
        DownloadData()
    ModelConfig.CATEGORIES = len([name for name in os.listdir(ModelConfig.TRAIN_DIR)])
    print('Train Categories:', ModelConfig.CATEGORIES)

    model = SVModel()
    model.build_model()
    data_service = Data()
    train_ds, validation_ds = data_service.load_dataset()

    model.train_model(train_ds, validation_ds)
    model.evaluate_model(validation_ds)

    model.save_model(filename)


if __name__ == '__main__':
    main()
