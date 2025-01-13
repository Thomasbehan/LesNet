import os

from run_data_scraper import main as DownloadData
from lesnet.config.model import ModelConfig
from lesnet.services.data import Data
from lesnet.services.model import SVModel


def main(filename='models/' + ModelConfig.MODEL_NAME):
    if not os.path.exists(ModelConfig.TRAIN_DIR):
        print('Missing Training Data: Downloading Dataset.')
        DownloadData()
    labels_gen = [name for name in os.listdir(ModelConfig.TRAIN_DIR)]
    print(labels_gen)
    exit()
    ModelConfig.CATEGORIES = len(labels_gen)
    print('Train Categories:', ModelConfig.CATEGORIES)

    model = SVModel()
    model.build_model()
    data_service = Data()
    train_ds, validation_ds = data_service.load_dataset()

    model.train_model(train_ds, validation_ds)
    model.evaluate_model(validation_ds)

    model.save_model(filename, labels_gen)


if __name__ == '__main__':
    main()
