import os


from run_data_scraper import main as DownloadData
from skinvestigatorai.config.model import ModelConfig
from skinvestigatorai.services.data import Data
from skinvestigatorai.services.model import SVModel


def main():
    if not os.path.exists(ModelConfig.TRAIN_DIR):
        print('Missing Training Data: Downloading Dataset.')
        DownloadData()

    print('Train Categories:', len([name for name in os.listdir(ModelConfig.TRAIN_DIR)]))

    ModelConfig.EPOCHS = 15
    model = SVModel()
    data_service = Data()
    model.build_model()
    train_ds, validation_ds = data_service.load_dataset()
    model.run_experiments(train_ds, validation_ds)


if __name__ == '__main__':
    main()
