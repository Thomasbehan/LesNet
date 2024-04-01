import os

from tensorboard.plugins.hparams import api as hp

from run_data_scraper import main as DownloadData
from skinvestigatorai.config.model import ModelConfig
from skinvestigatorai.services.data import Data
from skinvestigatorai.services.tuner import SVModelHPTuner


def main():
    if not os.path.exists(ModelConfig.TRAIN_DIR):
        print('Missing Training Data: Downloading Dataset.')
        DownloadData()

    print('Train Categories:', len([name for name in os.listdir(ModelConfig.TRAIN_DIR)]))

    hparams = {
        'learning_rate': hp.HParam('learning_rate', hp.Discrete([1e-4, 1e-3, 1e-2])),
        'num_classes': hp.HParam('num_classes',
                                 hp.Discrete([len([name for name in os.listdir(ModelConfig.TRAIN_DIR)])])),
        'dropout_rate': hp.HParam('dropout_rate', hp.Discrete([0.1, 0.2, 0.3, 0.4, 0.5])),
        'dropout_rate_2': hp.HParam('dropout_rate_2', hp.Discrete([0.1, 0.2, 0.3, 0.4, 0.5])),
        'units_in_dense_layer': hp.HParam('units_in_dense_layer', hp.Discrete([64, 128, 256, 512, 1024]))
    }
    tuner = SVModelHPTuner(hparams)
    data_service = Data()

    train_ds, validation_ds = data_service.load_dataset()
    tuner.run_experiment(train_ds, validation_ds)


if __name__ == '__main__':
    main()
