import os

import tensorflow as tf

from lesnet.config.model import ModelConfig
from lesnet.services.data import Data
from lesnet.services.model import SVModel
from run_data_scraper import main as DownloadData


def main(filename='models/' + ModelConfig.MODEL_NAME):
    if not os.path.exists(ModelConfig.TRAIN_DIR):
        print('Missing Training Data: Downloading Dataset.')
        DownloadData()
    labels_gen = [name for name in os.listdir(ModelConfig.TRAIN_DIR + "/train")]
    ModelConfig.CATEGORIES = len(labels_gen)
    print('Train Categories:', ModelConfig.CATEGORIES)

    model = SVModel()
    if ModelConfig.TPU_Train:
        # detect and init the TPU
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

        # instantiate a distribution strategy
        tf.tpu.experimental.initialize_tpu_system(tpu)
        tpu_strategy = tf.distribute.TPUStrategy(tpu)

        # instantiating the model in the strategy scope creates the model on the TPU
        with tpu_strategy.scope():
            model.build_model()
    else:
        model.build_model()

    data_service = Data()
    train_ds, validation_ds, test_ds = data_service.load_preprocessed_dataset()

    model.train_model(train_ds, validation_ds)

    model.save_model(filename, labels_gen)

    model.evaluate_model(test_ds)


if __name__ == '__main__':
    main()
