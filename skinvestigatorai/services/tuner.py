import datetime
import itertools
import os

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from skinvestigatorai.config.model import ModelConfig
from skinvestigatorai.services.model import SVModel


class SVModelHPTuner(SVModel):
    def __init__(self, hparams, model_type=ModelConfig.MODEL_TYPE):
        super().__init__(model=None, model_type=model_type)
        self.hparams = hparams
        self.hparams_dict = {hp.HParam(k): v for k, v in hparams.items()}

    def run_experiment(self, train_generator, val_generator):
        with tf.summary.create_file_writer(self.log_dir).as_default():
            hp.hparams_config(
                hparams=list(self.hparams_dict.keys()),
                metrics=[hp.Metric('accuracy', display_name='Accuracy')]
            )
        for param_combination in self._generate_param_combinations():
            hparams = {h: v for h, v in zip(self.hparams_dict.keys(), param_combination.values())}
            run_name = "run-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            run_dir = os.path.join(self.log_dir, run_name)
            with tf.summary.create_file_writer(run_dir).as_default():
                hp.hparams(hparams)
                callbacks = self._create_hparam_callbacks(run_dir)
                self.build_model(param_combination['num_classes'])
                self.optimizer = self._get_optimizer(param_combination['learning_rate'])
                history = self.train_model(train_generator, val_generator, callbacks)
                self._log_results(history, param_combination, run_dir)

    def _get_optimizer(self, learning_rate):
        return tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)

    def _generate_param_combinations(self):
        combinations = []
        for k in self.hparams_dict.keys():
            if isinstance(k.domain, hp.Discrete):
                values = k.domain.values
            elif isinstance(k.domain, hp.RealInterval):
                values = np.linspace(k.domain.min_value, k.domain.max_value, num=3)
            else:
                continue
            combinations.append((k, values))

        keys, value_lists = zip(*combinations)
        for value_combination in itertools.product(*value_lists):
            yield dict(zip(keys, value_combination))

    def _log_results(self, history, params, run_dir):
        final_accuracy = max(history.history['val_accuracy'])
        with tf.summary.create_file_writer(run_dir).as_default():
            tf.summary.scalar('accuracy', final_accuracy, step=1)

    def _create_hparam_callbacks(self, log_dir):
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
        reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-10)
        model_checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(ModelConfig.MODEL_DIRECTORY, "best_model.h5"), save_best_only=True)
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        return [tensorboard_callback, reduce_lr_callback, model_checkpoint_callback, early_stopping_callback]
