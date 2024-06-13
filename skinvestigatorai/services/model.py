import datetime
import itertools
import json
import os

import numpy as np
import tensorflow as tf

# Ensure the model and data are stored on the GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from sklearn.utils.class_weight import compute_class_weight
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Reshape, multiply
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.regularizers import l2
from skinvestigatorai.config.model import ModelConfig


class SVModel:
    def __init__(self, model=None, model_type=ModelConfig.MODEL_TYPE):
        self.model_type = model_type
        self.model = model
        self.feature_extractor = None
        self.dataset_embedding = None
        self.log_dir = ModelConfig.LOG_DIR
        self.img_size = ModelConfig.IMG_SIZE
        self.optimizer = Adam(ModelConfig.LEARNING_RATE)
        self.support_set = None  # Few-shot learning support set
        self.query_set = None  # Few-shot learning query set

    def create_feature_extractor(self):
        if self.model_type == 'KERAS':
            self.feature_extractor = Model(inputs=self.model.input, outputs=self.model.layers[-3].output)
        elif self.model_type == 'TFLITE':
            self.feature_extractor = self.model
        else:
            raise ValueError("Unsupported model type. Please use 'KERAS' or 'TFLITE'.")

    def prepare_few_shot_data(self, support_set, query_set):
        self.support_set = support_set
        self.query_set = query_set

    def compute_prototypes(self, support_set):
        self.create_feature_extractor()
        support_embeddings = self.feature_extractor.predict(support_set[0])

        # Ensure support_set[1] dimensions are corrected
        support_labels = np.argmax(support_set[1], axis=1) if len(support_set[1].shape) > 1 else support_set[1]

        prototypes = []
        for class_id in np.unique(support_labels):
            class_embeddings = support_embeddings[support_labels == class_id]
            prototypes.append(np.mean(class_embeddings, axis=0))
        return np.array(prototypes)

    def preprocess_image_for_tflite(self, img):
        img_resized = tf.image.resize(img, [self.img_size[0], self.img_size[1]])
        img_normalized = img_resized / 255.0
        return img_normalized

    def calculate_dataset_embedding(self, support_set):
        prototypes = self.compute_prototypes(support_set)
        self.prototypes = prototypes

    def evaluate_few_shot(self):
        query_embeddings = self.feature_extractor.predict(self.query_set[0])
        distances = -np.linalg.norm(query_embeddings[:, None] - self.dataset_embedding, axis=2)
        predicted_classes = np.argmax(distances, axis=1)
        accuracy = np.mean(predicted_classes == self.query_set[1])
        print(f'Few-Shot Learning Accuracy: {accuracy}')
        return accuracy

    def squeeze_excitation_block(self, input_tensor, ratio=16):
        filters = input_tensor.shape[-1]
        se = GlobalAveragePooling2D()(input_tensor)
        se = Reshape((1, 1, filters))(se)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
        return multiply([input_tensor, se])

    def bottleneck_block_v2(self, x, filters, stride=1, conv_shortcut=True):
        """
        A bottleneck residual block with optional convolutional shortcut.
        """
        regularizer = tf.keras.regularizers.l2(ModelConfig.L2_LAYER_1)

        # Shortcut connection
        shortcut = x
        if conv_shortcut:
            shortcut = layers.Conv2D(4 * filters, 1, strides=stride, kernel_regularizer=regularizer)(x)
            shortcut = layers.BatchNormalization()(shortcut)
        else:
            if stride != 1 or x.shape[-1] != 4 * filters:
                shortcut = layers.Conv2D(4 * filters, 1, strides=stride, use_bias=False,
                                         kernel_regularizer=regularizer)(x)
                shortcut = layers.BatchNormalization()(shortcut)

        # Pre-activation block
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # First convolutional layer
        x = layers.Conv2D(filters, 1, strides=1, use_bias=False, kernel_regularizer=regularizer)(x)

        # Pre-activation block
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Second convolutional layer
        x = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False, kernel_regularizer=regularizer)(x)

        # Pre-activation block
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Third convolutional layer
        x = layers.Conv2D(4 * filters, 1, strides=1, kernel_regularizer=regularizer)(x)

        # Add shortcut and return
        x = layers.add([shortcut, x])
        return x

    def build_model(self, input_shape, num_classes):
        regularizer = l2(ModelConfig.L2_LAYER_1)

        inputs = keras.Input(shape=input_shape)

        # Initial convolutional layer
        x = layers.Conv2D(ModelConfig.CONV_LAYER_1, 7, strides=2, padding='same', use_bias=False,
                          kernel_regularizer=regularizer)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

        # Stage 1
        for _ in range(ModelConfig.STAGE_1_LAYERS):
            x = self.bottleneck_block_v2(x, ModelConfig.BN_LAYER_1, conv_shortcut=False)

        # Stage 2
        x = self.bottleneck_block_v2(x, ModelConfig.BN_LAYER_2, stride=2)
        for _ in range(ModelConfig.STAGE_2_LAYERS):
            x = self.bottleneck_block_v2(x, ModelConfig.BN_LAYER_2, conv_shortcut=False)

        # Stage 3
        x = self.bottleneck_block_v2(x, ModelConfig.BN_LAYER_3, stride=2)
        for _ in range(ModelConfig.STAGE_3_LAYERS):
            x = self.bottleneck_block_v2(x, ModelConfig.BN_LAYER_3, conv_shortcut=False)

        # Stage 4
        x = self.bottleneck_block_v2(x, ModelConfig.BN_LAYER_4, stride=2)
        for _ in range(ModelConfig.STAGE_4_LAYERS):
            x = self.bottleneck_block_v2(x, ModelConfig.BN_LAYER_4, conv_shortcut=False)

        # Final layers
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.GlobalAveragePooling2D()(x)
        if ModelConfig.DROPOUT_1 > 0:
            x = layers.Dropout(ModelConfig.DROPOUT_1)(x)
        outputs = layers.Dense(num_classes, activation='softmax', kernel_regularizer=l2(ModelConfig.L2_LAYER_1))(x)

        self.model = Model(inputs, outputs)
        self.model.compile(
            optimizer=self.optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall()]
        )
        self.model.summary()
        return self.model

    def compute_class_weights(self, class_series):
        if len(class_series) == 0:
            raise ValueError("class_series is empty")

        class_labels = np.unique(class_series)
        class_weights = compute_class_weight('balanced', classes=class_labels, y=class_series)
        class_weight_dict = dict(zip(class_labels, class_weights))
        return class_weight_dict

    def get_latest_model(self, model_dir=ModelConfig.MODEL_DIRECTORY, extension=ModelConfig.MODEL_TYPE):
        list_of_files = [os.path.join(model_dir, basename) for basename in os.listdir(model_dir) if
                         basename.endswith(extension)]
        latest_model = max(list_of_files, key=os.path.getctime)
        print("LATEST MODEL:")
        print(latest_model)
        return latest_model

    def _check_model(self):
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")

    def evaluate_model(self, test_datagen):
        self._check_model()
        if self.support_set is not None and self.query_set is not None:
            return self.evaluate_few_shot()
        else:
            test_loss, test_acc, test_precision, test_recall = self.model.evaluate(test_datagen)
            print(
                f'Test Accuracy: {test_acc}, '
                f'Test Precision: {test_precision}, '
                f'Test Recall: {test_recall} '
                f'Test Loss: {test_loss}'
            )
            return test_loss, test_acc, test_precision, test_recall

    def run_experiments(self, train_ds, val_ds):
        # Define hyperparameters using TensorBoard HParams API
        HPARAMS = {
            'batch_size': hp.HParam('batch_size', hp.Discrete([32, 64, 128, 256])),
            'learning_rate': hp.HParam('learning_rate', hp.RealInterval(0.0001, 0.01)),
            'layer_1': hp.HParam('layer_1', hp.Discrete([128, 256, 512, 1024, 2048])),
            'layer_2': hp.HParam('layer_2', hp.Discrete([64, 128, 256, 512, 1024])),
            'layer_3': hp.HParam('layer_3', hp.Discrete([64, 128, 256, 256, 512])),
            'dropout_1': hp.HParam('dropout_1', hp.RealInterval(0.1, 0.5)),
            'base_layers_to_unfreeze': hp.HParam('base_layers_to_unfreeze', hp.Discrete([10, 15, 20, 25])),
            'l2_layer_1': hp.HParam('l2_layer_1', hp.RealInterval(0.001, 0.01)),
            'l2_layer_2': hp.HParam('l2_layer_2', hp.RealInterval(0.001, 0.01)),
            'l2_layer_3': hp.HParam('l2_layer_3', hp.RealInterval(0.001, 0.01))
        }

        METRICS = [
            hp.Metric('accuracy', display_name='Accuracy'),
            hp.Metric('loss', display_name='Loss'),
            hp.Metric('precision', display_name='Precision'),
            hp.Metric('recall', display_name='Recall')
        ]

        with tf.summary.create_file_writer(self.log_dir).as_default():
            hp.hparams_config(
                hparams=[HPARAMS[k] for k in HPARAMS],
                metrics=METRICS
            )

        # Generate all combinations of hyperparameter values
        keys, values = zip(*HPARAMS.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*(
            val.domain.values if hasattr(val.domain, 'values') else np.linspace(val.domain.min_value,
                                                                                val.domain.max_value, num=5)
            for val in values))]

        session_num = 0
        for hparams in experiments:
            hparams_dict = {hparam.name: value for hparam, value in zip(HPARAMS.values(), hparams.values())}
            run_name = f"run-{session_num}"
            print('--- Starting trial:', run_name)
            print(hparams_dict)
            ModelConfig.BATCH_SIZE = hparams_dict['batch_size']
            ModelConfig.LEARNING_RATE = hparams_dict['learning_rate']
            ModelConfig.BASE_LAYERS_TO_UNFREEZE = hparams_dict['base_layers_to_unfreeze']
            ModelConfig.DROPOUT_1 = hparams_dict['dropout_1']
            ModelConfig.LAYER_1 = hparams_dict['layer_1']
            ModelConfig.LAYER_2 = hparams_dict['layer_2']
            ModelConfig.LAYER_3 = hparams_dict['layer_3']
            ModelConfig.L2_LAYER_1 = hparams_dict['l2_layer_1']
            ModelConfig.L2_LAYER_2 = hparams_dict['l2_layer_2']
            ModelConfig.L3_LAYER_3 = hparams_dict['l2_layer_3']
            self.train_model(train_ds, val_ds, 15)
            loss, accuracy, precision, recall = self.evaluate_model(val_ds)
            metrics = {
                'accuracy': accuracy,
                'loss': loss,
                'precision': precision,
                'recall': recall
            }
            self.log_metrics(run_name, metrics)
            session_num += 1

    def log_metrics(self, run_name, metrics):
        log_dir = os.path.join("logs/hparam_tuning", run_name)
        with tf.summary.create_file_writer(log_dir).as_default():
            for key, value in metrics.items():
                tf.summary.scalar(key, value, step=1)
            tf.summary.flush()

    def quantize_model(self, model):
        try:
            tf.config.run_functions_eagerly(True)
            tf.config.experimental.set_synchronous_execution(True)
            model = self.remove_batch_norm(model)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()

            tflite_model_path = './models/LesNet.tflite'
            with open(tflite_model_path, 'wb') as f:
                f.write(tflite_model)

            print("Model converted successfully.")
            return tflite_model
        except Exception as e:
            print("Error during conversion:", e)
            return None

    def remove_batch_norm(self, model):
        for layer in model.layers:
            if 'batch_normalization' in layer.name:
                model.get_layer(layer.name).trainable = False
        return model

    def save_model(self, filename=ModelConfig.MODEL_NAME, class_labels=None):
        self._check_model()
        self.model.save(filename)

        labels_filename = filename.replace('.keras', '_labels.json')
        if class_labels is not None:
            with open(labels_filename, 'w') as f:
                json.dump(class_labels, f)

        print(f"Model saved as {filename}, and class labels in {labels_filename}")

    def _create_callbacks(self, log_dir, current_time, patience_lr, min_lr, min_delta, patience_es, cooldown_lr):
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True,
                                           update_freq='epoch', profile_batch=0)
        reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_lr, min_lr=min_lr,
                                               min_delta=min_delta, cooldown=cooldown_lr, verbose=1)
        model_checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(ModelConfig.MODEL_DIRECTORY, f"{current_time}_best_model.keras"), save_best_only=True,
            monitor='val_loss', mode='min', verbose=1)
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience_es, restore_best_weights=True,
                                                verbose=1)

        return [tensorboard_callback, reduce_lr_callback, model_checkpoint_callback, early_stopping_callback]

    def train_model(self, train_generator, val_generator, support_set=None, query_set=None, epochs=ModelConfig.EPOCHS,
                    patience_lr=ModelConfig.LR_PATIENCE, patience_es=ModelConfig.ES_PATIENCE,
                    min_lr=ModelConfig.MIN_LR, min_delta=ModelConfig.MIN_LR_DELTA, cooldown_lr=ModelConfig.LR_COOLDOWN):
        self._check_model()
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(self.log_dir, current_time)
        os.makedirs(log_dir, exist_ok=True)
        callbacks = self._create_callbacks(log_dir, current_time, patience_lr, min_lr, min_delta, patience_es,
                                           cooldown_lr)

        if support_set is not None and query_set is not None:
            self.prepare_few_shot_data(support_set, query_set)
            self.calculate_dataset_embedding(support_set)

        all_labels = np.concatenate([labels for _, labels in train_generator], axis=0)
        all_labels = np.argmax(all_labels, axis=1)

        class_weights = self.compute_class_weights(all_labels)

        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            class_weight=class_weights
        )
        return history

    def load_model(self, filename=ModelConfig.MODEL_NAME):
        file_location = os.path.join(ModelConfig.MODEL_DIRECTORY, filename)
        if not os.path.exists(file_location):
            raise FileNotFoundError(f"Model file {filename} not found.")

        if ModelConfig.MODEL_TYPE == 'KERAS':
            self.model = tf.keras.models.load_model(file_location)
        elif ModelConfig.MODEL_TYPE == 'TFLITE':
            self.model = tf.lite.Interpreter(model_path=file_location)
            self.model.allocate_tensors()
        else:
            raise ValueError("Unsupported model type. Please use 'KERAS' or 'TFLITE'.")

        labels_filename = os.path.join(ModelConfig.MODEL_DIRECTORY, ModelConfig.LABELS_NAME)
        try:
            with open(labels_filename, 'r') as f:
                class_labels = json.load(f)
        except FileNotFoundError:
            class_labels = None
            print("No class labels file found. Please ensure you have downloaded the class_labels.json file.")
            exit(1)

        print(f"Model loaded from {filename} with class labels.")
        return self.model, class_labels
