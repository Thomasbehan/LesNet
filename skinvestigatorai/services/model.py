import datetime
import itertools
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Dense, BatchNormalization, GlobalAveragePooling2D, Dropout
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
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

    def create_feature_extractor(self):
        if self.model_type == 'KERAS':
            self.feature_extractor = Model(inputs=self.model.input, outputs=self.model.layers[-3].output)
        elif self.model_type == 'TFLITE':
            self.feature_extractor = self.model
        else:
            raise ValueError("Unsupported model type. Please use 'KERAS' or 'TFLITE'.")

    def preprocess_image_for_tflite(self, img):
        img_resized = tf.image.resize(img, [self.img_size[0], self.img_size[1]])
        img_normalized = img_resized / 255.0
        return img_normalized

    def calculate_dataset_embedding(self, data_generator):
        features = []
        if self.model_type == 'KERAS':
            for _, (imgs, _) in enumerate(data_generator):
                features.append(self.feature_extractor.predict(imgs))
        elif self.model_type == 'TFLITE':
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()

            for imgs, _ in data_generator:
                for img in imgs:
                    img = self.preprocess_image_for_tflite(img)
                    img = np.expand_dims(img, axis=0).astype(input_details[0]['dtype'])
                    self.model.set_tensor(input_details[0]['index'], img)
                    self.model.invoke()
                    features.append(self.model.get_tensor(output_details[0]['index'])[0])
        else:
            raise ValueError("Unsupported model type. Please use 'KERAS' or 'TFLITE'.")

        features = np.concatenate(features, axis=0)
        self.dataset_embedding = np.mean(features, axis=0)

    def build_model(self):
        num_classes = ModelConfig.CATEGORIES
        input_shape = (self.img_size[0], self.img_size[1], 3)

        base_model = InceptionV3(
            include_top=False,
            weights='imagenet',
            input_tensor=Input(shape=input_shape)
        )

        # Freezing all base layers and unfreezing the last few
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[-ModelConfig.BASE_LAYERS_TO_UNFREEZE:]:
            layer.trainable = True

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(ModelConfig.LAYER_1, activation='relu', kernel_regularizer=l2(ModelConfig.L2_LAYER_1))(x)
        x = Dropout(ModelConfig.DROPOUT_1)(x)
        x = Dense(ModelConfig.LAYER_2, activation='relu', kernel_regularizer=l2(ModelConfig.L2_LAYER_2))(x)
        x = Dense(ModelConfig.LAYER_3, activation='relu', kernel_regularizer=l2(ModelConfig.L2_LAYER_3))(x)
        outputs = Dense(num_classes, activation='softmax')(x)

        self.model = Model(inputs=base_model.input, outputs=outputs)
        self.model.compile(
            optimizer=self.optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall()]
        )
        self.model.summary()

    def compute_class_weights(self, class_series):
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
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(
            test_datagen)
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
        quantized_model = tf.quantization.quantize(
            input=model.input,
            output=model.output,
            quantization_range=(0, 6),
            quantization_axis=-1,
            name=None
        )

        converter = tf.lite.TFLiteConverter.from_keras_model(quantized_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quant_model = converter.convert()

        tflite_model_path = './models/model-quantized.tflite'
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_quant_model)

        return tflite_quant_model

    def save_model(self, filename='models/model.keras', class_labels=None):
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

    def train_model(self, train_generator, val_generator, epochs=ModelConfig.EPOCHS,
                    patience_lr=ModelConfig.LR_PATIENCE,
                    patience_es=ModelConfig.ES_PATIENCE,
                    min_lr=ModelConfig.MIN_LR, min_delta=ModelConfig.MIN_LR_DELTA,
                    cooldown_lr=ModelConfig.LR_COOLDOWN):
        self._check_model()
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(self.log_dir, current_time)
        os.makedirs(log_dir, exist_ok=True)
        callbacks = self._create_callbacks(log_dir, current_time, patience_lr, min_lr, min_delta, patience_es,
                                           cooldown_lr)

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
