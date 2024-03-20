import os
import datetime
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class SkinCancerDetector:
    def __init__(self, train_dir, val_dir, test_dir, log_dir='logs', batch_size=32, model_dir='models',
                 img_size=(180, 180)):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.model_dir = model_dir
        self.model = None
        self.precision = Precision()
        self.recall = Recall()

    def preprocess_data(self):
        """Preprocess data and apply image augmentation."""
        train_generator = self.create_data_generator(self.train_dir)
        val_generator = self.create_data_generator(self.val_dir, augment=True)
        test_datagen = self.create_data_generator(augment=True)

        return train_generator, val_generator, test_datagen

    def create_data_generator(self, dir=None, augment=False):
        if augment:
            datagen = ImageDataGenerator(
                rescale=1. / 255,
                horizontal_flip=True,
                vertical_flip=True,
                brightness_range=[0.8, 1.2]
            )
        else:
            datagen = ImageDataGenerator(rescale=1. / 255)

        if dir:
            return datagen.flow_from_directory(
                dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='binary'
            )
        return datagen

    def quantize_model(self, model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quant_model = converter.convert()

        return tflite_quant_model

    def build_model(self, num_classes=2):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1. / 255, input_shape=(self.img_size[0], self.img_size[1], 3)),
            tf.keras.layers.Conv2D(180, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=[
                               'accuracy',
                               tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.AUC(name='auc')
                           ])

    def train_model(self, train_generator, val_generator, epochs=1000, patience_lr=12, patience_es=40, min_lr=1e-6,
                    min_delta=1e-4, cooldown_lr=5):
        """Train the model with callbacks."""
        self._check_model()

        # Create a log directory with a timestamp
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(self.log_dir, current_time)
        os.makedirs(log_dir, exist_ok=True)

        callbacks = self._create_callbacks(log_dir, current_time, patience_lr, min_lr, min_delta, patience_es,
                                           cooldown_lr)

        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks)
        return history

    def _create_callbacks(
            self, log_dir, current_time, patience_lr=10, min_lr=1e-5, min_delta=1e-3, patience_es=30, cooldown_lr=5
    ):
        """Callbacks for optimized learning rate adjustments and early stopping."""
        tensorboard_callback = TensorBoard(
            log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch', profile_batch=0
        )
        reduce_lr_callback = ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=patience_lr, min_lr=min_lr, min_delta=min_delta,
            cooldown=cooldown_lr, verbose=1
        )
        model_checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(self.model_dir, "{}_best_model.h5".format(current_time)),
            save_best_only=True, monitor='val_loss', mode='min', verbose=1
        )
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience_es, restore_best_weights=True,
                                                verbose=1)

        return [tensorboard_callback, reduce_lr_callback, model_checkpoint_callback, early_stopping_callback]

    def evaluate_model(self, test_datagen):
        """Evaluate the model for binary classification."""
        self._check_model()

        test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary')  # Updated to binary

        test_loss, test_acc, test_sensitivity, test_precision, test_f1, test_specificity, test_auc \
            = self.model.evaluate(test_generator)
        print('Test accuracy:', test_acc)
        print('Test sensitivity:', test_sensitivity)
        print('Test precision:', test_precision)
        print('Test F1-score:', test_f1)
        print('Test specificity:', test_specificity)
        print('Test AUC-ROC:', test_auc)
        return test_loss, test_acc, test_sensitivity, test_precision, test_f1, test_specificity, test_auc

    def save_model(self, filename='models/skinvestigator.h5'):
        """Save the original and quantized model."""
        self._check_model()

        # Save the original model
        self.model.save(filename)

        # Quantize and save the model
        tflite_model = self.quantize_model(self.model)
        tflite_model_path = filename.replace('.h5', '-quantized.tflite')
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Model saved as {filename} and {tflite_model_path}")

    def load_model(self, filename):
        """Load the model."""
        self.model = tf.keras.models.load_model(filename)
        print(f"Model loaded from {filename}")

    def _check_model(self):
        """Checking if the model has not been built."""
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")
