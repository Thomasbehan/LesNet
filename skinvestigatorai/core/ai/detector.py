import os
import datetime
import albumentations as A
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skinvestigatorai.core.data_gen import DataGen
from vit_keras import vit, utils


class SkinCancerDetector:
    def __init__(self, train_dir, val_dir, test_dir, log_dir='logs'):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.log_dir = log_dir
        self.model = None
        self.batch_size = 64

    def preprocess_data(self):
        aug = A.Compose([
            A.Rotate(limit=40),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2),
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(),
            A.CoarseDropout(max_holes=8),
        ])

        train_generator = DataGen(
            self.train_dir,
            batch_size=self.batch_size,
            img_height=150,
            img_width=150,
            augmentations=aug)

        val_datagen = ImageDataGenerator(rescale=1. / 255)
        val_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=(150, 150),
            batch_size=self.batch_size,
            class_mode='categorical')

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        return train_generator, val_generator, test_datagen

    def build_model(self, num_classes):
        image_size = 256
        vit_model = vit.vit_b32(
            image_size=image_size,
            activation='softmax',
            pretrained=True,
            include_top=False,
            pretrained_top=False,
            classes=num_classes)

        model = tf.keras.Sequential([
            vit_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1024, activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax', dtype=tf.float32)
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model

    def train_model(self, train_generator, val_generator, epochs=3000):
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")

        # Create a log directory with a timestamp
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(self.log_dir, current_time)
        os.makedirs(log_dir, exist_ok=True)

        # Set up the TensorBoard callback
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True,
                                           update_freq='epoch', profile_batch=0)
        reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=160, min_lr=1e-6,
                                               min_delta=1e-4)
        model_checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(log_dir, "{}_best_model.h5".format(current_time)),
            save_best_only=True, monitor='val_loss', mode='min', verbose=1)
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[tensorboard_callback, reduce_lr_callback, model_checkpoint_callback, early_stopping_callback])
        return history

    def evaluate_model(self, test_datagen):
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")

        test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical')

        test_loss, test_acc = self.model.evaluate(test_generator)
        print('Test accuracy:', test_acc)
        return test_loss, test_acc

    def save_model(self, filename='models/skinvestigator_nano_40MB_91_38_acc.h5'):
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")

        self.model.save(filename)
