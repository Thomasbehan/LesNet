import os
import datetime
import albumentations as A
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class SkinCancerDetector:
    def __init__(self, train_dir, val_dir, test_dir, log_dir='logs'):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.log_dir = log_dir
        self.model = None
        self.batch_size = 4028  # Increased batch size

    def preprocess_data(self):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            preprocessing_function=A.Compose([
                A.Rotate(limit=40),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2),
                A.HorizontalFlip(),
                A.RandomBrightnessContrast(),
                A.CoarseDropout(max_holes=8),
            ]),
            fill_mode='nearest'
        )

        val_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(150, 150),
            batch_size=self.batch_size,
            class_mode='categorical')

        val_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=(150, 150),
            batch_size=self.batch_size,
            class_mode='categorical')

        return train_generator, val_generator, test_datagen

    def build_model(self, num_classes):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(num_classes, activation='softmax', dtype=tf.float32))

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model

    def train_model(self, train_generator, val_generator, epochs=30):
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")

        # Create a log directory with a timestamp
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(self.log_dir, current_time)
        os.makedirs(log_dir, exist_ok=True)

        # Set up the TensorBoard callback
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True,
                                           update_freq='epoch', profile_batch=0)
        reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
        model_checkpoint_callback = ModelCheckpoint(filepath=os.path.join(log_dir, "best_model.h5"),
                                                    save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
        early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

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
