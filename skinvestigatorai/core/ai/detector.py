import os
import datetime
import albumentations as A
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from skinvestigatorai.core.data_gen import DataGen
from vit_keras import vit, utils


class SkinCancerDetector:
    def __init__(self, train_dir, val_dir, test_dir, log_dir='logs', batch_size=64, model_dir='models/v1.5',
                 img_size=(128, 128)):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.model_dir = model_dir
        self.model = None

        # Enable mixed precision training
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    def preprocess_data(self):
        """Preprocess data and apply image augmentation."""
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
            img_height=self.img_size[0],
            img_width=self.img_size[1],
            augmentations=aug)

        val_generator = self.create_data_generator(self.val_dir)
        test_datagen = self.create_data_generator()

        return train_generator, val_generator, test_datagen

    def create_data_generator(self, dir=None):
        """Create ImageDataGenerator."""
        datagen = ImageDataGenerator(rescale=1. / 255)

        if dir:
            return datagen.flow_from_directory(
                dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical')

        return datagen

    def build_model(self, num_classes):
        """Build the ViT model."""
        vit_model = vit.vit_b32(
            image_size=self.img_size[0],
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

    def train_model(self, train_generator, val_generator, epochs=3000, patience_lr=160, patience_es=50, min_lr=1e-6,
                    min_delta=1e-4):
        """Train the model with callbacks."""
        self._check_model()

        # Create a log directory with a timestamp
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(self.log_dir, current_time)
        os.makedirs(log_dir, exist_ok=True)

        callbacks = self._create_callbacks(log_dir, current_time, patience_lr, min_lr, min_delta, patience_es)

        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks)
        return history

    def _create_callbacks(self, log_dir, current_time, patience_lr, min_lr, min_delta, patience_es):
        """Create Keras callbacks."""
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True,
                                           update_freq='epoch', profile_batch=0)
        reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_lr, min_lr=min_lr,
                                               min_delta=min_delta)
        model_checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(self.model_dir, "{}_best_model.h5".format(current_time)),
            save_best_only=True, monitor='val_loss', mode='min', verbose=1)
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience_es, restore_best_weights=True)

        return [tensorboard_callback, reduce_lr_callback, model_checkpoint_callback, early_stopping_callback]

    def evaluate_model(self, test_datagen):
        """Evaluate the model."""
        self._check_model()

        test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical')

        test_loss, test_acc = self.model.evaluate(test_generator)
        print('Test accuracy:', test_acc)
        return test_loss, test_acc

    def save_model(self, filename='models/skinvestigator_nano_40MB_91_38_acc.h5'):
        """Save the model."""
        self._check_model()
        self.model.save(filename)

    def _check_model(self):
        """Check if model has been built."""
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")
