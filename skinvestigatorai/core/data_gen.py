import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataGen(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size, img_height, img_width, augmentations):
        self.directory = directory
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augmentations
        self.datagen = ImageDataGenerator(rescale=1. / 255)
        self.generator = self.datagen.flow_from_directory(directory,
                                                          target_size=(self.img_height, self.img_width),
                                                          batch_size=self.batch_size,
                                                          class_mode='categorical')

    @property
    def class_indices(self):
        return self.generator.class_indices

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, index):
        x, y = self.generator[index]
        return np.stack([self.augment(image=i)["image"] for i in x], axis=0), y
