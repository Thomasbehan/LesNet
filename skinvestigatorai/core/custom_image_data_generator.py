from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import UnidentifiedImageError
import numpy as np


class CustomImageDataGenerator(ImageDataGenerator):
    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=self.dtype)
        batch_y = np.zeros((len(index_array),) + self.label_shape, dtype=self.dtype)

        for i, j in enumerate(index_array):
            try:
                x, y = self._get_transformed_sample(self.filepaths[j], self.labels[j])
                batch_x[i] = x
                batch_y[i] = y
            except UnidentifiedImageError:
                print(f"Skipping problematic image file at index {j}")
        return batch_x, batch_y
