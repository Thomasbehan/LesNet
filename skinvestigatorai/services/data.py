import math
import os

import tensorflow as tf
from PIL import Image

from skinvestigatorai.config.model import ModelConfig


class Data:
    def __init__(self):
        self.img_size = ModelConfig.IMG_SIZE

    def verify_images(self, directory):
        invalid_images = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.JPG', '.png', '.jpg', '.jpeg')):
                    try:
                        img_path = os.path.join(root, file)
                        with Image.open(img_path) as img:
                            img.verify()
                    except (Image.UnidentifiedImageError, IOError):
                        invalid_images.append(img_path)
                        os.remove(img_path)
                        print('Deleted invalid file:', img_path)
        return invalid_images

    def prepare_for_training(self, ds, take_num=None, augment=False, cache=True, shuffle_buffer_size=1000,
                             repeat=False):
        if take_num:
            ds = ds.take(take_num)
        if cache:
            ds = ds.cache()
        if augment:
            ds = ds.map(self.augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        ds = ds.repeat() if repeat else ds
        ds = ds.batch(ModelConfig.BATCH_SIZE)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    def augment_image(self, image, label):
        """Apply image augmentation on the fly, ensuring that images remain in the correct range."""
        # Random horizontal and vertical flipping
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

        # Random 90-degree rotation
        image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

        # Random cropping and zooming
        crop_size = tf.random.uniform(shape=[], minval=int(image.shape[0] * 0.7), maxval=int(image.shape[0]),
                                      dtype=tf.int32)
        image = tf.image.random_crop(image, size=[crop_size, crop_size, 3])
        image = tf.image.resize(image, [image.shape[0], image.shape[1]])  # Resize back to original dimensions

        # Clip to ensure pixel values are still in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label

    def save_augmented_images(self, paths, labels, output_dir, total_augments_needed):
        """Save augmented images directly to file, distributing augmentations evenly across images."""
        num_original_images = len(paths)
        if num_original_images == 0:
            return

        augments_per_image = min(math.ceil(total_augments_needed / num_original_images), ModelConfig.MAX_AUG_PER_IMAGE)
        print(f"Augmenting images in {labels[0]}: {total_augments_needed} needed")

        for path, label in zip(paths, labels):
            folder_path = os.path.join(output_dir, label)
            base_filename = os.path.splitext(os.path.basename(path))[0]
            image = self.load_and_preprocess_image(path)

            for i in range(augments_per_image):
                augmented_image, _ = self.augment_image(image, label)
                augmented_image = tf.image.convert_image_dtype(augmented_image, tf.uint8)
                augmented_image = Image.fromarray(augmented_image.numpy(), 'RGB')
                augmented_filename = f"{base_filename}_augmented_{i}.jpg"
                save_location = os.path.join(folder_path, augmented_filename)
                augmented_image.save(save_location)

    def generate_augmented_images(self, paths, labels, augment_times=5):
        """Generate new images by applying augmentation to existing images."""
        for path, label in zip(paths, labels):
            image = self.load_and_preprocess_image(path)
            for _ in range(augment_times):
                augmented_image, _ = self.augment_image(image, label)
                yield augmented_image, tf.cast(label, tf.string)

    def load_and_preprocess_image(self, path):
        image = tf.io.read_file(path)
        return self.preprocess_image(image)

    def preprocess_image(self, image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.img_size)
        image /= 255.0
        return image

    def load_dataset(self):

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            ModelConfig.TRAIN_DIR,
            validation_split=0.2,
            subset="training",
            seed=42,
            label_mode='categorical',
            labels='inferred',
            image_size=self.img_size,
            batch_size=ModelConfig.BATCH_SIZE
        )

        validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
            ModelConfig.TRAIN_DIR,
            validation_split=0.2,
            subset="validation",
            seed=42,
            label_mode='categorical',
            labels='inferred',
            image_size=self.img_size,
            batch_size=ModelConfig.BATCH_SIZE
        )
        # Rescale pixel values (0, 255) to [0, 1]
        normalization_layer = tf.keras.layers.Rescaling(1. / 255)

        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        validation_ds = validation_ds.map(lambda x, y: (normalization_layer(x), y))

        return train_ds, validation_ds

    def load_support_set(self, support_dir=ModelConfig.SUPPORT_DIR):
        """Load support set for few-shot learning."""
        support_ds = tf.keras.preprocessing.image_dataset_from_directory(
            support_dir,
            label_mode='categorical',
            labels='inferred',
            image_size=self.img_size,
            batch_size=ModelConfig.BATCH_SIZE
        )

        support_images = []
        support_labels = []
        for images, labels in support_ds:
            support_images.append(images)
            support_labels.append(labels)

        support_images = tf.concat(support_images, axis=0)
        support_labels = tf.concat(support_labels, axis=0)

        return support_images, support_labels

    def load_query_set(self, query_dir=ModelConfig.QUERY_DIR):
        """Load query set for few-shot learning."""
        query_ds = tf.keras.preprocessing.image_dataset_from_directory(
            query_dir,
            label_mode='categorical',
            labels='inferred',
            image_size=self.img_size,
            batch_size=ModelConfig.BATCH_SIZE
        )

        query_images = []
        query_labels = []
        for images, labels in query_ds:
            query_images.append(images)
            query_labels.append(labels)

        query_images = tf.concat(query_images, axis=0)
        query_labels = tf.concat(query_labels, axis=0)

        return query_images, query_labels