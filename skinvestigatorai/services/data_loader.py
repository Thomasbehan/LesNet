import math
import os

import tensorflow as tf
from PIL import Image

from skinvestigatorai.config.model import ModelConfig


class DataLoader:
    def __init__(self):
        self.img_size = ModelConfig.IMG_SIZE
        self.patch_size = ModelConfig.PATCH_SIZE
        self.num_patches = (self.img_size[0] // self.patch_size) ** 2

    def verify_images(self, directory):
        """Verify images in the directory and remove invalid files."""
        invalid_images = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(root, file)
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                    except (Image.UnidentifiedImageError, IOError):
                        invalid_images.append(img_path)
                        os.remove(img_path)
                        print(f'Deleted invalid file: {img_path}')
        return invalid_images

    def load_dataset(self, validation_split=0.2, seed=42):
        """Load and preprocess the dataset from directories."""
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            ModelConfig.TRAIN_DIR,
            validation_split=validation_split,
            subset="training",
            seed=seed,
            label_mode='categorical',
            image_size=self.img_size,
            batch_size=ModelConfig.BATCH_SIZE
        )

        validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
            ModelConfig.TRAIN_DIR,
            validation_split=validation_split,
            subset="validation",
            seed=seed,
            label_mode='categorical',
            image_size=self.img_size,
            batch_size=ModelConfig.BATCH_SIZE
        )

        train_ds = train_ds.map(self.preprocess_and_patch_image, num_parallel_calls=tf.data.AUTOTUNE)
        validation_ds = validation_ds.map(self.preprocess_and_patch_image, num_parallel_calls=tf.data.AUTOTUNE)

        return train_ds, validation_ds

    def preprocess_and_patch_image(self, image, label):
        """Preprocess and divide the image into patches."""
        image = self.preprocess_image(image)
        patches = self.create_patches(image)
        return patches, label

    def preprocess_image(self, image):
        """Resize and normalize the image."""
        image = tf.image.resize(image, self.img_size)
        image /= 255.0  # Normalize to [0, 1]
        return image

    def create_patches(self, image):
        """Divide the image into patches for the Vision Transformer."""
        patches = tf.image.extract_patches(
            images=tf.expand_dims(image, 0),
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches = tf.reshape(patches, [self.num_patches, -1])
        return patches

    def prepare_for_training(self, ds, take_num=None, augment=False, cache=True, shuffle_buffer_size=1000,
                             repeat=False):
        """Prepare dataset for training with optional augmentations."""
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
        """Apply image augmentation on the fly."""
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        crop_size = tf.random.uniform(shape=[], minval=int(self.img_size[0] * 0.7), maxval=self.img_size[0],
                                      dtype=tf.int32)
        image = tf.image.random_crop(image, size=[crop_size, crop_size, 3])
        image = tf.image.resize(image, self.img_size)
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

    def load_and_preprocess_image(self, path):
        """Load an image from a path and preprocess it."""
        image = tf.io.read_file(path)
        return self.preprocess_image(image)

    def generate_augmented_images(self, paths, labels, augment_times=5):
        """Generate new images by applying augmentation to existing images."""
        for path, label in zip(paths, labels):
            image = self.load_and_preprocess_image(path)
            for _ in range(augment_times):
                augmented_image, _ = self.augment_image(image, label)
                yield augmented_image, tf.cast(label, tf.string)
