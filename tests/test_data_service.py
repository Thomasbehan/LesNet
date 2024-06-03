import os
from unittest.mock import patch, MagicMock, mock_open

import pytest
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from skinvestigatorai.config.model import ModelConfig
from skinvestigatorai.services.data import Data


@pytest.fixture
def data_instance():
    return Data()


def test_verify_images(data_instance):
    with patch('os.walk') as mock_walk, \
            patch('PIL.Image.open') as mock_open_image, \
            patch('os.remove') as mock_remove:
        mock_walk.return_value = [
            ('root', ('dir1',), ('file1.JPG', 'file2.png', 'file3.txt'))
        ]

        mock_open_image.side_effect = [MagicMock(), UnidentifiedImageError, MagicMock()]
        invalid_images = data_instance.verify_images('root')

        assert invalid_images == [os.path.join('root', 'file2.png')]
        mock_remove.assert_called_once_with(os.path.join('root', 'file2.png'))


def test_save_augmented_images(data_instance):
    paths = ["image1.jpg", "image2.jpg"]
    labels = ["label1", "label2"]

    with patch('os.path.exists', return_value=True), \
            patch('os.makedirs'), \
            patch('builtins.open', mock_open()), \
            patch.object(data_instance, 'load_and_preprocess_image', return_value=tf.random.uniform([160, 160, 3])), \
            patch.object(data_instance, 'augment_image', return_value=(tf.random.uniform([160, 160, 3]), "label")), \
            patch.object(Image, 'fromarray'), \
            patch.object(Image.Image, 'save'):
        data_instance.save_augmented_images(paths, labels, "output_dir", 10)


def test_generate_augmented_images(data_instance):
    paths = ["image1.jpg", "image2.jpg"]
    labels = ["label1", "label2"]

    with patch.object(data_instance, 'load_and_preprocess_image', return_value=tf.random.uniform([160, 160, 3])), \
            patch.object(data_instance, 'augment_image', return_value=(tf.random.uniform([160, 160, 3]), "label")):
        augmented_images = list(data_instance.generate_augmented_images(paths, labels, augment_times=3))
        assert len(augmented_images) == 6  # 3 augmentations per image


def test_load_dataset(data_instance):
    with patch('tensorflow.keras.preprocessing.image_dataset_from_directory') as mock_dataset:
        mock_dataset.return_value = MagicMock()
        train_ds, validation_ds = data_instance.load_dataset()

        assert train_ds is not None
        assert validation_ds is not None
        mock_dataset.assert_any_call(
            ModelConfig.TRAIN_DIR,
            validation_split=0.2,
            subset="training",
            seed=42,
            label_mode='categorical',
            labels='inferred',
            image_size=ModelConfig.IMG_SIZE,
            batch_size=ModelConfig.BATCH_SIZE
        )
        mock_dataset.assert_any_call(
            ModelConfig.TRAIN_DIR,
            validation_split=0.2,
            subset="validation",
            seed=42,
            label_mode='categorical',
            labels='inferred',
            image_size=ModelConfig.IMG_SIZE,
            batch_size=ModelConfig.BATCH_SIZE
        )
