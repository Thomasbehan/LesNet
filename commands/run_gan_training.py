import os

import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from skinvestigatorai.config.data import DataConfig
from skinvestigatorai.config.model import ModelConfig
from skinvestigatorai.models.gan_model import GAN


def load_images_from_folder(folder, img_size):
    images = []
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.JPG', '.jpg', '.jpeg', '.png')):
                img_path = os.path.join(subdir, file)
                img = load_img(img_path, target_size=img_size)
                img_array = img_to_array(img)
                img_array = (img_array - 127.5) / 127.5  # Normalize to [-1, 1]
                images.append(img_array)
    return np.array(images)


def train_gan_model(root_folder, img_size=(160, 160), latent_dim=100, epochs=10000, batch_size=128, sample_interval=200):
    gan = GAN(img_shape=img_size + (3,), latent_dim=latent_dim)

    # Load and preprocess images
    dataset = load_images_from_folder(root_folder, img_size=img_size)
    print(f"Loaded {dataset.shape[0]} images from {root_folder}.")

    gan.train(X_train=dataset, epochs=epochs, batch_size=batch_size, sample_interval=sample_interval)
    gan.save_model()


if __name__ == "__main__":
    # Set up the paths and parameters
    root_folder = os.path.join(DataConfig.OUTPUT_DIR, "train")
    img_size = ModelConfig.IMG_SIZE
    latent_dim = 100
    epochs = 10000
    batch_size = 128
    sample_interval = 20

    # Start training the GAN model
    train_gan_model(root_folder, img_size=img_size, latent_dim=latent_dim, epochs=epochs, batch_size=batch_size, sample_interval=sample_interval)
