import numpy as np

from skinvestigatorai.models.gan_model import GAN


class GANAugmentor:
    def __init__(self):
        self.gan = GAN(img_shape=(160, 160, 3), latent_dim=100)

    def generate_samples(self, num_samples):
        noise = np.random.normal(0, 1, (num_samples, self.gan.latent_dim))
        generated_images = self.gan.generator.predict(noise)
        return generated_images

    def augment_data(self, original_data, augmentation_factor=2):
        augmented_data = []
        num_generated = original_data.shape[0] * augmentation_factor

        generated_images = self.generate_samples(num_generated)
        augmented_data.extend(generated_images)
        return np.array(augmented_data)
