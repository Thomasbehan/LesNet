import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class GAN:
    def __init__(self, img_shape, latent_dim):
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.gan = self.build_gan()

    def build_generator(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(256, input_dim=self.latent_dim))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Dense(512))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Dense(1024))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(layers.Reshape(self.img_shape))
        return model

    def build_discriminator(self):
        model = tf.keras.Sequential()
        model.add(layers.Flatten(input_shape=self.img_shape))
        model.add(layers.Dense(512))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dense(256))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def build_gan(self):
        self.discriminator.trainable = False
        model = tf.keras.Sequential([self.generator, self.discriminator])
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def save_model(self):
        self.generator.save('generator_model.h5')
        self.discriminator.save('discriminator_model.h5')
        self.gan.save('gan_model.h5')

    def train(self, X_train, epochs, batch_size=128, sample_interval=200):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, valid)

            if epoch % sample_interval == 0:
                print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}] [G loss: {g_loss}]")
