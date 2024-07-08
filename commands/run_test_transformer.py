import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow import keras
from tensorflow.keras import layers

# Define the path to your dataset
train_dir = 'data/train'

# Load training data
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),  # Resize images to 224x224
    batch_size=32,
    label_mode='int'
)

# Split the dataset into training and validation sets
val_size = int(0.2 * len(train_dataset))
train_dataset = train_dataset.skip(val_size)
val_dataset = train_dataset.take(val_size)

# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

# Preprocess inputs for the model
preprocess_input = tf.keras.applications.resnet50.preprocess_input


# Apply the preprocessing and data augmentation
def preprocess(image, label):
    image = preprocess_input(image)
    image = data_augmentation(image)
    return image, label


train_dataset = train_dataset.map(preprocess)
val_dataset = val_dataset.map(preprocess)

# Prefetch data for performance
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


class VisionTransformer(tf.keras.Model):
    def __init__(self, num_classes, num_layers, d_model, num_heads, mlp_dim, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.patch_size = 16
        self.num_patches = (224 // self.patch_size) ** 2

        # Define projection layer for patches
        self.projection = layers.Dense(units=d_model, name="projection")

        # Initialize class token and position embeddings
        self.cls_token = self.add_weight(shape=(1, 1, d_model), initializer="random_normal", trainable=True,
                                         name="cls_token")
        self.position_embedding = self.add_weight(shape=(1, self.num_patches + 1, d_model), initializer="random_normal",
                                                  trainable=True, name="position_embedding")

        # Transformer blocks
        self.transformer_blocks = []
        for _ in range(num_layers):
            self.transformer_blocks.append(layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model))

        # MLP head for classification
        self.mlp_head = tf.keras.Sequential([
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(units=mlp_dim, activation="relu"),
            layers.Dropout(rate=dropout),
            layers.Dense(units=num_classes, name="output")
        ])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        # Extract patches from images
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        # Reshape patches for projection
        patches = tf.reshape(patches, [batch_size, self.num_patches, -1])

        # Project patches to d_model dimension
        x = self.projection(patches)

        # Add class token to the beginning of sequence
        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, self.d_model])
        x = tf.concat([cls_tokens, x], axis=1)

        # Add position embeddings
        x += self.position_embedding

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, x)  # Self-attention mechanism

        # Extract class token for classification
        cls_token_final = x[:, 0]

        # MLP head for classification
        output = self.mlp_head(cls_token_final)

        return output


# Instantiate the Vision Transformer model
# Get the list of all directories (classes) in the train_dir
classes = os.listdir(train_dir)

# Count the number of classes (folders)
num_classes = len(classes)
model = VisionTransformer(num_classes=num_classes, num_layers=8, d_model=64, num_heads=4, mlp_dim=128, dropout=0.1)

# Calculate class weights
y_train = np.concatenate([y for x, y in train_dataset], axis=0)
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    class_weight=class_weights
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_dataset)
print(f'Validation loss: {val_loss}')
print(f'Validation accuracy: {val_accuracy}')

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title('Loss')
plt.subplot(1, 2, 2)
plt.plot(history.history['sparse_categorical_accuracy'], label='train_accuracy')
plt.plot(history.history['val_sparse_categorical_accuracy'], label='val_accuracy')
plt.legend()
plt.title('Accuracy')
plt.show()
