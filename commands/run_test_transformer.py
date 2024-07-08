import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Hyperparameters
IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM = 64
NUM_HEADS = 8
TRANSFORMER_LAYERS = 8
MLP_UNITS = [PROJECTION_DIM * 2, PROJECTION_DIM]
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 3e-4


# Load and preprocess data
def load_data(data_dir):
    image_data = []
    labels = []
    class_names = os.listdir(data_dir)

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            img = tf.keras.preprocessing.image.img_to_array(img)
            image_data.append(img)
            labels.append(class_names.index(class_name))

    return np.array(image_data), np.array(labels), class_names


# Load the data
data_dir = "data/train"
images, labels, class_names = load_data(data_dir)
num_classes = len(class_names)

# Split the data
train_split = 0.8
num_samples = len(images)
num_train = int(num_samples * train_split)

x_train, x_val = images[:num_train], images[num_train:]
y_train, y_val = labels[:num_train], labels[num_train:]

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])


# Patch extraction layer
class PatchExtractor(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


# Patch encoding layer
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


# Build the ViT model
def create_vit_model():
    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    augmented = data_augmentation(inputs)
    patches = PatchExtractor(PATCH_SIZE)(augmented)
    encoded_patches = PatchEncoder(NUM_PATCHES, PROJECTION_DIM)(patches)

    for _ in range(TRANSFORMER_LAYERS):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(MLP_UNITS[0], activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Dense(MLP_UNITS[1], activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    features = layers.Dense(MLP_UNITS[0], activation=tf.nn.gelu)(representation)
    features = layers.Dropout(0.5)(features)
    logits = layers.Dense(num_classes)(features)

    return tf.keras.Model(inputs=inputs, outputs=logits)


# Create and compile the model
model = create_vit_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

# Train the model
history = model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_recall",
            patience=10,
            restore_best_weights=True,
        ),
    ],
)

# Evaluate the model
test_loss, test_accuracy, test_recall = model.evaluate(x_val, y_val)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test recall: {test_recall:.4f}")

# Save the model
model.save("skin_lesion_classifier.h5")