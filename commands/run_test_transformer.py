import tensorflow as tf
import numpy as np
import os

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Hyperparameters
IMAGE_SIZE = 224
PATCH_SIZE = 14
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM = 128
NUM_HEADS = 8
TRANSFORMER_LAYERS = 16
MLP_UNITS = [PROJECTION_DIM * 2, PROJECTION_DIM]
BATCH_SIZE = 26
EPOCHS = 50
LEARNING_RATE = 3e-4
AUTO = tf.data.AUTOTUNE


# Load and preprocess data
def parse_image(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def create_dataset(data_dir, batch_size):
    dataset = tf.data.Dataset.list_files(data_dir + "/*/*.jpg", shuffle=True)
    class_names = sorted([item.name for item in os.scandir(data_dir) if item.is_dir()])
    num_classes = len(class_names)

    def get_label(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return tf.argmax(tf.cast(tf.equal(class_names, parts[-2]), tf.int64))

    dataset = dataset.map(lambda x: (x, get_label(x)), num_parallel_calls=AUTO)
    dataset = dataset.map(parse_image, num_parallel_calls=AUTO)
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)
    return dataset, num_classes


# Load the data
data_dir = "data/train"
train_dataset, num_classes = create_dataset(data_dir, BATCH_SIZE)

# Split the dataset into train and validation
train_size = int(0.8 * tf.data.experimental.cardinality(train_dataset).numpy())
val_size = tf.data.experimental.cardinality(train_dataset).numpy() - train_size

train_dataset = train_dataset.take(train_size)
val_dataset = train_dataset.skip(train_size)

# Print dataset info for debugging
for images, labels in train_dataset.take(1):
    print("Image shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Labels:", labels)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])


# Patch extraction layer
class PatchExtractor(tf.keras.layers.Layer):
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
class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


# Build the ViT model
def create_vit_model():
    inputs = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    augmented = data_augmentation(inputs)
    patches = PatchExtractor(PATCH_SIZE)(augmented)
    encoded_patches = PatchEncoder(NUM_PATCHES, PROJECTION_DIM)(patches)

    for _ in range(TRANSFORMER_LAYERS):
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
        )(x1, x1)
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = tf.keras.layers.Dense(MLP_UNITS[0], activation=tf.nn.gelu)(x3)
        x3 = tf.keras.layers.Dropout(0.5)(x3)
        x3 = tf.keras.layers.Dense(MLP_UNITS[1], activation=tf.nn.gelu)(x3)
        x3 = tf.keras.layers.Dropout(0.3)(x3)
        encoded_patches = tf.keras.layers.Add()([x3, x2])

    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = tf.keras.layers.Flatten()(representation)
    representation = tf.keras.layers.Dropout(0.5)(representation)

    features = tf.keras.layers.Dense(MLP_UNITS[0], activation=tf.nn.gelu)(representation)
    features = tf.keras.layers.Dropout(0.5)(features)
    logits = tf.keras.layers.Dense(num_classes)(features)

    return tf.keras.Model(inputs=inputs, outputs=logits)


# Create and compile the model
model = create_vit_model()
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.CategoricalFocalCrossentropy(from_logits=True),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.Recall(name="recall", class_id=1),  # Assuming positive class is 1
    ],
)


# Train the model
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_recall",
            patience=10,
            restore_best_weights=True,
        ),
    ],
)

# Evaluate the model
test_loss, test_accuracy, test_recall = model.evaluate(val_dataset)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test recall: {test_recall:.4f}")

# Save the model
model.save("skin_lesion_classifier.h5")