import pathlib

import matplotlib.pyplot as plotter_lib
import tensorflow as tflow
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

plotter_lib.figure(figsize=(10, 10))

from skinvestigatorai.config.model import ModelConfig


img_height, img_width = 180, 180

batch_size = 32

train_ds = tflow.keras.preprocessing.image_dataset_from_directory(
    ModelConfig.TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

validation_ds = tflow.keras.preprocessing.image_dataset_from_directory(
    ModelConfig.TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

epochs = 10

## Build
resnet_model = Sequential()

pretrained_model_for_demo = tflow.keras.applications.ResNet50(include_top=False,

                                                              input_shape=(180, 180, 3),

                                                              pooling='avg', classes=5,

                                                              weights='imagenet')

for each_layer in pretrained_model_for_demo.layers:
    each_layer.trainable = False

resnet_model.add(pretrained_model_for_demo)
resnet_model.add(Flatten())

resnet_model.add(Dense(512, activation='relu'))

resnet_model.add(Dense(27, activation='softmax'))

## Train
resnet_model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history = resnet_model.fit(train_ds, validation_data=validation_ds, epochs=epochs)

## Eval
plotter_lib.figure(figsize=(8, 8))

epochs_range = range(epochs)

plotter_lib.plot(epochs_range, history.history['accuracy'], label="Training Accuracy")

plotter_lib.plot(epochs_range, history.history['val_accuracy'], label="Validation Accuracy")

plotter_lib.axis(ymin=0.4, ymax=1)

plotter_lib.grid()

plotter_lib.title('Model Accuracy')

plotter_lib.ylabel('Accuracy')

plotter_lib.xlabel('Epochs')

plotter_lib.legend(['train', 'validation'])

# plotter_lib.show()

plotter_lib.savefig('output-plot.png')
