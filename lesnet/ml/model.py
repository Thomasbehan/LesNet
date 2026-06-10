"""Multi-task triage model: image + metadata -> triage logits + fine logits (paper §5.1, §5.4).

A pretrained backbone (EfficientNetV2-S by default) produces image features; a small MLP
encodes metadata; the fused features feed a primary 3-way triage head and an auxiliary
fine-grained head. A 'tiny' backbone is available for fast CPU smoke runs.
"""
import tensorflow as tf
from tensorflow.keras import Input, Model, layers


def _build_backbone(config, image_input):
    if config.backbone == 'tiny':
        features = image_input
        for filters in (16, 32, 64):
            features = layers.Conv2D(filters, 3, padding='same', activation='swish')(features)
            features = layers.MaxPooling2D()(features)
        return layers.GlobalAveragePooling2D()(features)

    backbone = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights='imagenet' if config.pretrained else None,
        input_tensor=image_input,
        include_preprocessing=False,
        pooling='avg',
    )
    trainable_from = max(len(backbone.layers) - config.backbone_trainable_layers, 0)
    for layer in backbone.layers[:trainable_from]:
        layer.trainable = False
    return backbone.output


def build_triage_model(config, n_fine, metadata_dim):
    image_input = Input(shape=(config.image_size[0], config.image_size[1], 3), name='image')
    metadata_input = Input(shape=(metadata_dim,), name='metadata')

    image_features = _build_backbone(config, image_input)
    metadata_features = layers.Dense(32, activation='swish')(metadata_input)

    fused = layers.Concatenate()([image_features, metadata_features])
    shared = layers.Dense(config.shared_units, activation='swish')(fused)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(config.dropout, name='shared_features')(shared)

    triage_logits = layers.Dense(3, name='triage')(shared)
    fine_logits = layers.Dense(max(n_fine, 1), name='fine')(shared)

    return Model(
        inputs={'image': image_input, 'metadata': metadata_input},
        outputs={'triage': triage_logits, 'fine': fine_logits},
        name='lesnet_triage',
    )


def feature_model(model):
    """Sub-model exposing the shared feature vector (for OOD embeddings)."""
    return Model(inputs=model.inputs, outputs=model.get_layer('shared_features').output)


def triage_logits_model(model):
    """Sub-model exposing triage logits only (for temperature calibration)."""
    return Model(inputs=model.inputs, outputs=model.get_layer('triage').output)


def fine_logits_model(model):
    """Sub-model exposing the auxiliary fine-grained diagnosis logits."""
    return Model(inputs=model.inputs, outputs=model.get_layer('fine').output)
