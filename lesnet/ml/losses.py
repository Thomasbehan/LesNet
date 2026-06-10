"""Cost-aware focal loss for sensitivity-first training (paper §5.2).

Class-weighted multi-class focal loss operating on logits. Triage class weights combine
inverse-frequency with an explicit malignant cost multiplier so that missing a
malignancy is penalised far more than a false alarm.
"""
import numpy as np
import tensorflow as tf


def make_focal_loss(class_weights=None, gamma=2.0):
    """Return a Keras loss for one-hot targets and raw logits."""
    weights_tensor = None
    if class_weights is not None:
        weights_tensor = tf.constant(np.asarray(class_weights, dtype=np.float32))

    def focal_loss(y_true, y_logits):
        y_true = tf.cast(y_true, tf.float32)
        probabilities = tf.clip_by_value(tf.nn.softmax(y_logits, axis=-1), 1e-7, 1.0)
        cross_entropy = -y_true * tf.math.log(probabilities)
        modulating = tf.pow(1.0 - probabilities, gamma)
        loss = modulating * cross_entropy
        if weights_tensor is not None:
            loss = loss * weights_tensor
        return tf.reduce_sum(loss, axis=-1)

    return focal_loss


def triage_class_weights(triage_labels, malignant_cost=2.0, malignant_index=2):
    """Capped inverse-frequency weights with a mild extra weight on the malignant class.

    Raw inverse-frequency on extreme prevalence (e.g. 94%/0.6%/5%) drives the majority
    class weight toward zero, which destabilises training and wrecks calibration. We cap
    the weights to a sane band; sensitivity is delivered by the operating threshold, not
    by extreme loss weighting.
    """
    labels = np.asarray(triage_labels, dtype=int)
    n_classes = 3
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (n_classes * counts)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.5, 3.0)
    weights[malignant_index] = min(weights[malignant_index] * malignant_cost, 4.0)
    return weights.astype(np.float32)
