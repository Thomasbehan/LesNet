"""Training orchestrator: fit the multi-task model, calibrate, choose the operating
point, fit the OOD gate, and save the artifact bundle (paper §5.1-§5.5)."""
import os

import numpy as np
import tensorflow as tf

from lesnet.ml import artifacts, metrics
from lesnet.ml.calibration import TemperatureScaler, softmax
from lesnet.ml.conformal import SplitConformalClassifier
from lesnet.ml.data_loader import make_dataset
from lesnet.ml.features import METADATA_DIM
from lesnet.ml.losses import make_focal_loss, triage_class_weights
from lesnet.ml.model import build_triage_model, feature_model, triage_logits_model
from lesnet.ml.ood import MahalanobisOODDetector
from lesnet.ml.taxonomy import MALIGNANT, TRIAGE_CLASSES, build_fine_vocabulary
from lesnet.ml.triage import triage_decision

MAX_OOD_FIT_BATCHES = 64


def _compile(model, config, class_weights, n_fine):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        loss={
            'triage': make_focal_loss(class_weights, config.focal_gamma),
            'fine': make_focal_loss(None, config.focal_gamma),
        },
        loss_weights={'triage': 1.0, 'fine': config.aux_loss_weight},
        metrics={'triage': [tf.keras.metrics.CategoricalAccuracy(name='accuracy')]},
    )


def train(config, records_train, records_val):
    tf.keras.utils.set_random_seed(config.seed)

    fine_vocabulary = build_fine_vocabulary(records_train)
    train_dataset, n_fine, train_triage = make_dataset(records_train, config, fine_vocabulary, training=True)
    val_dataset, _, val_triage = make_dataset(records_val, config, fine_vocabulary, training=False)

    model = build_triage_model(config, n_fine, METADATA_DIM)
    class_weights = triage_class_weights(train_triage, config.malignant_cost, MALIGNANT)
    _compile(model, config, class_weights, n_fine)

    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=max(2, config.epochs // 5), restore_best_weights=True)]
    model.fit(train_dataset, validation_data=val_dataset, epochs=config.epochs,
              callbacks=callbacks, verbose=2)

    # Calibrate triage logits on the validation set.
    logits = triage_logits_model(model).predict(val_dataset, verbose=0)
    scaler = TemperatureScaler().fit(logits, val_triage)
    calibrated = softmax(logits / scaler.temperature)

    # Sensitivity-first operating point for malignant-vs-rest.
    p_malignant = calibrated[:, MALIGNANT]
    y_malignant = (val_triage == MALIGNANT).astype(int)
    operating_threshold = metrics.select_threshold_for_sensitivity(
        y_malignant, p_malignant, config.target_sensitivity)

    # Conformal calibration on the (calibrated) 3-way probabilities.
    conformal = SplitConformalClassifier(alpha=config.conformal_alpha).calibrate(calibrated, val_triage)

    # OOD detector on shared embeddings from training data.
    embeddings = feature_model(model).predict(train_dataset.take(MAX_OOD_FIT_BATCHES), verbose=0)
    ood = MahalanobisOODDetector().fit(embeddings)

    bundle = {
        'config': vars(config),
        'metadata_dim': METADATA_DIM,
        'n_fine': n_fine,
        'label_maps': {'fine_vocabulary': fine_vocabulary, 'triage_classes': TRIAGE_CLASSES},
        'calibration': {'temperature': scaler.temperature},
        'conformal': {'alpha': conformal.alpha, 'q_hat': conformal.q_hat},
        'thresholds': {
            'operating_threshold': operating_threshold,
            'target_sensitivity': config.target_sensitivity,
            'refer': float(operating_threshold * 0.5),
            'urgent': float(operating_threshold),
            'abstain_band': [float(max(operating_threshold - 0.1, 0.0)),
                             float(min(operating_threshold + 0.1, 1.0))],
        },
        'ood': {'mahalanobis': ood.to_dict(), 'min_laplacian_variance': 1e-4},
    }
    os.makedirs(config.artifacts_dir, exist_ok=True)
    model.save(artifacts.model_path(config.artifacts_dir))
    artifacts.save_bundle(config.artifacts_dir, bundle)
    return model, bundle


def preview_triage(calibrated_probabilities, thresholds):
    """Convenience: map calibrated malignant probabilities to triage actions."""
    band = tuple(thresholds['abstain_band'])
    return [
        triage_decision(float(row[MALIGNANT]), valid_image=True,
                        refer_threshold=thresholds['refer'],
                        urgent_threshold=thresholds['urgent'], abstain_band=band)
        for row in np.atleast_2d(calibrated_probabilities)
    ]
