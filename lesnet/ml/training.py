"""Training orchestrator: fit the multi-task model (optionally until all metric targets
are met), calibrate, choose the operating point, fit the OOD gate, and save the artifact
bundle (paper §5.1-§5.5). TensorBoard logging is enabled for live monitoring."""
import os

import tensorflow as tf

from lesnet.ml import artifacts, metrics
from lesnet.ml.calibration import TemperatureScaler, softmax
from lesnet.ml.conformal import SplitConformalClassifier
from lesnet.ml.data_loader import filter_valid, make_dataset
from lesnet.ml.evaluation import build_report
from lesnet.ml.features import METADATA_DIM, normalize_site
from lesnet.ml.losses import make_focal_loss, triage_class_weights
from lesnet.ml.model import build_triage_model, feature_model, triage_logits_model
from lesnet.ml.ood import MahalanobisOODDetector
from lesnet.ml.taxonomy import MALIGNANT, TRIAGE_CLASSES, build_fine_vocabulary

MAX_OOD_FIT_BATCHES = 64


def _compile(model, config, class_weights):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        loss={
            'triage': make_focal_loss(class_weights, config.focal_gamma),
            'fine': make_focal_loss(None, config.focal_gamma),
        },
        loss_weights={'triage': 1.0, 'fine': config.aux_loss_weight},
        metrics={'triage': [tf.keras.metrics.CategoricalAccuracy(name='accuracy')]},
    )


def _age_band(age):
    return 'unknown' if not age else f"{int(age) // 20 * 20}s"


def _subgroups(records):
    return {
        'fitzpatrick': [str(r.fitzpatrick) if r.fitzpatrick else 'unknown' for r in records],
        'site': [normalize_site(r.anatomical_site) for r in records],
        'age_band': [_age_band(r.age) for r in records],
    }


def _validation_report(model, val_dataset, val_triage, val_records, config):
    logits = triage_logits_model(model).predict(val_dataset, verbose=0)
    scaler = TemperatureScaler().fit(logits, val_triage)
    probabilities = softmax(logits / scaler.temperature)
    return build_report(val_triage, probabilities, subgroups=_subgroups(val_records),
                        target_sensitivity=config.target_sensitivity)


def _targets_met(report, config):
    met = report['sensitivity'] >= config.target_sensitivity
    met = met and report['specificity'] >= config.target_specificity
    met = met and report['ece'] <= config.max_ece
    if config.require_fairness_gate:
        met = met and report['fairness_gate']['passed']
    return met


def _gated_fit(model, train_dataset, val_dataset, val_triage, val_records, config, callbacks, log_dir):
    """Train in rounds until every target is in the ideal range, or the budget is spent.

    Tracks the best-generalising checkpoint (lowest val_loss) across all rounds; if the
    budget is spent without meeting the targets, the best weights are restored rather
    than keeping a possibly-overfit final epoch.
    """
    writer = tf.summary.create_file_writer(os.path.join(log_dir, 'gate'))
    best_weights_path = os.path.join(log_dir, 'best.weights.h5')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        best_weights_path, monitor='val_loss', save_best_only=True, save_weights_only=True)
    round_callbacks = callbacks + [checkpoint]

    total_epochs = 0
    targets_met = False
    while True:
        model.fit(train_dataset, validation_data=val_dataset, epochs=config.epochs_per_round,
                  callbacks=round_callbacks, verbose=2)
        total_epochs += config.epochs_per_round
        report = _validation_report(model, val_dataset, val_triage, val_records, config)
        with writer.as_default():
            tf.summary.scalar('gate/sensitivity', report['sensitivity'], step=total_epochs)
            tf.summary.scalar('gate/specificity', report['specificity'], step=total_epochs)
            tf.summary.scalar('gate/ece', report['ece'], step=total_epochs)
            tf.summary.scalar('gate/roc_auc', report['roc_auc'], step=total_epochs)
            tf.summary.scalar('gate/fairness_passed', float(report['fairness_gate']['passed']), step=total_epochs)
            writer.flush()
        print(f"[gate] epochs={total_epochs} sens={report['sensitivity']:.3f} "
              f"spec={report['specificity']:.3f} ece={report['ece']:.3f} "
              f"roc_auc={report['roc_auc']:.3f} fairness={report['fairness_gate']['passed']}")
        if _targets_met(report, config):
            print(f"[gate] ALL TARGETS MET at {total_epochs} epochs.")
            targets_met = True
            break
        if total_epochs >= config.max_epochs:
            print(f"[gate] reached max_epochs={config.max_epochs} without meeting all targets.")
            break

    # Capped without meeting targets -> restore the best-generalising checkpoint.
    if not targets_met and os.path.exists(best_weights_path):
        model.load_weights(best_weights_path)
        print("[gate] restored best-val_loss weights (avoids keeping an overfit final epoch).")
    return targets_met


def train(config, records_train, records_val):
    tf.keras.utils.set_random_seed(config.seed)

    cache_train = cache_val = None
    if config.cache_dataset:
        cache_directory = os.path.join(config.artifacts_dir, 'cache')
        os.makedirs(cache_directory, exist_ok=True)
        cache_train = os.path.join(cache_directory, 'train')
        cache_val = os.path.join(cache_directory, 'val')

    fine_vocabulary = build_fine_vocabulary(records_train)
    train_dataset, n_fine, train_triage = make_dataset(
        records_train, config, fine_vocabulary, training=True, cache_path=cache_train)
    val_dataset, _, val_triage = make_dataset(
        records_val, config, fine_vocabulary, training=False, cache_path=cache_val)
    val_records = filter_valid(records_val)

    model = build_triage_model(config, n_fine, METADATA_DIM)
    class_weights = triage_class_weights(train_triage, config.malignant_cost, MALIGNANT)
    _compile(model, config, class_weights)

    log_dir = os.path.join(config.artifacts_dir, 'tensorboard')
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)] if config.tensorboard else []

    targets_met = None
    if config.train_until_target:
        targets_met = _gated_fit(model, train_dataset, val_dataset, val_triage, val_records,
                                 config, callbacks, log_dir)
    else:
        stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=max(2, config.epochs // 5), restore_best_weights=True)
        model.fit(train_dataset, validation_data=val_dataset, epochs=config.epochs,
                  callbacks=callbacks + [stopping], verbose=2)

    # Finalise: calibration, operating point, conformal, OOD, save.
    logits = triage_logits_model(model).predict(val_dataset, verbose=0)
    scaler = TemperatureScaler().fit(logits, val_triage)
    calibrated = softmax(logits / scaler.temperature)

    p_malignant = calibrated[:, MALIGNANT]
    y_malignant = (val_triage == MALIGNANT).astype(int)
    operating_threshold = metrics.select_threshold_for_sensitivity(
        y_malignant, p_malignant, config.target_sensitivity)

    conformal = SplitConformalClassifier(alpha=config.conformal_alpha).calibrate(calibrated, val_triage)

    embeddings = feature_model(model).predict(train_dataset.take(MAX_OOD_FIT_BATCHES), verbose=0)
    ood = MahalanobisOODDetector().fit(embeddings)

    bundle = {
        'config': vars(config),
        'metadata_dim': METADATA_DIM,
        'n_fine': n_fine,
        'targets_met': targets_met,
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
