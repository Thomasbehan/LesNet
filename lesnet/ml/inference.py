"""Selective triage inference (paper §2.3, §5.5, §7).

Loads the artifact bundle and runs: quality gate -> OOD gate -> calibrated triage ->
abstention/conformal set. Output is referral-biased and never a definitive diagnosis.
"""
import numpy as np
import tensorflow as tf

from lesnet.ml import artifacts
from lesnet.ml.calibration import softmax
from lesnet.ml.features import metadata_vector
from lesnet.ml.model import feature_model, triage_logits_model
from lesnet.ml.ood import MahalanobisOODDetector, is_low_quality
from lesnet.ml.preprocessing import PreprocessingPipeline
from lesnet.ml.taxonomy import MALIGNANT, TRIAGE_CLASSES
from lesnet.ml.triage import ABSTAIN, triage_decision


class TriagePredictor:
    def __init__(self, directory, use_test_time_augmentation=False):
        self.bundle = artifacts.load_bundle(directory)
        self.model = tf.keras.models.load_model(artifacts.model_path(directory), compile=False)
        self.logits_model = triage_logits_model(self.model)
        self.feature_model = feature_model(self.model)
        self.temperature = self.bundle['calibration']['temperature']
        self.q_hat = self.bundle['conformal']['q_hat']
        self.thresholds = self.bundle['thresholds']
        self.ood = MahalanobisOODDetector.from_dict(self.bundle['ood']['mahalanobis'])
        self.min_variance = self.bundle['ood']['min_laplacian_variance']
        self.image_size = tuple(self.bundle['config']['image_size'])
        # Inference preprocessing must match training exactly (same remove_hair setting).
        self.pipeline = PreprocessingPipeline(
            image_size=self.image_size, remove_hair=not self.bundle['config'].get('smoke', False))
        self.use_test_time_augmentation = use_test_time_augmentation

    def _logits(self, image):
        batch = image[None, ...].astype('float32')
        variants = [batch]
        if self.use_test_time_augmentation:
            variants += [batch[:, :, ::-1, :], batch[:, ::-1, :, :]]
        return np.mean([self.logits_model.predict(
            {'image': variant, 'metadata': self._metadata}, verbose=0) for variant in variants], axis=0)

    def predict(self, image_array, record):
        if is_low_quality(image_array, self.min_variance):
            return {'triage': ABSTAIN, 'valid_image': False, 'reason': 'low_quality'}

        image = self.pipeline(np.asarray(image_array))
        self._metadata = metadata_vector(record)[None, ...]
        inputs = {'image': image[None, ...].astype('float32'), 'metadata': self._metadata}

        embedding = self.feature_model.predict(inputs, verbose=0)[0]
        if self.ood.is_out_of_distribution(embedding):
            return {'triage': ABSTAIN, 'valid_image': False, 'reason': 'out_of_distribution'}

        probabilities = softmax(self._logits(image) / self.temperature)[0]
        p_malignant = float(probabilities[MALIGNANT])
        decision = triage_decision(
            p_malignant, valid_image=True,
            refer_threshold=self.thresholds['refer'],
            urgent_threshold=self.thresholds['urgent'],
            abstain_band=tuple(self.thresholds['abstain_band']))
        conformal_set = [TRIAGE_CLASSES[index] for index in range(3)
                         if (1.0 - probabilities[index]) <= self.q_hat]
        return {
            'triage': decision,
            'valid_image': True,
            'p_malignant': p_malignant,
            'probabilities': {TRIAGE_CLASSES[index]: float(probabilities[index]) for index in range(3)},
            'conformal_set': conformal_set,
        }
