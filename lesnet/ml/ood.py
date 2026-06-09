"""Out-of-distribution / image-quality gate (paper §5.5).

Two cheap checks run before any diagnosis: a Mahalanobis distance on shared embeddings
(rejects images far from the training feature distribution) and a Laplacian-variance blur
check (rejects out-of-focus images). Either firing -> abstain ("see a clinician").
"""
import numpy as np


class MahalanobisOODDetector:
    def __init__(self):
        self.mean = None
        self.inverse_covariance = None
        self.threshold = None

    def fit(self, embeddings, quantile=0.99, ridge=1e-3):
        embeddings = np.asarray(embeddings, dtype=float)
        n_samples, n_features = embeddings.shape
        self.mean = embeddings.mean(axis=0)
        if n_samples <= n_features:
            # Too few samples for a full covariance — fall back to a diagonal estimate.
            variances = embeddings.var(axis=0) + ridge
            self.inverse_covariance = np.diag(1.0 / variances)
        else:
            covariance = np.cov(embeddings.T) + ridge * np.eye(n_features)
            self.inverse_covariance = np.linalg.pinv(covariance)
        self.threshold = float(np.quantile(self._distances(embeddings), quantile))
        return self

    def _distances(self, embeddings):
        difference = np.atleast_2d(embeddings) - self.mean
        return np.einsum('ij,jk,ik->i', difference, self.inverse_covariance, difference)

    def score(self, embedding):
        return float(self._distances(embedding)[0])

    def is_out_of_distribution(self, embedding):
        return self.score(embedding) > self.threshold

    def to_dict(self):
        return {
            'mean': self.mean.tolist(),
            'inverse_covariance': self.inverse_covariance.tolist(),
            'threshold': self.threshold,
        }

    @classmethod
    def from_dict(cls, payload):
        detector = cls()
        detector.mean = np.asarray(payload['mean'], dtype=float)
        detector.inverse_covariance = np.asarray(payload['inverse_covariance'], dtype=float)
        detector.threshold = float(payload['threshold'])
        return detector


def laplacian_variance(image):
    """Variance of the Laplacian of the grayscale image; low values indicate blur."""
    array = np.asarray(image, dtype=float)
    if array.ndim == 3:
        array = array.mean(axis=2)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
    padded = np.pad(array, 1, mode='reflect')
    response = (
        kernel[1, 1] * array
        + padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
    )
    return float(np.var(response))


def is_low_quality(image, min_variance=1e-4):
    return laplacian_variance(image) < min_variance
