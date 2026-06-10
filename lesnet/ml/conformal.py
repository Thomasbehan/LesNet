"""Split-conformal prediction sets (paper §5.5).

Least-Ambiguous set-valued Classifier (LAC): the nonconformity score is 1 minus the
predicted probability of the true class. With calibration-set quantile q-hat, the
prediction set {classes with (1 - prob) <= q-hat} covers the true label with
probability >= 1 - alpha under exchangeability.
"""
import numpy as np


class SplitConformalClassifier:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q_hat = None

    def calibrate(self, probabilities, labels):
        probabilities = np.asarray(probabilities, dtype=float)
        labels = np.asarray(labels, dtype=int)
        scores = 1.0 - probabilities[np.arange(len(labels)), labels]
        count = len(scores)
        level = min(np.ceil((count + 1) * (1 - self.alpha)) / count, 1.0)
        self.q_hat = float(np.quantile(scores, level, method='higher'))
        return self

    def predict_set(self, probabilities):
        if self.q_hat is None:
            raise RuntimeError("Call calibrate() before predict_set().")
        probabilities = np.asarray(probabilities, dtype=float)
        return [np.where(1.0 - row <= self.q_hat)[0] for row in probabilities]

    def empirical_coverage(self, probabilities, labels):
        labels = np.asarray(labels, dtype=int)
        prediction_sets = self.predict_set(probabilities)
        covered = [labels[index] in prediction_sets[index] for index in range(len(labels))]
        return float(np.mean(covered))
