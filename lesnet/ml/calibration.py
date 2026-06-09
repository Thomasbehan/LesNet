"""Temperature scaling for probability calibration (paper §5.5).

A single scalar temperature is fit by minimising negative log-likelihood on a held-out
set; it sharpens or softens softmax outputs without changing the arg-max prediction.
"""
import numpy as np
from scipy.optimize import minimize_scalar


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=1, keepdims=True)


class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, labels):
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=int)
        row_index = np.arange(len(labels))

        def negative_log_likelihood(temperature):
            probabilities = softmax(logits / max(temperature, 1e-3))
            true_class = np.clip(probabilities[row_index, labels], 1e-12, 1.0)
            return -np.mean(np.log(true_class))

        result = minimize_scalar(negative_log_likelihood, bounds=(1e-2, 100.0), method='bounded')
        self.temperature = float(result.x)
        return self

    def transform(self, logits):
        return softmax(np.asarray(logits, dtype=float) / self.temperature)
