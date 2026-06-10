"""Clinical evaluation metrics for malignant-vs-benign triage (paper §6).

Sensitivity-first operating-point selection, prevalence-aware PPV/NPV, calibration
error, and risk-coverage / selective-risk curves. All inputs are plain numpy arrays;
`y_malignant` is 1 for malignant, 0 for benign; `p_malignant` is the (ideally
calibrated) probability of malignancy.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# numpy >= 2.0 renamed trapz -> trapezoid
_trapezoid = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


def confusion_at_threshold(y_malignant, p_malignant, threshold):
    y_malignant = np.asarray(y_malignant).astype(int)
    predicted_positive = np.asarray(p_malignant) >= threshold
    true_positive = int(np.sum(predicted_positive & (y_malignant == 1)))
    false_positive = int(np.sum(predicted_positive & (y_malignant == 0)))
    false_negative = int(np.sum(~predicted_positive & (y_malignant == 1)))
    true_negative = int(np.sum(~predicted_positive & (y_malignant == 0)))
    return true_positive, false_positive, true_negative, false_negative


def sensitivity(true_positive, false_negative):
    denominator = true_positive + false_negative
    return true_positive / denominator if denominator > 0 else 0.0


def specificity(true_negative, false_positive):
    denominator = true_negative + false_positive
    return true_negative / denominator if denominator > 0 else 0.0


def ppv_at_prevalence(sensitivity_value, specificity_value, prevalence):
    numerator = sensitivity_value * prevalence
    denominator = numerator + (1 - specificity_value) * (1 - prevalence)
    return numerator / denominator if denominator > 0 else 0.0


def npv_at_prevalence(sensitivity_value, specificity_value, prevalence):
    numerator = specificity_value * (1 - prevalence)
    denominator = numerator + (1 - sensitivity_value) * prevalence
    return numerator / denominator if denominator > 0 else 0.0


def select_threshold_for_sensitivity(y_malignant, p_malignant, target_sensitivity):
    """Largest threshold whose sensitivity still meets the target.

    Sensitivity is monotonically non-increasing in the threshold, so the largest
    qualifying threshold maximises specificity — the sensitivity-first operating point.
    """
    candidate_thresholds = np.unique(p_malignant)
    chosen = float(np.min(candidate_thresholds))
    for threshold in candidate_thresholds:
        true_positive, _, _, false_negative = confusion_at_threshold(y_malignant, p_malignant, threshold)
        if sensitivity(true_positive, false_negative) >= target_sensitivity:
            chosen = float(threshold)
    return chosen


def safe_roc_auc(y_malignant, p_malignant):
    y_malignant = np.asarray(y_malignant)
    if len(np.unique(y_malignant)) < 2:
        return float('nan')
    return float(roc_auc_score(y_malignant, p_malignant))


def safe_pr_auc(y_malignant, p_malignant):
    y_malignant = np.asarray(y_malignant)
    if len(np.unique(y_malignant)) < 2:
        return float('nan')
    return float(average_precision_score(y_malignant, p_malignant))


def expected_calibration_error(correct, confidence, n_bins=10):
    """ECE: average gap between accuracy and confidence across equal-width bins."""
    correct = np.asarray(correct).astype(float)
    confidence = np.asarray(confidence).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confidence)
    error = 0.0
    for index in range(n_bins):
        low, high = edges[index], edges[index + 1]
        in_bin = (confidence > low) & (confidence <= high) if index > 0 else (confidence >= low) & (confidence <= high)
        if np.any(in_bin):
            accuracy = np.mean(correct[in_bin])
            mean_confidence = np.mean(confidence[in_bin])
            error += (np.sum(in_bin) / total) * abs(accuracy - mean_confidence)
    return float(error)


def risk_coverage_curve(correct, confidence):
    """Selective risk as a function of coverage, ranking by descending confidence."""
    correct = np.asarray(correct).astype(float)
    order = np.argsort(-np.asarray(confidence))
    ordered_correct = correct[order]
    count = len(ordered_correct)
    included = np.arange(1, count + 1)
    coverage = included / count
    risk = 1.0 - np.cumsum(ordered_correct) / included
    return coverage, risk


def area_under_risk_coverage(correct, confidence):
    coverage, risk = risk_coverage_curve(correct, confidence)
    return float(_trapezoid(risk, coverage))


def clinical_report(y_malignant, p_malignant, target_sensitivity=0.97, prevalences=(0.02, 0.05, 0.10)):
    """Sensitivity-first operating point plus prevalence-aware PPV/NPV and AUCs."""
    threshold = select_threshold_for_sensitivity(y_malignant, p_malignant, target_sensitivity)
    true_positive, false_positive, true_negative, false_negative = confusion_at_threshold(
        y_malignant, p_malignant, threshold)
    sensitivity_value = sensitivity(true_positive, false_negative)
    specificity_value = specificity(true_negative, false_positive)
    return {
        'threshold': threshold,
        'target_sensitivity': target_sensitivity,
        'sensitivity': sensitivity_value,
        'specificity': specificity_value,
        'roc_auc': safe_roc_auc(y_malignant, p_malignant),
        'pr_auc': safe_pr_auc(y_malignant, p_malignant),
        'confusion': {
            'true_positive': true_positive, 'false_positive': false_positive,
            'true_negative': true_negative, 'false_negative': false_negative,
        },
        'ppv_at_prevalence': {
            prevalence: ppv_at_prevalence(sensitivity_value, specificity_value, prevalence)
            for prevalence in prevalences
        },
        'npv_at_prevalence': {
            prevalence: npv_at_prevalence(sensitivity_value, specificity_value, prevalence)
            for prevalence in prevalences
        },
    }
