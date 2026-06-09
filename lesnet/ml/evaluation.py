"""Evaluation report + fairness gate + model card (paper §6, §7).

`build_report` is pure numpy (testable without a model): it consumes ground-truth triage
labels, calibrated probabilities, and subgroup arrays, and returns clinical metrics,
calibration, risk-coverage, conformal coverage, and a release-blocking fairness gate.
"""
import numpy as np

from lesnet.ml import metrics
from lesnet.ml.taxonomy import MALIGNANT

MIN_SUBGROUP_MALIGNANT_SUPPORT = 5


def build_report(y_triage, calibrated_probabilities, subgroups=None, target_sensitivity=0.97,
                 fairness_margin=0.1, q_hat=None):
    y_triage = np.asarray(y_triage).astype(int)
    probabilities = np.asarray(calibrated_probabilities, dtype=float)
    p_malignant = probabilities[:, MALIGNANT]
    y_malignant = (y_triage == MALIGNANT).astype(int)

    report = metrics.clinical_report(y_malignant, p_malignant, target_sensitivity)
    threshold = report['threshold']
    overall_sensitivity = report['sensitivity']

    predicted = probabilities.argmax(axis=1)
    correct = (predicted == y_triage).astype(float)
    confidence = probabilities.max(axis=1)
    report['ece'] = metrics.expected_calibration_error(correct, confidence)
    report['aurc'] = metrics.area_under_risk_coverage(correct, confidence)

    if q_hat is not None:
        in_set = (1.0 - probabilities) <= q_hat
        report['conformal_coverage'] = float(np.mean(in_set[np.arange(len(y_triage)), y_triage]))

    subgroup_report = {}
    fairness_failures = []
    for name, values in (subgroups or {}).items():
        values = np.asarray(values)
        levels = {}
        for level in sorted({str(value) for value in values}):
            mask = values.astype(str) == level
            true_positive, _, _, false_negative = metrics.confusion_at_threshold(
                y_malignant[mask], p_malignant[mask], threshold)
            subgroup_sensitivity = metrics.sensitivity(true_positive, false_negative)
            malignant_support = int(np.sum(y_malignant[mask] == 1))
            levels[level] = {
                'sensitivity': subgroup_sensitivity,
                'malignant_support': malignant_support,
                'n': int(np.sum(mask)),
            }
            if (malignant_support >= MIN_SUBGROUP_MALIGNANT_SUPPORT
                    and subgroup_sensitivity < overall_sensitivity - fairness_margin):
                fairness_failures.append(
                    f"{name}={level}: sensitivity {subgroup_sensitivity:.3f} "
                    f"< overall {overall_sensitivity:.3f} - {fairness_margin}")
        subgroup_report[name] = levels

    report['subgroups'] = subgroup_report
    report['fairness_gate'] = {
        'passed': len(fairness_failures) == 0,
        'failures': fairness_failures,
        'margin': fairness_margin,
    }
    return report


def write_model_card(report, bundle, path):
    confusion = report['confusion']
    lines = [
        "# LesNet Triage — Model Card",
        "",
        "**Intended use:** education + triage with a referral bias. **Not a diagnosis.**",
        "",
        "## Operating point (sensitivity-first)",
        f"- Target sensitivity: {report['target_sensitivity']}",
        f"- Threshold: {report['threshold']:.4f}",
        f"- Sensitivity: {report['sensitivity']:.4f} · Specificity: {report['specificity']:.4f}",
        f"- ROC-AUC: {report['roc_auc']:.4f} · PR-AUC: {report['pr_auc']:.4f}",
        f"- Confusion (malignant-vs-rest): {confusion}",
        "",
        "## Calibration & selective prediction",
        f"- ECE: {report['ece']:.4f} · AURC: {report['aurc']:.4f}",
        f"- Conformal coverage: {report.get('conformal_coverage', 'n/a')}",
        "",
        "## PPV / NPV at prevalence",
    ]
    for prevalence, value in report['ppv_at_prevalence'].items():
        lines.append(f"- prevalence {prevalence}: PPV {value:.3f}, NPV {report['npv_at_prevalence'][prevalence]:.3f}")
    lines += ["", "## Fairness gate",
              f"- Passed: {report['fairness_gate']['passed']}"]
    for failure in report['fairness_gate']['failures']:
        lines.append(f"  - FAIL: {failure}")
    lines += ["", "## Subgroups"]
    for name, levels in report['subgroups'].items():
        lines.append(f"- **{name}**")
        for level, stats in levels.items():
            lines.append(f"  - {level}: sensitivity {stats['sensitivity']:.3f} "
                         f"(malignant n={stats['malignant_support']}, total n={stats['n']})")
    lines += ["", "## Limitations",
              "- Bounded by dataset coverage (skin tone, devices); see docs/model-redesign.md."]
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write("\n".join(lines) + "\n")
    return path
