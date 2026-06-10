import numpy as np
import pytest

from lesnet.ml import calibration, conformal, metrics, preprocessing, splits, triage


def test_sensitivity_first_threshold_meets_target():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.ones(100), np.zeros(100)]).astype(int)
    # Malignant cases score higher on average, with overlap.
    p = np.concatenate([rng.uniform(0.4, 1.0, 100), rng.uniform(0.0, 0.6, 100)])
    threshold = metrics.select_threshold_for_sensitivity(y, p, target_sensitivity=0.95)
    tp, fp, tn, fn = metrics.confusion_at_threshold(y, p, threshold)
    assert metrics.sensitivity(tp, fn) >= 0.95


def test_ppv_drops_at_low_prevalence():
    # Same classifier, lower prevalence -> lower PPV (the prevalence lesson).
    high = metrics.ppv_at_prevalence(0.95, 0.90, prevalence=0.50)
    low = metrics.ppv_at_prevalence(0.95, 0.90, prevalence=0.02)
    assert high > low


def test_clinical_report_shape():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.05])
    report = metrics.clinical_report(y, p, target_sensitivity=0.9)
    assert {'threshold', 'sensitivity', 'specificity', 'roc_auc', 'ppv_at_prevalence'} <= set(report)
    assert 0.0 <= report['sensitivity'] <= 1.0


def test_risk_coverage_monotone_endpoints():
    correct = np.array([1, 1, 0, 1, 0, 1])
    confidence = np.array([0.99, 0.95, 0.9, 0.6, 0.55, 0.5])
    coverage, risk = metrics.risk_coverage_curve(correct, confidence)
    assert coverage[-1] == pytest.approx(1.0)
    assert 0.0 <= metrics.area_under_risk_coverage(correct, confidence) <= 1.0


def test_grouped_split_has_no_leakage():
    groups = np.repeat(np.arange(50), 4)  # 50 patients, 4 images each
    train, val, test = splits.grouped_train_val_test(groups, test_size=0.2, val_size=0.2, seed=1)
    splits.assert_no_group_leakage(groups, train, val, test)
    assert len(train) + len(val) + len(test) == len(groups)


def test_temperature_scaling_reduces_nll_for_overconfident_logits():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 3, size=300)
    base = rng.normal(size=(300, 3))
    base[np.arange(300), labels] += 1.0
    logits = base * 6.0  # exaggerate -> overconfident

    def nll(probabilities):
        true = np.clip(probabilities[np.arange(300), labels], 1e-12, 1.0)
        return -np.mean(np.log(true))

    scaler = calibration.TemperatureScaler().fit(logits, labels)
    before = nll(calibration.softmax(logits))
    after = nll(scaler.transform(logits))
    assert scaler.temperature > 1.0  # softening an overconfident model
    assert after <= before + 1e-9


def test_conformal_achieves_nominal_coverage():
    rng = np.random.default_rng(2)
    n, classes, alpha = 2000, 4, 0.1
    labels = rng.integers(0, classes, size=n)
    logits = rng.normal(size=(n, classes))
    logits[np.arange(n), labels] += 1.5
    probabilities = calibration.softmax(logits)
    half = n // 2
    model = conformal.SplitConformalClassifier(alpha=alpha)
    model.calibrate(probabilities[:half], labels[:half])
    coverage = model.empirical_coverage(probabilities[half:], labels[half:])
    assert coverage >= 1 - alpha - 0.05  # within sampling tolerance of the guarantee


def test_shades_of_gray_neutralises_uniform_tint():
    tinted = np.ones((8, 8, 3))
    tinted[..., 0] *= 0.8  # reddish cast
    tinted[..., 2] *= 0.4
    corrected = preprocessing.shades_of_gray(tinted)
    channel_means = corrected.reshape(-1, 3).mean(axis=0)
    assert np.std(channel_means) < 1e-6  # channels equalised


def test_triage_decision_branches():
    assert triage.triage_decision(0.95) == triage.URGENT
    assert triage.triage_decision(0.50) == triage.ABSTAIN
    assert triage.triage_decision(0.35) == triage.REFER
    assert triage.triage_decision(0.05) == triage.REASSURE
    assert triage.triage_decision(0.95, valid_image=False) == triage.ABSTAIN
