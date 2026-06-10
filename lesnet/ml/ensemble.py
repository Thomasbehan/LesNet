"""Deep-ensemble averaging over independently trained triage predictors (paper §5.4)."""
import numpy as np

from lesnet.ml.taxonomy import MALIGNANT, TRIAGE_CLASSES
from lesnet.ml.triage import ABSTAIN, triage_decision


def ensemble_predict(predictors, image_array, record):
    """Average member probabilities; abstain if every member abstains."""
    member_results = [predictor.predict(image_array, record) for predictor in predictors]
    answered = [result for result in member_results if result.get('valid_image')]
    if not answered:
        return {'triage': ABSTAIN, 'valid_image': False, 'reason': 'all_members_abstained'}

    mean_probabilities = np.mean(
        [[result['probabilities'][name] for name in TRIAGE_CLASSES] for result in answered], axis=0)
    thresholds = predictors[0].thresholds
    p_malignant = float(mean_probabilities[MALIGNANT])
    decision = triage_decision(
        p_malignant, valid_image=True,
        refer_threshold=thresholds['refer'], urgent_threshold=thresholds['urgent'],
        abstain_band=tuple(thresholds['abstain_band']))
    return {
        'triage': decision,
        'valid_image': True,
        'p_malignant': p_malignant,
        'probabilities': {name: float(mean_probabilities[index]) for index, name in enumerate(TRIAGE_CLASSES)},
        'members': len(answered),
    }
