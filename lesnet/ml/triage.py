"""Selective triage decision (paper §2.3, §5.5).

Maps a calibrated malignancy probability and an image-validity flag to a conservative,
referral-biased action. Thresholds default to placeholders and are meant to be set from
the validation operating point (paper §5.2); they are not clinical settings yet.
"""

REASSURE = 'reassure'
REFER = 'refer'
URGENT = 'urgent'
ABSTAIN = 'abstain'


def triage_decision(p_malignant, valid_image=True, refer_threshold=0.30,
                    urgent_threshold=0.70, abstain_band=(0.45, 0.55)):
    """Return a triage action; abstains on invalid images or the uncertain band."""
    if not valid_image:
        return ABSTAIN
    if p_malignant >= urgent_threshold:
        return URGENT
    if abstain_band[0] <= p_malignant < abstain_band[1]:
        return ABSTAIN
    if p_malignant >= refer_threshold:
        return REFER
    return REASSURE
