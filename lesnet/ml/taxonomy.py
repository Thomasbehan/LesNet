"""Clinical taxonomy: raw diagnosis labels -> triage group + fine vocabulary (paper §5.1).

The triage group is the load-bearing clinical decision; the fine diagnosis is auxiliary
explanation. Mapping is by keyword because ISIC/PAD-UFES/Fitzpatrick17k/DDI use different
label strings. Unknown/unmappable labels return None so callers can drop them.
"""

TRIAGE_CLASSES = ['benign', 'suspicious', 'malignant']
BENIGN, SUSPICIOUS, MALIGNANT = 0, 1, 2

_MALIGNANT_TERMS = (
    'melanoma', 'basal cell carcinoma', 'bcc', 'squamous cell carcinoma', 'scc',
    'malignant', 'merkel', 'carcinoma', 'angiosarcoma', 'mycosis fungoides',
)
_SUSPICIOUS_TERMS = (
    'actinic keratosis', 'aimp', 'atypical', 'dysplastic', 'bowen',
    'lentigo maligna', 'melanocytic proliferation', 'spitz', 'in situ', 'precursor',
)
_BENIGN_TERMS = (
    'nevus', 'nevi', 'naevus', 'seborrheic keratosis', 'seborrheic', 'dermatofibroma',
    'angioma', 'angiokeratoma', 'acrochordon', 'lentigo', 'benign', 'cafe-au-lait',
    'fibrous papule', 'angiofibroma', 'clear cell acanthoma', 'vascular lesion',
    'hemangioma', 'wart', 'scar', 'cyst', 'molluscum',
)


def triage_index(raw_label):
    """Return the triage class index for a raw label, or None if unmappable."""
    if raw_label is None:
        return None
    text = str(raw_label).strip().lower()
    if not text or text in {'unknown', 'nan', 'none'}:
        return None
    if any(term in text for term in _MALIGNANT_TERMS):
        return MALIGNANT
    if any(term in text for term in _SUSPICIOUS_TERMS):
        return SUSPICIOUS
    if any(term in text for term in _BENIGN_TERMS):
        return BENIGN
    return None


def build_fine_vocabulary(records):
    """Stable {raw_label: index} map over the mappable labels present in the records."""
    labels = sorted({
        str(record.raw_label).strip().lower()
        for record in records
        if triage_index(record.raw_label) is not None
    })
    return {label: index for index, label in enumerate(labels)}


def fine_index(raw_label, fine_vocabulary):
    return fine_vocabulary.get(str(raw_label).strip().lower())
