"""Patient-metadata feature encoding for image+metadata fusion (paper §5.1)."""
import numpy as np

SITE_VOCABULARY = [
    'head/neck', 'upper extremity', 'lower extremity', 'torso',
    'palms/soles', 'oral/genital', 'unknown',
]
METADATA_DIM = 3 + len(SITE_VOCABULARY) + 1  # age, sex(2), site one-hot, fitzpatrick


def normalize_site(site):
    if not site:
        return 'unknown'
    text = str(site).strip().lower()
    if any(key in text for key in ('head', 'neck', 'face', 'scalp', 'ear')):
        return 'head/neck'
    if any(key in text for key in ('upper', 'arm', 'hand', 'forearm', 'shoulder')):
        return 'upper extremity'
    if any(key in text for key in ('lower', 'leg', 'thigh', 'foot', 'knee')):
        return 'lower extremity'
    if any(key in text for key in ('torso', 'trunk', 'back', 'chest', 'abdomen')):
        return 'torso'
    if any(key in text for key in ('palm', 'sole', 'acral', 'nail')):
        return 'palms/soles'
    if any(key in text for key in ('oral', 'genital', 'mucos')):
        return 'oral/genital'
    return 'unknown'


def metadata_vector(record):
    """Fixed-length metadata vector for one LesionRecord."""
    age = (record.age or 0.0) / 100.0
    sex = str(record.sex or '').lower()
    sex_male = 1.0 if sex.startswith('m') else 0.0
    sex_female = 1.0 if sex.startswith('f') else 0.0
    site = normalize_site(record.anatomical_site)
    site_one_hot = [1.0 if site == known else 0.0 for known in SITE_VOCABULARY]
    fitzpatrick = (record.fitzpatrick or 0) / 6.0
    return np.array([age, sex_male, sex_female, *site_one_hot, fitzpatrick], dtype=np.float32)
