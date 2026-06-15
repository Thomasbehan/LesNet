"""Curated diagnosis-name canonicalisation (stage 1, post-download reconciliation).

Different sources spell the same diagnosis differently ("melanoma" / "Melanoma, NOS" /
"malignant melanoma"; "nevus" / "naevus" / "melanocytic nevus"). We collapse variants to a
single canonical folder name via a CURATED, ordered map — not fuzzy auto-merge — so the
mapping is reviewable and stable. Unmapped labels fall back to a slug of the raw name and
are reported (``unmapped_diagnoses``) so the map can be extended by hand.

Order matters: more-specific / malignant canonicals are listed first so e.g.
"lentigo maligna melanoma" maps to ``melanoma`` (not ``lentigo``).
"""
import re

# (canonical_name, (substring terms that identify it,)) — checked in order, first match wins.
CANONICAL_DIAGNOSES = (
    # --- malignant ---
    ('melanoma', ('melanoma', 'mel ', 'malignant melanocytic')),
    ('basal cell carcinoma', ('basal cell carcinoma', 'basal cell ca', 'bcc')),
    ('squamous cell carcinoma', ('squamous cell carcinoma', 'squamous cell ca', 'scc')),
    ('merkel cell carcinoma', ('merkel',)),
    ('angiosarcoma', ('angiosarcoma',)),
    ('mycosis fungoides', ('mycosis fungoides',)),
    # --- suspicious / indeterminate ---
    ('actinic keratosis', ('actinic keratosis', 'actinic keratoses', 'solar keratosis')),
    ('bowen disease', ('bowen', 'in situ')),
    ('dysplastic nevus', ('dysplastic', 'atypical')),
    ('lentigo maligna', ('lentigo maligna',)),
    ('spitz nevus', ('spitz',)),
    # --- benign ---
    ('seborrheic keratosis', ('seborrheic keratosis', 'seborrhoeic keratosis', 'seborrheic')),
    ('dermatofibroma', ('dermatofibroma',)),
    ('vascular lesion', ('vascular', 'hemangioma', 'haemangioma', 'angioma', 'angiokeratoma',
                         'pyogenic granuloma')),
    ('acrochordon', ('acrochordon', 'skin tag', 'fibroepithelial polyp')),
    ('solar lentigo', ('solar lentigo', 'lentigo simplex', 'lentigo')),
    ('cafe-au-lait macule', ('cafe-au-lait', 'cafe au lait')),
    ('wart', ('wart', 'verruca')),
    ('molluscum', ('molluscum',)),
    ('nevus', ('nevus', 'naevus', 'nevi')),   # generic nevus last (most permissive)
)


def slugify(raw_label):
    """Filesystem-safe slug for an unmapped diagnosis (lowercase, underscores)."""
    slug = re.sub(r'[^a-z0-9]+', '_', str(raw_label).strip().lower()).strip('_')
    return slug or 'unknown'


def canonical_diagnosis(raw_label):
    """Map a raw diagnosis string to its curated canonical name, or None if unmapped."""
    if raw_label is None:
        return None
    text = str(raw_label).strip().lower()
    if not text or text in {'unknown', 'nan', 'none'}:
        return None
    for canonical, terms in CANONICAL_DIAGNOSES:
        if any(term in text for term in terms):
            return canonical
    return None


def canonical_or_slug(raw_label):
    """Canonical name if known, else a slug of the raw label (so every record gets a folder)."""
    return canonical_diagnosis(raw_label) or slugify(raw_label)


def unmapped_diagnoses(raw_labels):
    """Sorted unique raw labels that have no curated canonical mapping (for human review)."""
    seen = set()
    for raw_label in raw_labels:
        if raw_label is None:
            continue
        text = str(raw_label).strip()
        if text and text.lower() not in {'unknown', 'nan', 'none'} and canonical_diagnosis(text) is None:
            seen.add(text)
    return sorted(seen)
