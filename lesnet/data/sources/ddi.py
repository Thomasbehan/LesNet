"""DDI (Diverse Dermatology Images) source — best dark-skin coverage, biopsy-confirmed.

Stanford requires a Research Use Agreement, so there is no automated download; place the
release at ``root`` (``ddi_metadata.csv`` + image files) and this parses it.
"""
import os

from lesnet.data.records import LesionRecord, read_csv_rows

# DDI encodes skin tone as 12/34/56 (Fitzpatrick pairs); map to a representative band.
_SKIN_TONE_TO_FITZPATRICK = {'12': 1, '34': 3, '56': 5}


def skin_tone_to_fitzpatrick(skin_tone):
    return _SKIN_TONE_TO_FITZPATRICK.get(str(skin_tone).strip(), None)


def parse(root, limit=None):
    rows = read_csv_rows(os.path.join(root, 'ddi_metadata.csv'))
    rows = rows[:limit] if limit else rows
    records = []
    for row in rows:
        image_file = row.get('DDI_file')
        if not image_file:
            continue
        records.append(LesionRecord(
            image_path=os.path.join(root, image_file),
            source_dataset='ddi',
            raw_label=row.get('disease') or 'unknown',
            group_id=image_file,
            fitzpatrick=skin_tone_to_fitzpatrick(row.get('skin_tone')),
        ))
    return records
