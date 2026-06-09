"""Save/load the trained triage artifact bundle (model + calibration + gates + maps)."""
import json
import os

MODEL_FILE = 'triage_model.keras'
BUNDLE_FILE = 'artifacts.json'


def save_bundle(directory, bundle):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, BUNDLE_FILE), 'w', encoding='utf-8') as handle:
        json.dump(bundle, handle, indent=2)


def load_bundle(directory):
    with open(os.path.join(directory, BUNDLE_FILE), encoding='utf-8') as handle:
        return json.load(handle)


def model_path(directory):
    return os.path.join(directory, MODEL_FILE)
