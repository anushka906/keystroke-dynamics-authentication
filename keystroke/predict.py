"""Identify samples or verify a claimed known identity using a saved model."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import KeyEvent, extract_features, feature_names


def read_samples(path, events=False):
    if events:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        samples = pd.DataFrame([extract_features([KeyEvent(**e) for e in payload['events']])])
    else:
        samples = pd.read_csv(path)
    columns = feature_names()
    missing = set(columns) - set(samples.columns)
    if missing:
        raise ValueError(f'Missing timing features: {sorted(missing)}')
    samples = samples[columns].astype(float)
    if samples.empty or not np.isfinite(samples.to_numpy()).all():
        raise ValueError('Samples must contain finite timing values')
    nonnegative = [c for c in columns if c.startswith(('H.', 'DD.'))]
    if (samples[nonnegative] < 0).any().any():
        raise ValueError('Hold and down-down times cannot be negative')
    return samples


def predict(bundle, samples, claimed_subject=None):
    columns = bundle['features']
    if columns != feature_names():
        raise ValueError('Model feature schema does not match this application')
    model = bundle['model']
    classes = list(model.classes_)
    if claimed_subject is not None and claimed_subject not in classes:
        raise ValueError('Claimed subject was not enrolled in this model')
    scores = model.predict_proba(samples[columns])
    result = []
    for row in scores:
        best = int(row.argmax())
        item = dict(predicted_subject=str(classes[best]), confidence=float(row[best]))
        if claimed_subject is not None:
            score = float(row[classes.index(claimed_subject)])
            item.update(claimed_subject=claimed_subject, claim_probability=score,
                        threshold=bundle['threshold'], accepted=score >= bundle['threshold'])
        result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='A trusted local model produced by keystroke.train')
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--events', action='store_true', help='Input is a captured event JSON file instead of feature CSV')
    parser.add_argument('--claim', help='Known subject ID to verify')
    args = parser.parse_args()
    try:
        bundle = joblib.load(args.model)
        samples = read_samples(args.samples, args.events)
        print(json.dumps(predict(bundle, samples, args.claim), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
