"""Session-separated identification and claimed-identity verification."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def split_sessions(data):
    parts = (data[data.sessionIndex <= 5].copy(),
             data[data.sessionIndex == 6].copy(),
             data[data.sessionIndex >= 7].copy())
    classes = set(parts[0].subject)
    if len(classes) < 2 or any(set(part.subject) != classes for part in parts):
        raise ValueError('Every subject must occur in train, validation, and test sessions')
    return parts


def verification_rates(probabilities, labels, classes, threshold):
    """Test each sample against its true identity and every other known identity."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels, classes = np.asarray(labels), np.asarray(classes)
    if probabilities.shape != (len(labels), len(classes)) or len(labels) == 0 or len(classes) < 2:
        raise ValueError('Invalid probability matrix shape')
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError('Probabilities must be finite and between zero and one')
    if not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError('Probability rows must sum to one')
    if not 0 <= threshold <= 1 or len(set(classes)) != len(classes):
        raise ValueError('Invalid threshold or duplicate classes')
    genuine = labels[:, None] == classes[None, :]
    if not np.all(genuine.sum(axis=1) == 1):
        raise ValueError('Unknown genuine identity')
    accepted = probabilities >= threshold
    false_accepts = int((accepted & ~genuine).sum())
    false_rejects = int((~accepted & genuine).sum())
    genuine_trials = int(genuine.sum())
    impostor_trials = int((~genuine).sum())
    return dict(threshold=float(threshold), false_accepts=false_accepts,
                false_rejects=false_rejects, genuine_trials=genuine_trials,
                impostor_trials=impostor_trials,
                false_accept_rate=false_accepts / impostor_trials,
                false_reject_rate=false_rejects / genuine_trials)


def select_threshold(probabilities, labels, classes):
    """Minimize mean FAR/FRR on validation data only, using a fixed grid."""
    candidates = [verification_rates(probabilities, labels, classes, float(t))
                  for t in np.linspace(0, 1, 201)]
    return min(candidates, key=lambda m: ((m['false_accept_rate'] + m['false_reject_rate']) / 2,
                                          m['false_accept_rate'], -m['threshold']))['threshold']


def identification_metrics(labels, predictions):
    return dict(accuracy=float(accuracy_score(labels, predictions)),
                macro_f1=float(f1_score(labels, predictions, average='macro', zero_division=0)))
