import numpy as np
import pandas as pd
import pytest

from keystroke.evaluation import select_threshold, split_sessions, verification_rates


def test_session_split_keeps_later_sessions_out_of_training():
    data = pd.DataFrame([dict(subject=s, sessionIndex=i) for s in ['a', 'b'] for i in range(1, 9)])
    train, validation, test = split_sessions(data)
    assert set(train.sessionIndex) == {1, 2, 3, 4, 5}
    assert set(validation.sessionIndex) == {6}
    assert set(test.sessionIndex) == {7, 8}
    assert not set(train.index) & set(test.index)


def test_missing_validation_subject_is_rejected():
    with pytest.raises(ValueError):
        split_sessions(pd.DataFrame([dict(subject='a', sessionIndex=1)]))


def test_verification_counts_known_genuine_and_impostor_trials():
    metrics = verification_rates([[.9, .1], [.6, .4]], ['a', 'b'], ['a', 'b'], .5)
    assert metrics['false_accepts'] == metrics['false_rejects'] == 1
    assert metrics['genuine_trials'] == metrics['impostor_trials'] == 2
    assert metrics['false_accept_rate'] == metrics['false_reject_rate'] == .5


def test_threshold_uses_validation_scores():
    scores = np.array([[.9, .1], [.2, .8]])
    threshold = select_threshold(scores, ['a', 'b'], ['a', 'b'])
    metrics = verification_rates(scores, ['a', 'b'], ['a', 'b'], threshold)
    assert metrics['false_accepts'] == metrics['false_rejects'] == 0


def test_unknown_identity_is_rejected():
    with pytest.raises(ValueError, match='Unknown'):
        verification_rates([[.9, .1]], ['c'], ['a', 'b'], .5)
