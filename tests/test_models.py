import numpy as np
import pytest

from keystroke.models import MODEL_NAMES, make_model


@pytest.mark.parametrize('name', MODEL_NAMES)
def test_classifiers_produce_aligned_probabilities(name):
    rng = np.random.default_rng(42)
    x = np.concatenate([rng.normal(-2, .1, (12, 3)), rng.normal(2, .1, (12, 3))])
    y = np.array(['alice'] * 12 + ['bob'] * 12)
    model = make_model(name)
    model.fit(x, y)
    probabilities = model.predict_proba(x[:2])
    assert list(model.classes_) == ['alice', 'bob']
    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert (model.predict(x) == y).mean() > .9


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        make_model('other')
