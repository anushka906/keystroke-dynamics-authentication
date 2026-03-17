import numpy as np
import pandas as pd
import pytest

from keystroke.features import feature_names
from keystroke.models import make_model
from keystroke.predict import predict, read_samples


def test_claim_uses_claimed_identity_probability():
    columns = feature_names()
    x = pd.DataFrame([[.1] * 31] * 5 + [[.8] * 31] * 5, columns=columns)
    model = make_model('random_forest').fit(x, ['a'] * 5 + ['b'] * 5)
    bundle = dict(model=model, features=columns, threshold=.5)
    assert predict(bundle, x.iloc[:1], 'a')[0]['accepted']
    assert not predict(bundle, x.iloc[:1], 'b')[0]['accepted']
    assert predict(bundle, x.iloc[:1])[0]['predicted_subject'] == 'a'
    with pytest.raises(ValueError, match='not enrolled'):
        predict(bundle, x.iloc[:1], 'unknown')


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1])
def test_prediction_input_rejects_invalid_hold_times(tmp_path, value):
    row = {c: .1 for c in feature_names()}
    row['H.period'] = value
    path = tmp_path / 'bad.csv'
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_samples(path)
