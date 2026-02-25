import pandas as pd
import pytest

from keystroke.data import load_dataset
from keystroke.features import feature_names


def row(**changes):
    result = dict(subject='s001', sessionIndex=1, rep=1)
    result.update({name: .1 for name in feature_names()})
    result.update(changes)
    return result


def test_load_preserves_negative_flight_time(tmp_path):
    path = tmp_path / 'samples.csv'
    pd.DataFrame([row(**{'UD.period.t': -.02})]).to_csv(path, index=False)
    assert load_dataset(path).iloc[0]['UD.period.t'] == -.02


@pytest.mark.parametrize('rows', [
    [row(sessionIndex=1.5)], [row(rep=0)], [row(subject=' ')],
    [row(**{'H.period': -1})], [row(**{'H.period': float('inf')})],
    [row(), row()],
])
def test_reject_invalid_data(tmp_path, rows):
    path = tmp_path / 'bad.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_dataset(path)


def test_missing_features_are_rejected(tmp_path):
    path = tmp_path / 'bad.csv'
    pd.DataFrame([{'subject': 's001'}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match='Missing columns'):
        load_dataset(path)
