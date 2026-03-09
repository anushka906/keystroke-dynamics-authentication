import json

import joblib
import numpy as np
import pandas as pd

from keystroke.features import feature_names
from keystroke.train import run


def test_training_saves_model_and_disjoint_session_report(tmp_path):
    rng = np.random.default_rng(7)
    rows = []
    for subject, center in [('a', .1), ('b', .5)]:
        for session in range(1, 9):
            for rep in range(1, 4):
                values = dict(zip(feature_names(), rng.normal(center, .005, 31)))
                rows.append(dict(subject=subject, sessionIndex=session, rep=rep, **values))
    source = tmp_path / 'data.csv'
    pd.DataFrame(rows).to_csv(source, index=False)
    report_path = tmp_path / 'results.json'
    run(source, tmp_path / 'models', report_path, ['random_forest'])
    report = json.loads(report_path.read_text())
    assert report['split']['test']['sessions'] == [7, 8]
    assert report['models']['random_forest']['identification']['accuracy'] == 1
    bundle = joblib.load(tmp_path / 'models/random_forest.joblib')
    assert 'sessionIndex' not in bundle['features']
    assert 'subject' not in bundle['features']
    assert len(bundle['features']) == 31
