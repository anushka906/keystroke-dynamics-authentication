"""Train classifiers and write reproducible, session-held-out evaluation reports."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn

from .data import DEFAULT_PATH, load_dataset
from .evaluation import identification_metrics, select_threshold, split_sessions, verification_rates
from .features import feature_names
from .models import MODEL_NAMES, make_model


def run(data_path, output_dir, report_path, names=MODEL_NAMES, seed=42):
    data_path, output_dir, report_path = map(Path, (data_path, output_dir, report_path))
    data = load_dataset(data_path)
    train, validation, test = split_sessions(data)
    columns = feature_names()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(dataset='CMU DSL Strong Password',
                  dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                  seed=seed, feature_count=len(columns), subjects=int(data.subject.nunique()),
                  split={name: dict(rows=len(part), sessions=sorted(int(s) for s in part.sessionIndex.unique()))
                         for name, part in zip(('train', 'validation', 'test'), (train, validation, test))},
                  environment=dict(python=platform.python_version(), scikit_learn=sklearn.__version__,
                                   numpy=np.__version__, pandas=pd.__version__, joblib=joblib.__version__),
                  models={})
    for name in names:
        model = make_model(name, seed)
        start = perf_counter()
        model.fit(train[columns], train.subject)
        fit_seconds = perf_counter() - start
        val_scores = model.predict_proba(validation[columns])
        threshold = select_threshold(val_scores, validation.subject.to_numpy(), model.classes_)
        test_scores = model.predict_proba(test[columns])
        predictions = model.classes_[test_scores.argmax(axis=1)]
        result = dict(identification=identification_metrics(test.subject, predictions),
                      verification=verification_rates(test_scores, test.subject.to_numpy(), model.classes_, threshold),
                      fit_seconds=round(fit_seconds, 3), estimator=repr(model))
        report['models'][name] = result
        joblib.dump(dict(model=model, features=columns, threshold=threshold,
                         scikit_learn_version=sklearn.__version__, seed=seed), output_dir / f'{name}.joblib')
        pd.DataFrame(dict(actual=test.subject.to_numpy(), predicted=predictions,
                          session=test.sessionIndex.to_numpy(), repetition=test.rep.to_numpy())).to_csv(
                              output_dir / f'{name}_predictions.csv', index=False)
        print(f'{name}: accuracy={result["identification"]["accuracy"]:.4f}, '
              f'macro-F1={result["identification"]["macro_f1"]:.4f}, threshold={threshold:.3f}', flush=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_PATH)
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts'))
    parser.add_argument('--report', type=Path, default=Path('reports/results.json'))
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        run(args.data, args.output_dir, args.report, args.models, args.seed)
    except (ValueError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
