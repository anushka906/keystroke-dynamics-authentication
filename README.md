# Keystroke Dynamics Authentication

Classifies fixed-text typing samples using hold, down-down, and up-down timings. Includes Random Forest, SVM, Logistic Regression, and a local sample collector.

## Run

Use Python 3.10 or later in a virtual environment. Run commands from the repository root:

```sh
pip install -r requirements.txt
python -m keystroke.data
python -m keystroke.train
```

Random Forest reached **82.00% identification accuracy** on CMU benchmark sessions 7-8, with sessions 1-5 for training and session 6 for validation. See [evaluation details](docs/evaluation.md) and [results](reports/results.json).

To identify a feature CSV or verify a known subject:

```sh
python -m keystroke.predict --model artifacts/random_forest.joblib --samples path/to/features.csv --claim s002
```

Omit `--claim` for identification. Add `--events` when supplying a captured JSON file. Load only model files you trust.

## Collect and test

```sh
python -m keystroke.collect
python -m keystroke.prepare
pip install -r requirements-dev.txt
python -m pytest -q
```

The collector records only its fixed-text input; captures stay local. See [capture instructions](docs/capture.md) for training on your own samples. This is an experimental known-user classifier, not a production login system.

Data: [CMU benchmark](https://www.cs.cmu.edu/~keystroke/), Kevin Killourhy and Roy Maxion, DSN 2009. This project's evaluation differs from the original paper.
