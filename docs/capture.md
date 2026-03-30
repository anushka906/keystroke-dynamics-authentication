# Local samples

Run `python -m keystroke.collect`. In its typing entry, type the public benchmark text `.tie5Roanl`, then press and release Enter. Use Shift or Caps Lock for the capital R. The collector ignores modifier events and measures the R key itself.

Only events in that entry are captured. No global keyboard hook is installed. Use a pseudonymous subject ID, a session from 1-8, and a repetition from 1-50. A mistake or repeated key requires restarting. Save after releasing all keys; existing sample files are not overwritten. Samples are kept in `data/captures/`, which Git ignores.

```sh
python -m keystroke.prepare
python -m keystroke.train --data data/captures/features.csv --output-dir artifacts/local --report artifacts/local/results.json
```

Training requires at least two subjects, with samples for every subject in sessions 1-5, session 6, and sessions 7-8. Gather multiple repetitions on separate occasions. A tiny local collection is useful for checking the workflow, not for estimating dependable authentication performance.

To verify a sample from a held-out session against a model trained on your local subjects:

```sh
python -m keystroke.predict --model artifacts/local/random_forest.joblib --samples data/captures/local01_8_1.json --events --claim local01
```

The CMU-trained model recognizes CMU subject IDs only; it has not enrolled you. Personal captures and their timing CSVs should stay local. The `prepare` command converts the JSON events into the same 31 timing features used for training.

An event JSON contains `subject`, `sessionIndex`, `rep`, and an `events` list. Each event has a canonical `key`, an `action` (`down` or `up`), and a `time` in seconds relative to sample start. Keys and sequence are defined in `keystroke/features.py`.
