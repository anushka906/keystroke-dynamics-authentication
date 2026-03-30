# Evaluation

## Data and features

The [CMU DSL Strong Password dataset](https://www.cs.cmu.edu/~keystroke/) contains 20,400 samples from 51 subjects, each typing `.tie5Roanl` followed by Return. The downloader checks the published MD5 checksum. The run's SHA-256 and package versions are recorded in `reports/results.json`.

The benchmark supplies 31 timing features in seconds:

- Hold (`H`): key release minus the same key's press.
- Down-down (`DD`): next key press minus current key press.
- Up-down (`UD`): next key press minus current key release. Negative values represent overlapping presses and are retained.

`keystroke.features` also computes these features from locally captured press/release events. Subject, session, and repetition identifiers are never model inputs. The benchmark run uses the supplied timings; it does not reconstruct unavailable raw events.

## Split and models

Each subject occurs in all three partitions:

| Partition | Sessions | Samples |
| --- | --- | ---: |
| Training | 1-5 | 12,750 |
| Validation | 6 | 2,550 |
| Test | 7-8 | 5,100 |

StandardScaler is fitted on training data only for SVM and Logistic Regression. Random Forest uses 200 trees; SVM uses the RBF kernel with C=10 and probability estimates; Logistic Regression uses up to 3,000 iterations. Seed: 42. No hyperparameter search is performed. Identification chooses the identity with the highest predicted probability.

## Measured results

| Model | Identification accuracy | Macro F1 | Verification FAR | Verification FRR |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 82.00% | 0.8116 | 2.36% | 6.53% |
| SVM | 77.08% | 0.7633 | 2.97% | 8.04% |
| Logistic Regression | 71.57% | 0.7075 | 3.93% | 9.33% |

For claimed-identity verification, a sample is accepted when the claimed subject's probability meets the model's threshold. A separate threshold for each classifier minimizes the mean of false-accept and false-reject rates on session 6, over a fixed 0.005-spaced grid. It is then held fixed for test sessions 7-8.

Each test sample supplies one genuine claim and 50 incorrect known-user claims: 5,100 genuine trials and 255,000 impostor trials. FAR is accepted incorrect claims / impostor trials. FRR is rejected genuine claims / genuine trials. These claims share samples and should not be interpreted as independent attacks. Probabilities and thresholds are model-specific.

## Limits and reproduction

Run `python -m keystroke.data` followed by `python -m keystroke.train`. The JSON report includes exact counts and estimator settings; serialized models and sample-level predictions are saved locally in `artifacts/`. Runtime varies by machine, and package versions can affect results.

This is closed-set identification and verification: every evaluated person is already represented in training. It does not establish performance on unseen people, different passwords, devices, or real attacks. It is not the original CMU paper's anomaly-detector protocol, and its accuracy is not comparable to that paper's equal-error rates.

Reference: Kevin S. Killourhy and Roy A. Maxion, "Comparing Anomaly Detectors for Keystroke Dynamics," DSN 2009, pp. 125-134.
