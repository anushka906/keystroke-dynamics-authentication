"""Three supervised baselines with fixed, reproducible settings."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_NAMES = ('random_forest', 'svm', 'logistic_regression')


def make_model(name, seed=42):
    if name == 'random_forest':
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    if name == 'svm':
        return make_pipeline(StandardScaler(), SVC(C=10, probability=True, random_state=seed))
    if name == 'logistic_regression':
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed))
    raise ValueError(f'Unknown model: {name}')
