"""Download and validate the CMU fixed-password benchmark."""

import argparse
import hashlib
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

from .features import feature_names

SOURCE = 'https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv'
MD5 = '470235f96568f28f9ea0da62234ec857'
DEFAULT_PATH = Path('data/raw/DSL-StrongPasswordData.csv')


def download(path=DEFAULT_PATH):
    path = Path(path)
    with urllib.request.urlopen(SOURCE, timeout=60) as response:
        content = response.read()
    if hashlib.md5(content).hexdigest() != MD5:
        raise ValueError('Dataset checksum differs from the published CMU checksum')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def load_dataset(path):
    data = pd.read_csv(path)
    required = ['subject', 'sessionIndex', 'rep', *feature_names()]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if data.empty or data[required].isna().any().any():
        raise ValueError('Dataset is empty or has missing values')
    if data['subject'].astype(str).str.strip().eq('').any():
        raise ValueError('Subject identifiers must not be blank')
    numeric = data[['sessionIndex', 'rep', *feature_names()]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError('Dataset contains non-finite values')
    for column, upper in [('sessionIndex', 8), ('rep', 50)]:
        values = data[column].to_numpy(dtype=float)
        if not ((values == np.floor(values)) & (values >= 1) & (values <= upper)).all():
            raise ValueError(f'Invalid {column}')
    nonnegative = [n for n in feature_names() if n.startswith(('H.', 'DD.'))]
    if (data[nonnegative] < 0).any().any():
        raise ValueError('Hold and down-down times cannot be negative')
    if data.duplicated(['subject', 'sessionIndex', 'rep']).any():
        raise ValueError('Duplicate subject/session/repetition identifiers')
    data['subject'] = data['subject'].astype(str)
    return data[required].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    path = download(args.output)
    data = load_dataset(path)
    print(f'Saved {len(data)} samples from {data.subject.nunique()} subjects to {path}')


if __name__ == '__main__':
    main()
