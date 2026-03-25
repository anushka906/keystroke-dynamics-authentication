"""Convert locally captured event files into a timing-feature CSV."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import load_dataset
from .features import KeyEvent, extract_features


def convert(input_dir, output):
    paths = sorted(Path(input_dir).glob('*.json'))
    if not paths:
        raise ValueError('No captured JSON samples found')
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding='utf-8'))
        timings = extract_features([KeyEvent(**event) for event in payload['events']])
        rows.append(dict(subject=payload['subject'], sessionIndex=payload['sessionIndex'],
                         rep=payload['rep'], **timings))
    # Validate before writing so malformed captures cannot replace good output.
    from io import StringIO
    validated = load_dataset(StringIO(pd.DataFrame(rows).to_csv(index=False)))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(output, index=False)
    return len(validated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('data/captures'))
    parser.add_argument('--output', type=Path, default=Path('data/captures/features.csv'))
    args = parser.parse_args()
    try:
        count = convert(args.input_dir, args.output)
        print(f'Exported {count} samples to {args.output}')
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
