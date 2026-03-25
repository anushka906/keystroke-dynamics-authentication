import pandas as pd
import pytest

from keystroke.collect import Capture
from keystroke.features import PASSWORD
from keystroke.prepare import convert


def test_captured_samples_become_training_rows(tmp_path):
    captures = tmp_path / 'captures'
    sample = Capture()
    for i, character in enumerate(PASSWORD + '\r'):
        sample.press(i, character, i * .2)
        sample.release(i, i * .2 + .05)
    sample.export(captures, 'a', 1, 1)
    sample.export(captures, 'b', 2, 1)
    output = tmp_path / 'features.csv'
    assert convert(captures, output) == 2
    data = pd.read_csv(output)
    assert data.shape == (2, 34)
    assert list(data.subject) == ['a', 'b']


def test_empty_capture_directory_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='No captured'):
        convert(tmp_path, tmp_path / 'features.csv')
