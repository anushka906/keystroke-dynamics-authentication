import json

import pytest

from keystroke.collect import Capture
from keystroke.features import PASSWORD
from keystroke.predict import read_samples


def completed_sample():
    sample = Capture()
    for i, character in enumerate(PASSWORD + '\r'):
        sample.press(i, character, i * .2)
        sample.release(i, i * .2 + .08)
    return sample


def test_capture_export_round_trips_into_feature_reader(tmp_path):
    capture = completed_sample()
    assert capture.complete
    path = capture.export(tmp_path, 'local01', 1, 1)
    assert len(json.loads(path.read_text())['events']) == 22
    samples = read_samples(path, events=True)
    assert samples.shape == (1, 31)
    assert samples.iloc[0]['H.period'] == pytest.approx(.08)
    with pytest.raises(FileExistsError):
        capture.export(tmp_path, 'local01', 1, 1)


def test_capture_requires_exact_text_and_no_key_repeat():
    with pytest.raises(ValueError):
        Capture().press(1, 'wrong', 0)
    capture = Capture()
    capture.press(1, '.', 0)
    with pytest.raises(ValueError):
        capture.press(1, 't', .1)


def test_subject_cannot_escape_capture_directory(tmp_path):
    with pytest.raises(ValueError):
        completed_sample().export(tmp_path, '../outside', 1, 1)


def test_release_tracks_physical_key_code():
    capture = Capture()
    capture.press(11, '.', 0)
    capture.release(12, .1)
    assert 11 in capture.active
    capture.release(11, .2)
    assert capture.events[-1].key == 'period'
