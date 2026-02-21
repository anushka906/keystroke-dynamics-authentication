import pytest

from keystroke.features import KeyEvent, extract_features, feature_names


def test_overlapping_keys_keep_negative_up_down_time():
    events = [KeyEvent('a', 'down', 1), KeyEvent('b', 'down', 1.1),
              KeyEvent('a', 'up', 1.2), KeyEvent('b', 'up', 1.3)]
    values = extract_features(events, ('a', 'b'))
    assert values['H.a'] == pytest.approx(.2)
    assert values['DD.a.b'] == pytest.approx(.1)
    assert values['UD.a.b'] == pytest.approx(-.1)
    assert values['H.a'] + values['UD.a.b'] == pytest.approx(values['DD.a.b'])


@pytest.mark.parametrize('events', [
    [KeyEvent('a', 'down', 1)],
    [KeyEvent('a', 'up', 1)],
    [KeyEvent('a', 'down', 1), KeyEvent('a', 'down', 2)],
    [KeyEvent('a', 'down', float('nan'))],
    [KeyEvent('a', 'down', 2), KeyEvent('a', 'up', 1)],
    [KeyEvent('b', 'down', 1), KeyEvent('b', 'up', 2)],
])
def test_invalid_samples_are_rejected(events):
    with pytest.raises(ValueError):
        extract_features(events, ('a',))


def test_benchmark_schema_has_31_unique_features():
    names = feature_names()
    assert len(names) == len(set(names)) == 31
    assert names[-1] == 'H.Return'
