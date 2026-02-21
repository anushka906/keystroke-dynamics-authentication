"""Extract seconds-based timings from one fixed-text typing sample."""

from dataclasses import dataclass
from math import isfinite

PASSWORD = '.tie5Roanl'
KEYS = ('period', 't', 'i', 'e', 'five', 'Shift.r', 'o', 'a', 'n', 'l', 'Return')


def feature_names(keys=KEYS):
    names = []
    for i, key in enumerate(keys):
        names.append(f'H.{key}')
        if i + 1 < len(keys):
            pair = f'{key}.{keys[i + 1]}'
            names.extend((f'DD.{pair}', f'UD.{pair}'))
    return names


@dataclass(frozen=True)
class KeyEvent:
    key: str
    action: str
    time: float


def extract_features(events, keys=KEYS):
    """Require one down/up pair per key; allow overlapping key presses."""
    downs, ups, order = {}, {}, []
    previous = -float('inf')
    for event in events:
        if event.key not in keys or event.action not in {'down', 'up'}:
            raise ValueError('Unexpected key or action')
        if not isfinite(event.time) or event.time < 0 or event.time < previous:
            raise ValueError('Event times must be finite, nonnegative, and ordered')
        previous = event.time
        if event.action == 'down':
            if event.key in downs:
                raise ValueError('Repeated keydown; restart the sample')
            downs[event.key] = event.time
            order.append(event.key)
        else:
            if event.key not in downs or event.key in ups:
                raise ValueError('Unmatched or repeated keyup')
            ups[event.key] = event.time
    if tuple(order) != tuple(keys) or set(ups) != set(keys):
        raise ValueError('Incomplete sample or incorrect key sequence')
    result = {}
    for i, key in enumerate(keys):
        result[f'H.{key}'] = ups[key] - downs[key]
        if i + 1 < len(keys):
            following = keys[i + 1]
            pair = f'{key}.{following}'
            result[f'DD.{pair}'] = downs[following] - downs[key]
            result[f'UD.{pair}'] = downs[following] - ups[key]
    return result
