"""Fixed-horizon reasoning outcomes, explicit closure and conservative bounds."""
from .common import CLASSES, require


def occupancy(labels, pulse, horizon, *, reasoning_end=None, natural_eos=None, nonreasoning=()):
    """labels maps continuation-token position to class or None."""
    require(isinstance(pulse, int) and isinstance(horizon, int) and 0 <= pulse < horizon,
            'invalid measurement horizon')
    ends = [x for x in (reasoning_end, natural_eos) if x is not None]
    end = min(ends) if ends else None
    counts = {c: 0 for c in CLASSES}
    unknown = ended = 0
    nonreasoning = set(nonreasoning)
    for p in range(pulse, horizon):
        if end is not None and p >= end:
            ended += 1
        elif p in nonreasoning:
            continue
        elif labels.get(p) in counts:
            counts[labels[p]] += 1
        else:
            unknown += 1
    n = horizon - pulse
    return {'classes': {c: [v/n, (v+unknown)/n] for c, v in counts.items()},
            'unknown': unknown/n, 'reasoning_ended': ended/n, 'denominator': n}


def first_sentence(sentences, *, natural_eos=False, truncated=False):
    complete = [s for s in sentences if s.get('complete', True)]
    if complete:
        label = min(complete, key=lambda s: s['start']).get('label')
        return {c: [int(label == c), int(label == c)] if label in CLASSES else [0, 1] for c in CLASSES}
    return {c: [0, 0] if natural_eos and not truncated else [0, 1] for c in CLASSES}


def difference(a, b, sign=1):
    low, high = a[0]-b[1], a[1]-b[0]
    return [low, high] if sign == 1 else [-high, -low]
