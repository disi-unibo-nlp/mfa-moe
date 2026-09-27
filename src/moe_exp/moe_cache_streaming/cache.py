"""Dependency-free, per-layer cache simulation with simultaneous top-k requests."""
from collections import OrderedDict


def simulate(sequence, capacity, pinned=()):
    """Cold cache per response. Pins load once, charged even if never requested.

    Top-k experts are a set: hits are evaluated before any insertion. Current
    experts cannot evict each other. Sorted IDs break simultaneous recency ties.
    """
    pinned = set(pinned)
    if capacity < 1 or len(pinned) > capacity:
        raise ValueError('Invalid cache capacity/pin budget')
    dynamic = OrderedDict()
    hits = misses = requests = 0
    for step in sequence:
        ids = set(step)
        if len(ids) != len(step) or not ids or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError('Expected distinct nonnegative expert IDs')
        if len(pinned | ids) > capacity:
            raise ValueError('Capacity cannot hold pins and this simultaneous top-k set')
        hits += sum(i in pinned or i in dynamic for i in ids)
        absent = ids - pinned - dynamic.keys()
        misses += len(absent)
        requests += len(ids)
        for i in sorted(ids - pinned):
            if i in dynamic:
                dynamic.move_to_end(i)
        for i in sorted(absent):
            while len(dynamic) + len(pinned) >= capacity:
                victim = next(k for k in dynamic if k not in ids)
                del dynamic[victim]
            dynamic[i] = None
    return dict(requests=requests, hits=hits, demand_loads=misses,
                pin_loads=len(pinned), total_loads=misses + len(pinned))
