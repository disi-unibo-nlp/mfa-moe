"""Whole-study admission, priced batch tails, and hard allocation reservations."""
import math
import numpy as np
from .common import require


def preparation(config):
    s, b = config['slurm'], config['budget']
    rows = [
        {'stage': 'qualification', 'models': 8, 'attempts': 2, 'hours': 4, 'billing_units': 16},
        {'stage': 'reader_pilot', 'models': 1, 'attempts': 2, 'hours': 2, 'billing_units': 16},
        {'stage': 'validation', 'models': 1, 'attempts': 1, 'hours': 8, 'billing_units': 4},
    ]
    total = sum(r['models']*r['attempts']*r['hours']*r['billing_units'] for r in rows)*b['margin']
    require(total <= b['preparation_hours'], 'preparation exceeds authorized envelope')
    return {'rows': rows, 'margin_inclusive_hours': total}


def reader_price(batches):
    """A batch is priced once. Tail batches cost a whole batch unless measured."""
    require(batches, 'reader price requires actual batch receipts')
    profiles = {}
    for r in batches:
        require(r['status'] == 'complete' and 0 < r['batch_size'] <= 64 and r['seconds'] > 0,
                'invalid reader batch price')
        key = str(r['input_bin'])
        profile = profiles.setdefault(key, {'batch_seconds': 0, 'load_seconds': 0,
                                            'input_min': r['input_min'], 'input_max': r['input_max'],
                                            'measured_batches': 0, 'max_output_tokens': 16384})
        require(r['max_output_tokens'] == 16384, 'reader instrument changed')
        profile['batch_seconds'] = max(profile['batch_seconds'], r['seconds'])
        profile['load_seconds'] = max(profile['load_seconds'], r['load_seconds'])
        profile['input_min'] = min(profile['input_min'], r['input_min'])
        profile['input_max'] = max(profile['input_max'], r['input_max'])
        profile['measured_batches'] += 1
    return profiles


def reader_hours(counts, profiles, *, allocations, billing_units=16, batch_size=64):
    seconds = 0
    for key, count in counts.items():
        require(str(key) in profiles, f'unpriced reader input profile: {key}')
        require(count >= 0, 'negative context count')
        seconds += math.ceil(count/batch_size)*profiles[str(key)]['batch_seconds']
    seconds += allocations*max((p['load_seconds'] for p in profiles.values()), default=0)
    return seconds/3600*billing_units


def context_forecast(rows, multiplicities, *, draws=10000, seed=20261015):
    """Joint family resamples of raw workload, not scientific analysis weights."""
    require(rows and draws > 0, 'missing discovery context inventory')
    rng = np.random.default_rng(seed)
    groups = {}
    for r in rows:
        groups.setdefault((r['benchmark'], r['fold']), {}).setdefault(r['family'], []).append(r)
    simulated = np.zeros(draws)
    for families in groups.values():
        values = [sum(r['contexts'] * multiplicities[r['model']] for r in members)
                  for members in families.values()]
        array = np.asarray(values, dtype=float)
        simulated += array[rng.integers(0, len(array), size=(draws, len(array)))].sum(1)
    return {'point_contexts': sum(sum(r['contexts']*multiplicities[r['model']] for r in members)
                                  for families in groups.values() for members in families.values()),
            'upper_contexts': int(math.ceil(np.quantile(simulated, .99))), 'draws': draws,
            'risk_statement': 'Discovery-family forecast; does not guarantee treatment density or output tails.'}


def admit(config, forecasts, *, actual_hours, committed_hours, available_hours):
    """Full -> predetermined fallback -> explicit shortfall. Never alter science."""
    b = config['budget']
    available = min(available_hours, b['ceiling_hours'] - actual_hours) - committed_hours
    require(available >= 0, 'existing commitments exhaust entitlement')
    decisions = []
    for mode in ('full', 'fallback'):
        forecast = forecasts.get(mode)
        if not forecast or not forecast.get('profiles_complete'):
            decisions.append({'mode': mode, 'status': 'unpriced',
                              'missing': (forecast or {}).get('missing', ['complete measured forecast'])})
            continue
        require(all(v >= 0 and math.isfinite(v) for v in forecast['base_hours'].values()),
                'invalid resource price')
        reserved = sum(forecast['base_hours'].values())*b['margin'] + b['closeout_hours']
        decisions.append({'mode': mode, 'status': 'admitted' if reserved <= available else 'over_budget',
                          'reserved_hours': reserved, 'available_hours': available,
                          'shortfall_hours': max(0, reserved-available)})
        if reserved <= available:
            return {'status': 'admitted', 'mode': mode, 'decisions': decisions,
                    'reserved_hours': reserved, 'remaining_entitlement': available}
    return {'status': 'not_admitted', 'mode': None, 'decisions': decisions,
            'remaining_entitlement': available, 'production_released': False}


def concurrency(config, footprints, free_bytes, *, gpu_limit=None):
    reserve = config['budget']['reserve_gib_per_filesystem']*1024**3
    limit = min(config['slurm']['max_gpu_workers'], gpu_limit or config['slurm']['max_gpu_workers'])
    for filesystem in ('work', 'fast'):
        require(filesystem in footprints and filesystem in free_bytes, 'missing filesystem measurement')
        fixed = footprints[filesystem]['fixed_bytes']
        per_worker = footprints[filesystem]['per_worker_peak_bytes']
        require(per_worker > 0 and fixed >= 0, 'invalid storage footprint')
        limit = min(limit, max(0, int((free_bytes[filesystem]-reserve-fixed)//per_worker)))
    return limit
