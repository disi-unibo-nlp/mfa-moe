"""Probability-sampled trajectory disagreement with exact estimand coefficients.

Each sampled occurrence retains its token and planned-population coefficient.
Context deduplication saves reader work without changing inclusion probabilities.
The finite-population variance is distinct from the shared family bootstrap.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from moe_exp.io import BENCHMARKS, CLASSES, digest, require


def audit_frame(value: dict, assignment: dict, replay: dict) -> list[dict]:
    """Return all occurrences contributing lexical occupancy, including truncation.

    Args:
        value: Full planned cohort.
        assignment: Original intervention assignment.
        replay: Qualified token extraction.

    Returns:
        Occurrences with the actual full-window contrast coefficient.
    """
    if assignment['arm'] in ('native', 'native_repeat'):
        return []
    families = {(row['assignment']['benchmark'], row['assignment']['family']) for row in value['records']}
    per_benchmark = Counter(benchmark for benchmark, _ in families)
    direction = assignment['arm'].split('_')[1]
    selector = assignment['arm'].split('_')[0]
    sign = (1 if direction == 'activate' else -1)*(1 if selector == 'target' else -1)
    coefficient = sign/(len(BENCHMARKS)*per_benchmark[assignment['benchmark']]*2*len(CLASSES)*2*512)
    ended = replay['reasoning_end']
    if replay['finish_reason'] == 'stop':
        ended = min(ended, replay['generated_tokens']) if ended is not None else replay['generated_tokens']
    excluded = set(replay['nonreasoning'])
    result = []
    for sentence in replay['sentences']:
        tokens = [position for position in range(max(0, sentence['token_start']), min(512, sentence['token_end']))
                  if position not in excluded and (ended is None or position < ended)]
        if not tokens:
            continue
        result.append({
            'occurrence_uid': digest(['trajectory-occurrence', assignment['uid'], sentence['sentence_index']]),
            'uid': assignment['uid'],
            'sentence_index': sentence['sentence_index'],
            'model': assignment['model'],
            'layer_regime': assignment['layer_regime'],
            'direction': direction,
            'selector': selector,
            'target_class': assignment['class'],
            'family': assignment['family'],
            'benchmark': assignment['benchmark'],
            'fold': assignment['fold'],
            'tokens': len(tokens),
            'coefficient': coefficient*len(tokens),
            'straddles_prefix': sentence['straddles_prefix'],
            'complete': sentence['complete'],
            'inputs': sentence['inputs'],
        })
    return result


def sample_occurrences(value: dict, occurrences: list[dict]) -> dict:
    """Freeze stratified simple random occurrence samples without inspecting labels.

    Args:
        value: Cohort with frozen seed and purpose.
        occurrences: Complete eligible occupancy occurrence frame.

    Returns:
        Frame digest, all frame identities, selected occurrences and probabilities.
    """
    strata = defaultdict(list)
    for row in occurrences:
        strata['|'.join(row[key] for key in ('model', 'layer_regime', 'direction', 'selector'))].append(row)
    selected, sizes = [], {}
    for model in ('gpt', 'qwen36'):
        for regime in ('single', 'triple'):
            for direction in ('activate', 'deactivate'):
                for selector in ('target', 'random'):
                    key = '|'.join((model, regime, direction, selector))
                    rows = strata[key]
                    ordered = sorted(rows, key=lambda row: digest([
                        value['selection']['seed'], value['selection']['audit_purpose'], row['occurrence_uid']]))
                    size = min(value['selection']['audit_per_stratum'], len(rows))
                    sizes[key] = {'population': len(rows), 'sample': size}
                    selected.extend({**row, 'stratum': key, 'inclusion_probability': size/len(rows)}
                                    for row in ordered[:size])
    require(len({row['occurrence_uid'] for row in occurrences}) == len(occurrences), 'duplicate audit occurrence')
    return {'frame_sha256': digest(sorted(occurrences, key=lambda row: row['occurrence_uid'])),
            'frame': [{'occurrence_uid': row['occurrence_uid'], 'stratum': '|'.join(row[key] for key in
                       ('model', 'layer_regime', 'direction', 'selector'))} for row in occurrences],
            'strata': sizes, 'selected': selected, 'sampling': 'SRS without replacement by purpose-specific hash'}


def _sampling_variance(size: dict, observed: list[float]) -> float:
    """Use the finite-population correction only for fully labelled SRS samples."""
    import numpy as np
    population, sample = size['population'], size['sample']
    if sample == population:
        return 0.
    if len(observed) != sample or sample < 2:
        return math.nan
    return population**2*(1-sample/population)*float(np.var(observed, ddof=1))/sample


def disagreement(audit: dict, labels: dict, bootstrap=None) -> dict:
    """Estimate semantic-minus-lexical error by Horvitz–Thompson totals.

    Args:
        audit: Frozen occurrence frame and selected sample.
        labels: Exact context labels; missing or unsupported contexts stay unknown.
        bootstrap: Optional shared family draws, separate from survey sampling.

    Returns:
        Directional totals, identification bounds and separate sampling uncertainty.
    """
    import numpy as np
    result = {}
    for direction in ('activate', 'deactivate'):
        lower = upper = variance = 0.
        lower_draws = np.zeros(len(bootstrap.weights)) if bootstrap is not None else None
        upper_draws = np.zeros(len(bootstrap.weights)) if bootstrap is not None else None
        missing = 0
        for key, size in audit['strata'].items():
            rows = [row for row in audit['selected'] if row['stratum'] == key and row['direction'] == direction]
            if not rows:
                continue
            observed = []
            for row in rows:
                lexical = int(row['lexical_label'] == row['target_class'])
                label = labels.get(row['annotation_uid'])
                coefficient = row['coefficient']
                if label is None:
                    bounds = sorted([coefficient*(-lexical), coefficient*(1-lexical)])
                    missing += 1
                else:
                    delta = coefficient*(int(label == row['target_class'])-lexical)
                    bounds = [delta, delta]
                    observed.append(delta)
                lower += bounds[0]/row['inclusion_probability']
                upper += bounds[1]/row['inclusion_probability']
                if bootstrap is not None:
                    weights = bootstrap.weights[:, bootstrap.index[row['family']]]
                    lower_draws += weights*bounds[0]/row['inclusion_probability']
                    upper_draws += weights*bounds[1]/row['inclusion_probability']
            variance += _sampling_variance(size, observed)
        result[direction] = {
            'lower': lower, 'upper': upper, 'point_identified': missing == 0,
            'estimate': lower if missing == 0 else None,
            'unknown_sampled_occurrences': missing,
            'finite_population_variance': variance if math.isfinite(variance) else None,
            'finite_population_se': math.sqrt(variance) if math.isfinite(variance) else None,
            'family_bootstrap_bound_interval': ([float(np.quantile(lower_draws, .025)),
                                                  float(np.quantile(upper_draws, .975))]
                                                 if bootstrap is not None else None),
            'uncertainties_combined': False,
        }
    return {'estimator': 'Horvitz–Thompson total with fixed token/family/benchmark/model/class/regime coefficients',
            'directions': result, 'deduplicated_contexts_are_not_independent_annotations': True,
            'interpretation': 'measurement disagreement diagnostic; no confusion correction or sub-percentage bias certification'}
