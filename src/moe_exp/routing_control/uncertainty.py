"""Conservative simultaneous bounds for paired binary outcomes, including zero discordance."""
import math
import numpy as np


def paired_binary_bounds(left, right, families, *, contrasts, alpha=.05):
    left, right = np.asarray(left), np.asarray(right)
    if (left.shape != right.shape or left.ndim != 2 or left.shape[0] != len(families)
            or not len(families) or left.shape[1] < 1
            or not np.isin(left, [0, 1]).all() or not np.isin(right, [0, 1]).all()):
        raise ValueError('aligned question-by-seed binary outcomes and frozen families required')
    if type(contrasts) is not int or contrasts < 1 or not 0 < alpha < 1:
        raise ValueError('fixed positive multiplicity and valid error level required')
    groups = sorted(set(families))
    weights = np.array([families.count(f)/len(families) for f in groups])
    difference = float(np.mean(left.astype(float)-right))
    discordant = np.any(left != right, axis=1)
    events = sum(any(discordant[i] for i, f in enumerate(families) if f == group) for group in groups)
    if events == 0:
        # Independent family event probabilities need not be identical: the
        # AM-GM bound on their no-event probability bounds their average.
        event_upper = 1-(alpha/contrasts)**(1/len(groups))
        bound = min(1., float(len(groups)*weights.max()*event_upper))
        interval = [-bound, bound]
        method = 'zero-discordance independent-family event bound; weighted worst case'
    else:
        half = math.sqrt(2*float(np.sum(weights*weights))*math.log(2*contrasts/alpha))
        interval = [max(-1., difference-half), min(1., difference+half)]
        method = 'weighted independent-family Hoeffding bound'
    return {'estimate': difference, 'simultaneous_interval': interval,
        'method': method, 'family_error_level': alpha, 'multiplicity_contrasts': contrasts,
        'n_questions': len(families), 'n_families': len(groups), 'discordant_families': events,
        'assumptions': 'independent frozen families, binary outcomes, equal question weights and seed means',
        'interpretation': 'no zero-width interval or equivalence conclusion from zero observed discordance'}
