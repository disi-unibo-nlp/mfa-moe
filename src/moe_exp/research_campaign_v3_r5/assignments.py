"""Balanced diagnostic allocation overlay; original enrollment stays sealed."""
from collections import defaultdict
from pathlib import Path
from .contracts import BENCHMARKS, load, publish, digest, require


def balanced(records):
    groups = defaultdict(set)
    for row in records:
        groups[row['benchmark']].add(row['family'])
    assignments = {}
    cursor = 0
    for benchmark in BENCHMARKS:
        families = sorted(groups[benchmark], key=lambda f: (digest(['diagnostic-family', f]), f))
        for family in families:
            assignments[(benchmark, family)] = [0, 1 + cursor % 8]
            cursor += 1
    return [{**r, 'configuration_indices': assignments[r['benchmark'], r['family']]} for r in records]


def roster(campaign, model):
    root = Path(campaign['root'])
    source = load(root/model/'PREFIXES.json')
    if not campaign.get('balanced_sensitivity_overlay'):
        return source['records']
    # The common cohort assigns a family once, independent of model, prefix
    # availability, repeated questions, and observed outcomes.
    common = balanced(load(root/'COHORTS.json')['causal'])
    overlay = {(r['benchmark'], r['family']): r['configuration_indices'] for r in common}
    rows = [{**r, 'configuration_indices': overlay[r['benchmark'], r['family']]} for r in source['records']]
    manifest = publish(root/model/'SENSITIVITY_ASSIGNMENTS.json', {
        'prefix_roster_sha256': source['sha256'], 'records': rows,
        'central_primary': 'MMM', 'secondary_role': 'diagnostic',
        'original_enrollment_and_prefixes_unchanged': True, 'outcomes_used': False})
    require(len(rows) == len(source['records']), 'overlay changed enrollment')
    return manifest['records']
