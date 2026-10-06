"""Recover the COMPLETE duplicate graph from small cached prompt metadata, before any new fit.

The old near-duplicate report stores only ten top pairs and the confirm-crossing subset;
using that report alone would lose transitive families. This reproduces its frozen method
and writes all edges. No original split or model fit is modified.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from moe_exp.routing_control.design import digest, families, freeze_pools

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
OUT = Path(__file__).resolve().parents[2] / 'report/experimental-resume-v1'


def main():
    paths = {'split': ROOT / 'steering-v1/manifests/split-v1.json',
             'prompts': ROOT / 'steering-v1/manifests/prompt-table-v1.json',
             'historical_audit': ROOT / 'forum/tests/r3_integrity/out/near_dup.v2.json'}
    inputs = {name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
              for name, path in paths.items()}
    split_record = json.loads(paths['split'].read_text())
    split = {f"{q['dataset']}|{q['source_problem_id']}": q['split'] for q in split_record['questions']}
    prompts = json.loads(paths['prompts'].read_text())['questions']
    keys = sorted(prompts)
    if set(keys) != set(split):
        raise ValueError('canonical prompt and split populations differ')
    by_sha, buckets = defaultdict(list), defaultdict(list)
    shingles = {}
    for i, key in enumerate(keys):
        by_sha[prompts[key]['prompt_sha256']].append(key)
        ids = prompts[key]['prompt_token_ids']
        # Tuple hashes are stable for integers. Keep the actual tuples to remove the
        # theoretical hash-collision ambiguity in the old metadata audit.
        shingles[i] = {tuple(ids[j:j + 8]) for j in range(max(0, len(ids) - 7))}
        for shingle in shingles[i]:
            buckets[shingle].append(i)
    uncommon = {shingle for shingle, members in buckets.items() if len(members) <= 20}
    effective = {i: len(shingle_set & uncommon) for i, shingle_set in shingles.items()}
    counts = Counter()
    for shingle in uncommon:
        members = buckets[shingle]
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                counts[members[x], members[y]] += 1
    edges = [{'a': keys[i], 'b': keys[j], 'shared': count,
              'containment': count / min(effective[i], effective[j])}
             for (i, j), count in counts.items()
             if min(effective[i], effective[j]) >= 6 and count / min(effective[i], effective[j]) >= .85]
    edges.sort(key=lambda edge: (edge['a'], edge['b']))
    audit = json.loads(paths['historical_audit'].read_text())
    if len(edges) != audit['n_near_duplicate_pairs']:
        raise ValueError('full graph no longer reproduces the historical audit edge count')
    crossing = [e for e in edges if (split[e['a']] == 'confirm') != (split[e['b']] == 'confirm')]
    if len(crossing) != audit['n_cross_confirm_pairs']:
        raise ValueError('confirm-crossing count differs from the historical audit')
    exact = [sorted(group) for group in by_sha.values() if len(group) > 1]
    groups = families(keys, edges, exact)
    payload = {'schema': 'routing-control-family-freeze-v1', 'inputs': inputs,
               'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'method': {'token_ngram': 8, 'max_bucket': 20, 'min_shingles': 6, 'containment': .85},
               'canonical_question_count': len(keys), 'full_edges': edges, 'exact_groups': exact,
               'split_preserved': True, 'clean_primary': 'exact-ID dev+tune; original split untouched',
               'clean_sensitivity': 'exclude every dev/tune family connected to any confirm question',
               'new_parent_pools': freeze_pools(groups, split),
               'behavioral_eligibility': 'UNMEASURED; parent counts do not qualify transition detectors'}
    result = {**payload, 'sha256': digest(payload)}
    path = OUT / 'family-freeze.json'
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError('family grouping is frozen; changed inputs require a new protocol version')
    path.write_text(json.dumps(result, indent=1) + '\n')
    assert json.loads(path.read_text()) == result
    print(json.dumps({'questions': len(keys), 'edges': len(edges), 'cross_confirm_edges': len(crossing),
                      'families': len(groups), 'eligible_new_families': result['new_parent_pools']['eligible_family_count'],
                      'parent_pool_status': result['new_parent_pools']['feasibility'], 'sha256': result['sha256']}))


if __name__ == '__main__':
    main()
