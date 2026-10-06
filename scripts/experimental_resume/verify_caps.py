"""Bounded dry enumeration from the existing eligibility cache; no model or job launch."""
import hashlib
import json
from pathlib import Path

import x2_build as X
from moe_steer import manifests as M, policies


def key(r):
    return r['question'], r['arm'], r['policy_name'], r['seed_k']


def main():
    campaign = policies.CAMPAIGN
    out = campaign / 'runs/s2prop'
    inventory = json.loads((campaign / 'addenda/s2/PROPOSED_TREE.json').read_text())
    world, split = M.load_world(), M.load_split()
    rows = json.loads((out / 'eligibility.json').read_text())['rows']
    manifest, cells = X.build_x2_manifest(world, rows, name='x2-dryrun-s2', n_shards=1,
                                        max_new_tokens=1024, n_policy='sham', code_tree=inventory['proposal_tree'])
    evidence = X.check_manifest(manifest, rows, cells, split=split, infos=world.infos,
                               n_policy='sham', max_new_tokens=1024,
                               expect_tree=inventory['proposal_tree'])
    path = out / 'x2-dryrun-s2.json'
    M.write_manifest(path, manifest)
    assert M.load_manifest(path) == manifest
    assert len(manifest['requests']) == 6630 and len(manifest['questions']) == 39
    assert all(r['max_tokens'] == 1024 for r in manifest['requests'])
    original = M.load_manifest(campaign / 'runs/x2prep/x2-dryrun-shamN.json')
    old = {key(r): r for r in original['requests']}
    assert set(old) == {key(r) for r in manifest['requests']}
    different = {}
    for request in manifest['requests']:
        previous = old[key(request)]
        fields = [k for k in request if request[k] != previous[k]]
        for field in fields:
            different[field] = different.get(field, 0) + 1
        # Name is a UID salt, and sharding/order depend on the UID. Validate the induced
        # UID change explicitly instead of pretending these unequal names preserve UIDs.
        assert M.request_uid(original['name'], request) == previous['uid']
        assert set(fields) <= {'max_tokens', 'uid', 'shard', 'order'}
    # Same-name parity removes that necessary name/UID difference altogether.
    parity, _ = X.build_x2_manifest(world, rows, name=original['name'],
                                   n_shards=original['n_shards'], max_new_tokens=1024, n_policy='sham',
                                   code_tree=inventory['proposal_tree'])
    old_by_uid = {r['uid']: r for r in original['requests']}
    for request in parity['requests']:
        prior = old_by_uid[request['uid']]
        assert {k: v for k, v in request.items() if k not in {'max_tokens', 'order', 'shard'}} == {
            k: v for k, v in prior.items() if k not in {'max_tokens', 'order', 'shard'}}
    other_fields = [k for k in parity if parity[k] != original[k] and k != 'requests']
    assert set(other_fields) <= {'sha256', 'code_tree', 'routed'}
    result = {'status': 'PASS', 'tree': inventory['proposal_tree'], 'manifest_path': str(path),
              'manifest_sha256': manifest['sha256'],
              'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'requests': len(manifest['requests']), 'questions': len(manifest['questions']),
              'policies': len(M.manifest_table(manifest).policies),
              'decode_tokens_cap': sum(r['max_tokens'] for r in manifest['requests']),
              'different_name_request_differences': different,
              'same_name_parity': 'all request fields equal except max_tokens and assignment',
              'same_name_manifest_differences': other_fields,
              'routed': manifest['routed'],
              'routed_ids_raw_bytes_per_request': 1024 * 30 * 8,
              'routed_ids_raw_bytes_total': 6630 * 1024 * 30 * 8,
              'assertions': evidence}
    (out / 'dry-verification.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'assertions'}, indent=1))


if __name__ == '__main__':
    main()
