"""Engineering-only replay evidence; unknown native routes stay unknown."""


def snapshot(completion):
    import numpy as np
    return {'tokens': list(completion.token_ids), 'routes': None if completion.routed_experts is None
            else np.array(completion.routed_experts, copy=True)}


def compare(left, right):
    import numpy as np
    a, b = left['tokens'], right['tokens']
    result = {'tokens_equal': a == b, 'token_lengths': [len(a), len(b)],
        'first_token_difference': next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
            min(len(a), len(b)) if len(a) != len(b) else None),
        'routes_equal': None, 'route_sets_equal': None, 'route_shapes': None,
        'first_route_difference': None}
    a, b = left['routes'], right['routes']
    if a is None or b is None:
        result['routes_status'] = 'UNKNOWN'; return result
    result['route_shapes'] = [list(a.shape), list(b.shape)]
    result['routes_status'] = 'CAPTURED_NATIVE'
    result['routes_equal'] = bool(np.array_equal(a, b))
    if a.shape != b.shape:
        result['route_sets_equal'] = False; return result
    result['route_sets_equal'] = bool(np.array_equal(np.sort(a, axis=-1), np.sort(b, axis=-1)))
    changed = np.argwhere(a != b)
    result['differing_expert_slots'] = len(changed)
    if len(changed):
        position = tuple(int(i) for i in changed[0])
        result['first_route_difference'] = {'position': list(position), 'left': int(a[position]), 'right': int(b[position])}
    return result


def save_evidence(allocation, fixture_ids, results, captures):
    import numpy as np
    import native_finalization_v1 as N
    paths = {}
    for name, result in results.items():
        path = allocation / ('ENGINEERING_' + name + '.npz')
        with path.open('xb') as stream:
            np.savez_compressed(stream, token_ids=np.array(result['tokens'], dtype=np.int64),
                **({'native_ids': result['routes']} if result['routes'] is not None else {}))
        paths[name] = {'path': str(path), 'file_sha256': N.U.file_sha(path), 'tokens': result['tokens'],
                       'native_routes': 'UNKNOWN' if result['routes'] is None else 'CAPTURED'}
    return N.save(allocation / 'ENGINEERING_REPLAY.json', {
        'schema': 'native-finalization-engineering-replay-v3', 'excluded_from_outcomes': True,
        'manifest_sha256': N.U.sealed(N.DOC / 'MANIFEST.json')['sha256'],
        'fixture_prompt_ids': fixture_ids, 'files': paths, 'captures': captures,
        'native_repeat': compare(results['native_reference'], results['native_repeat']),
        'instrumented_repeat': compare(results['native_reference'], results['instrumented']),
        'claim_limit': 'Engineering replay evidence only; no scientific grading or effect estimate.'})
