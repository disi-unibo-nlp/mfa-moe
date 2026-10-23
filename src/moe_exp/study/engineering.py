"""Label-blind matched engineering blocks and online delivered-dose validation.

Selection uses only original metadata. All routed layers are captured during
the actual intervention; native replay is never used as its substitute.
Canaries have separate identities and never enter scientific effect estimates.
"""
from __future__ import annotations

from pathlib import Path
import time
from moe_exp.io import BENCHMARKS, CLASSES, digest, publish, require, verified
from .cohort import assignments, identity, read


def select_blocks(rows: list[dict], seed: int) -> list[dict]:
    """Select eight unique-family blocks satisfying all frozen coverage constraints.

    Args:
        rows: Original model assignments; no effects or continuations are read.
        seed: Frozen purpose-specific hash ordering seed.

    Returns:
        Eight target assignments defining matched four-cell blocks.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    candidates = sorted([row for row in rows if row['arm'].startswith('target_')],
                        key=lambda row: digest([seed, 'engineering-blocks', row['uid']]))
    matrix, lower, upper = [np.ones(len(candidates))], [8], [8]
    for key, categories in (
        ('family', sorted({row['family'] for row in candidates})),
        ('class', CLASSES), ('benchmark', BENCHMARKS), ('fold', range(5)),
    ):
        for category in categories:
            matrix.append(np.array([row[key] == category for row in candidates], float))
            lower.append(0 if key == 'family' else 1)
            upper.append(1 if key == 'family' else 8)
    for regime in ('single', 'triple'):
        for direction in ('activate', 'deactivate'):
            matrix.append(np.array([row['layer_regime'] == regime and row['arm'] == 'target_'+direction
                                    for row in candidates], float))
            lower.append(2)
            upper.append(2)
    result = milp(np.arange(1, len(candidates)+1, dtype=float), integrality=np.ones(len(candidates)),
                  bounds=Bounds(0, 1), constraints=LinearConstraint(np.array(matrix), lower, upper),
                  options={'time_limit': 60, 'mip_rel_gap': 0})
    require(result.success, 'deterministic constrained engineering selection failed: '+result.message)
    selected = [row for row, flag in zip(candidates, result.x) if flag > .5]
    require(len(selected) == len({row['family'] for row in selected}) == 8, 'engineering families are not distinct')
    return selected


def plan(path: Path) -> dict:
    """Freeze 64 matched cells and separate non-scientific canaries.

    Args:
        path: Immutable cohort descriptor with source/runtime bindings.

    Returns:
        Engineering manifest derived only from metadata and frozen hashes.
    """
    from .rescue_source import load_study_binding
    value = read(path)
    study = load_study_binding(Path(value['study_binding_path']))
    records, canaries = [], []
    for model in ('gpt', 'qwen36'):
        original = assignments(value, model)
        selected = select_blocks(original, value['selection']['seed'])
        for block, target in enumerate(selected):
            matched = [target]
            matched.append(next(row for row in original if row['family'] == target['family']
                                and row['class'] == target['class'] and row['layer_regime'] == target['layer_regime']
                                and row['arm'] == target['arm'].replace('target', 'random')))
            matched.extend(next(row for row in original if row['family'] == target['family'] and row['arm'] == arm)
                           for arm in ('native', 'native_repeat'))
            for row in matched:
                cell = {**row, 'original_uid': row['uid'], 'block': block,
                        'engineering_only': True, 'capture_layers': study['models'][model]['layers']}
                cell['uid'] = digest([value['sha256'], 'engineering-cell', cell])
                records.append(cell)
        native = next(row for row in original if row['family'] == selected[0]['family'] and row['arm'] == 'native')
        action = selected[0]
        longest = max(original, key=lambda row: (row['source']['prefix_tokens'], row['uid']))
        for role, base, cap in (
            ('native-unhooked', native, 32), ('native-hooked', native, 32),
            ('zero-pulse', action, 32), ('pulse-boundary', action, 32), ('longest-prefix', longest, 512),
            ('abort', action, 32),
        ):
            row = {**base, 'engineering_role': role, 'engineering_only': True,
                   'capture_layers': study['models'][model]['layers'], 'horizon': cap,
                   'seed': native['seed'] if role in ('native-unhooked', 'native-hooked', 'zero-pulse') else base['seed']}
            if role in ('zero-pulse', 'pulse-boundary'):
                pulse = 0 if role == 'zero-pulse' else 8
                row['actions'] = [{**entry, 'pulse_tokens': pulse} for entry in base['actions']]
                row['pulse_tokens'] = pulse
            row['uid'] = digest([value['sha256'], 'engineering-canary', row])
            canaries.append(row)
    return publish(Path(value['paths']['control'])/'ENGINEERING.json', {
        'cohort_sha256': value['sha256'],
        'seed': value['selection']['seed'],
        'purpose': 'engineering-blocks',
        'effects_used': False,
        'records': records,
        'canaries': canaries,
    })


def telemetry(row: dict, capture: dict, tokens: int, profile: dict) -> dict:
    """Validate every online layer and summarize membership and weight displacement.

    Args:
        row: Frozen engineering cell.
        capture: Actual online routing capture.
        tokens: Actual generated token count.
        profile: Frozen model geometry.

    Returns:
        Per-layer dose evidence, without requiring changes for absent suppressed experts.
    """
    import numpy as np
    actions = {action['layer']: action for action in row.get('actions') or []}
    require(set(capture['capture']) == {str(layer) for layer in profile['layers']}, 'all-layer telemetry incomplete')
    result = []
    for layer in profile['layers']:
        data = capture['capture'][str(layer)]
        require(data['positions'] == list(range(tokens)), 'engineering capture has gaps')
        native_ids, executed_ids = np.array(data['native_ids']), np.array(data['executed_ids'])
        native_weights, weights = np.array(data['native_weights']), np.array(data['executed_weights'])
        require(native_ids.shape == executed_ids.shape == native_weights.shape == weights.shape
                == (tokens, profile['native_k']), 'native top-k shape differs')
        require(np.isfinite(weights).all() and (weights >= 0).all(), 'invalid routed mixture weights')
        require(all(len(set(ids)) == profile['native_k'] for ids in executed_ids.tolist()), 'expert cardinality differs')
        active = np.array(data['active'], bool)
        action = actions.get(layer)
        expected = np.array([bool(action and position < action['pulse_tokens'])
                             for position in data['positions']])
        # Reasoning may terminate within the window. Captured active flags must
        # be a leading interval, with no intervention after its declared pulse.
        require(not (active & ~expected).any(), 'intervention escaped pulse boundary')
        require(not any(active[index] and not active[index-1] for index in range(1, len(active))),
                'intervention restarted after reasoning termination')
        wanted = set(action['experts']) if action else set()
        if wanted:
            require(all(wanted <= set(ids) if action['direction'] == 'activate' else not wanted.intersection(ids)
                        for ids in executed_ids[active].tolist()), 'intended expert inclusion/exclusion failed')
        displacement = []
        for before_ids, after_ids, before_weights, after_weights in zip(native_ids, executed_ids, native_weights, weights):
            delta = np.zeros(profile['experts'])
            delta[before_ids] -= before_weights
            delta[after_ids] += after_weights
            displacement.append(float(np.abs(delta).sum()))
        result.append({
            'layer': layer,
            'active_tokens': int(active.sum()),
            'tokens': tokens,
            'weight_l1_mean': float(np.mean(displacement)),
            'native_target_mass': float(sum(weight for ids, ws in zip(native_ids, native_weights)
                                            for expert, weight in zip(ids, ws) if expert in wanted)/max(1, tokens)),
            'executed_target_mass': float(sum(weight for ids, ws in zip(executed_ids, weights)
                                              for expert, weight in zip(ids, ws) if expert in wanted)/max(1, tokens)),
        })
    return {'uid': row['uid'], 'layers': result, 'same_trajectory_local_native_comparator': True}


def _canary_run(llm, study, model, rows, store, deadline, *, hooked=True, capacity=1):
    """Run a separate canary roster with original seeds and durable publication."""
    from .driver import Engine, intervention_builder, intervention_finisher
    from .runtime import closure
    engine = Engine(llm, store, deadline, hooked=hooked, max_inflight=capacity, drain_seconds=60)
    return engine.run(rows, intervention_builder(study, model, llm.get_tokenizer(),
                      closure(llm.get_tokenizer(), model)), intervention_finisher(store, study, model))


def _mixed_order(rows: list[dict]) -> list[dict]:
    """Put eight distinct families and four arm types in the first mixed batch."""
    ordered = sorted(rows, key=lambda row: digest([20261020, 'mixed-engineering-order', row['uid']]))
    selected, families = [], set()
    for role in ('target', 'random', 'native', 'native_repeat')*2:
        row = next(row for row in ordered if row['family'] not in families
                   and (row['arm'] == role or role in ('target', 'random') and row['arm'].startswith(role+'_')))
        selected.append(row)
        families.add(row['family'])
    return selected+[row for row in ordered if row['uid'] not in {cell['uid'] for cell in selected}]


def _abort_probe(llm, study: dict, model: str, canaries: list[dict], store, deadline: float, root: Path) -> dict:
    """Exercise installed randomized request IDs and external-ID abort cleanup."""
    from .driver import Engine, intervention_builder, intervention_finisher
    from .runtime import closure
    abort = next(row for row in canaries if row['engineering_role'] == 'abort')
    probe = Engine(llm, store, deadline, hooked=True, max_inflight=1)
    if store.state(abort['uid']) is None:
        probe.admit(abort, intervention_builder(study, model, llm.get_tokenizer(), closure(llm.get_tokenizer(), model)))
        mapping = {uid: state['internal'] for uid, state in probe.inflight.items()}
        require(mapping[abort['uid']] != abort['uid'], 'installed internal request-ID mapping was not exercised')
        probe.step(intervention_finisher(store, study, model))
        probe.drain()
        publish(root/'ABORT.json', {'mapping': mapping})
    else:
        mapping = verified(root/'ABORT.json')['mapping']
    require(all(all(count == 0 for count in state.values()) for state in llm.collective_rpc('fs_state')),
            'abort cleanup retained request aliases or captures')
    return mapping


def _validate_cells(rows: list[dict], store, profile: dict) -> list[dict]:
    """Check every completed engineering cell and the actual pulse-end canary."""
    diagnostics = []
    for row in rows:
        if row.get('engineering_role') in ('native-unhooked', 'abort'):
            continue
        require(store.state(row['uid'])['status'] == 'complete', 'engineering cell lacks complete result')
        raw = verified(store.folder(row['uid'])/'OUTPUT.json')
        capture = verified(store.folder(row['uid'])/'CAPTURE.json')
        diagnostics.append(telemetry(row, capture, len(raw['token_ids']), profile))
        if row.get('engineering_role') == 'pulse-boundary':
            require(any(8 in data['positions'] and not data['active'][data['positions'].index(8)]
                        for data in capture['capture'].values()), 'pulse-end canary did not reach boundary')
    return diagnostics


def _matched_routing(rows: list[dict], store, profile: dict) -> list[dict]:
    """Compare actual all-layer trajectories with native and native-repeat variation."""
    import numpy as np
    diagnostics = []
    for block in range(8):
        cells = [row for row in rows if row['block'] == block]
        captures = {row['arm']: verified(store.folder(row['uid'])/'CAPTURE.json')['capture'] for row in cells}
        for layer in profile['layers']:
            vectors, memberships = {}, {}
            for arm, capture in captures.items():
                data = capture[str(layer)]
                ids = np.array(data['executed_ids'])
                weights = np.array(data['executed_weights'])
                vector = np.zeros((len(ids), profile['experts']))
                vector[np.arange(len(ids))[:, None], ids] = weights
                vectors[arm] = vector
                membership = np.zeros_like(vector, dtype=bool)
                membership[np.arange(len(ids))[:, None], ids] = True
                memberships[arm] = membership
            length = min(map(len, vectors.values()))
            native = vectors['native'][:length]
            for arm, vector in vectors.items():
                if arm == 'native':
                    continue
                observed = vector[:length]
                diagnostics.append({
                    'block': block,
                    'layer': layer,
                    'arm': arm,
                    'compared_positions': length,
                    'weight_l1_vs_native': float(np.abs(observed-native).sum(1).mean()),
                    'membership_symmetric_difference_vs_native': float((memberships[arm][:length] !=
                                                                         memberships['native'][:length]).sum(1).mean()),
                })
    return diagnostics


def _fidelity_canaries(llm, study: dict, model: str, rows: list[dict], canaries: list[dict], store, deadline: float, root: Path) -> dict:
    """Check unhooked/hooked equality, zero pulse, longest prefix and abort cleanup."""
    from .driver import contract_for
    unhooked = next(row for row in canaries if row['engineering_role'] == 'native-unhooked')
    _canary_run(llm, study, model, [unhooked], store, deadline, hooked=False)
    contract = contract_for(study, model, rows+canaries)
    reports = llm.collective_rpc('fs_begin', kwargs={'contract': contract})
    require(len(reports) == 2, 'engineering needs two consistent TP ranks')
    mapping = _abort_probe(llm, study, model, canaries, store, deadline, root)
    serial = [row for row in canaries if row['engineering_role'] in ('native-hooked', 'zero-pulse')]
    for row in serial:
        _canary_run(llm, study, model, [row], store, deadline)
    original = verified(store.folder(unhooked['uid'])/'OUTPUT.json')['token_ids']
    require(all(verified(store.folder(row['uid'])/'OUTPUT.json')['token_ids'] == original for row in serial),
            'same-seed batch-one native/hooked/zero-pulse outputs differ')
    remaining = [row for row in canaries if row['engineering_role'] in ('pulse-boundary', 'longest-prefix')]
    for row in remaining:
        _canary_run(llm, study, model, [row], store, deadline)
    return mapping


def run(path: Path, model: str) -> dict:
    """Qualify both fidelity and all-layer dose in one approved model allocation.

    Args:
        path: Immutable cohort descriptor.
        model: GPT or Qwen model allocated two GPUs for at most two hours.

    Returns:
        QUALIFIED only after every frozen cell, canary and abort check passes.
    """
    from .driver import Store, deadline_from_slurm, hooked_llm
    from .ownership import check_time, worker
    from .rescue_source import load_study_binding
    value = read(path)
    manifest = verified(Path(value['paths']['control'])/'ENGINEERING.json')
    require(manifest['cohort_sha256'] == value['sha256'], 'engineering cohort differs')
    saved = verified(Path(value['paths']['qualification']))
    require(saved['cohort_sha256'] == value['sha256'], 'saved qualification differs')
    rows = [row for row in manifest['records'] if row['model'] == model]
    canaries = [row for row in manifest['canaries'] if row['model'] == model]
    execution = identity(value, model, 'engineering', [row['uid'] for row in rows+canaries])
    root = Path(value['paths']['bulk'])/'engineering'/model
    deadline = min(check_time(value), deadline_from_slurm())
    with worker(root, execution):
        require(time.time()+900 < deadline and rows, 'empty or expired engineering work before model loading')
        study = load_study_binding(Path(value['study_binding_path']))
        store = Store(root/'units')
        require(all((store.state(row['uid']) or {}).get('status') not in ('partial', 'output')
                    for row in rows+canaries if row.get('engineering_role') != 'abort'),
                'engineering contains an interrupted admitted request; no redraw is authorized')
        llm, kwargs, load_seconds = hooked_llm(study, model, 8)
        try:
            mapping = _fidelity_canaries(llm, study, model, rows, canaries, store, deadline, root)
            # Eight simultaneous requests mix families, arms and layer regimes.
            ordered = _mixed_order(rows)
            metrics = _canary_run(llm, study, model, ordered, store, deadline, capacity=8)
            require(metrics['remaining_queue'] == 0 and metrics['drained'] == 0, 'engineering unfinished')
            require(metrics['peak_inflight'] == 8 and metrics['mixed_eight_verified'], 'eight-request mixed concurrency was not exercised')
            validation = llm.collective_rpc('fs_validation')
            require(len(validation) == 2 and digest(validation[0]) == digest(validation[1]), 'TP validation differs')
            require(set(map(int, validation[0])) == set(study['models'][model]['layers']), 'native scaling validation incomplete')
            diagnostics = _validate_cells(rows+canaries, store, study['models'][model])
            matched = _matched_routing(rows, store, study['models'][model])
            llm.collective_rpc('fs_end')
        finally:
            llm.llm_engine.engine_core.shutdown()
        read(path)
        return publish(Path(value['paths']['control'])/'engineering'/model/'QUALIFIED.json', {
            'status': 'complete',
            'cohort_sha256': value['sha256'],
            'manifest_sha256': manifest['sha256'],
            'model': model,
            'cells': len(rows),
            'load_seconds': load_seconds,
            'metrics': metrics,
            'native_validation': validation,
            'dose_diagnostics': diagnostics,
            'canaries_excluded_from_science': True,
            'abort_internal_mapping': mapping,
            'matched_actual_trajectory_routing': matched,
            'trajectory_alignment': 'continuation positions; text may diverge, so downstream differences are descriptive',
        })
