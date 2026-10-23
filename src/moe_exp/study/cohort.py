"""Immutable rescue population and per-assignment evidence qualification.

The source receipt precedes this descriptor. Execution identities depend on the
descriptor and a fixed authorized roster, never on unfinished work. Historical
outputs remain in their original stores and retain their provenance limits.
Run through ``python -m moe_exp.study.rescue cohort`` before allocated validation.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from moe_exp.io import CLASSES, HOME, WORK, check_files, digest, file_sha, publish, require, verified

MODELS = ('gpt', 'qwen36')
WINDOW = 512
SELECTION_SEED = 20261020
SCIENCE_CUTOFF = '2026-10-23T05:30:00+02:00'
REPORT_DEADLINE = '2026-10-23T11:30:00+02:00'


def writable(path: Path) -> Path:
    """Resolve an explicitly user-owned HOME/WORK destination."""
    path = path.resolve()
    require(any(path.is_relative_to(root) for root in (HOME, WORK)),
            f'cohort writes require user HOME/WORK storage: {path}')
    return path


def _historical_population(original: dict) -> tuple[list[dict], dict]:
    """Freeze historical metadata without qualifying captures or relabeling outputs."""
    from .driver import DRIVER, STORES, bulk
    from .rescue_reader import generation_matches, generation_release
    records, manifests = [], {}
    for model in MODELS:
        manifest = verified(DRIVER/model/'RESCUE.json')
        manifests[model] = {'path': str(DRIVER/model/'RESCUE.json'), 'sha256': manifest['sha256']}
        historical_units = bulk(original)/model/STORES['rescue']
        jobs = set(generation_release(model)['jobs'])
        for assignment in manifest['records']:
            folder = historical_units/assignment['uid']
            require(not (folder/'PARTIAL.json').exists() or (folder/'RESULT.json').exists(),
                    'historical partial requires explicit unknown adjudication')
            if not folder.exists() or not any(folder.iterdir()):
                disposition = 'authorized_generation'
                require(model == 'qwen36', 'only untouched Qwen assignments are authorized')
                evidence = {}
            else:
                raw, result = verified(folder/'OUTPUT.json'), verified(folder/'RESULT.json')
                require(raw['assignment_sha256'] == result['assignment_sha256'] == digest(assignment),
                        'historical assignment join changed')
                disposition = ('historical_candidate' if result['status'] == 'complete'
                               and generation_matches(assignment, raw, jobs) else 'incompatible_unknown')
                evidence = {'output_sha256': raw['sha256'], 'result_sha256': result['sha256']}
            records.append({
                'assignment': assignment,
                'disposition': disposition,
                'historical_folder': str(folder),
                'historical_jobs': sorted(jobs),
                'historical_evidence': evidence,
            })
    return records, manifests


def _annotation_indexes(study: dict) -> dict:
    """Bind original index bytes to their original metadata receipts."""
    files = {}
    for model in MODELS:
        index = Path(study['source_campaign'])/model/'LABEL_INDEX.sqlite'
        metadata = index.with_suffix('.json')
        if not index.exists():
            continue
        expected = verified(metadata)['sha256_file']
        require(file_sha(index) == expected, 'original label index bytes differ from its receipt')
        files.update({str(index): expected, str(metadata): file_sha(metadata)})
    return files


def _destinations(root: Path, control: Path) -> dict:
    """Keep every writable stage beneath its explicit cohort WORK/control roots."""
    paths = {
        'bulk': str(root),
        'control': str(control),
        'generation_index': str(control/'GENERATION_INDEX.json'),
        'qualification': str(control/'SAVED_QUALIFICATION.json'),
        'extraction': str(root/'extraction'),
        'reader': str(root/'reader'),
        'analysis': str(root/'analysis'),
    }
    return paths


def create(source_path: Path, destination: Path, classifier: Path, *, predecessor_path: Path | None = None) -> dict:
    """Freeze the original population and the authorized untouched-generation list.

    Args:
        source_path: Newly sealed source receipt with explicit fresh study roots.
        destination: New immutable cohort descriptor.
        classifier: Deployed C10 directory containing READOUT and all five folds.
        predecessor_path: Explicit terminal release whose allocations remain charged.

    Returns:
        Descriptor retaining every assignment and its original evidence reference.
    """
    from .lexical import artifact_binding
    from .rescue_reader import frozen_campaign
    from .rescue_source import STUDY_BINDING, REPORT, load_study_binding, verify

    source = verify(source_path)
    study = load_study_binding(Path(source['study_binding_path']))
    original = verified(STUDY_BINDING)
    root = writable(Path(study['config']['bulk_root']))
    control = writable(destination.parent)
    require(root != Path(original['config']['bulk_root']).resolve(), 'cohort needs a fresh generation store')
    records, manifests = _historical_population(original)
    counts = dict(Counter(row['disposition'] for row in records))
    require(counts == {'historical_candidate': 5380, 'incompatible_unknown': 37,
                      'authorized_generation': 2007}, f'saved rescue population differs from the approved audit: {counts}')
    campaign = frozen_campaign(study)
    paths = _destinations(root, control)
    from .reservations import freeze_predecessor
    predecessor = freeze_predecessor(predecessor_path, records, root) if predecessor_path is not None else None
    return publish(destination, {
        'schema': 'causal-rescue-cohort-v1',
        'source': {'path': str(source_path.resolve()), 'sha256': source['sha256']},
        'study_binding_path': source['study_binding_path'],
        'manifests': manifests,
        'historical_audit': {'path': str(REPORT/'GENERATION_PROVENANCE_AUDIT.json'),
                             'sha256': file_sha(REPORT/'GENERATION_PROVENANCE_AUDIT.json')},
        'historical_files': {str(REPORT/name): file_sha(REPORT/name) for name in
                             ('GENERATION_PROVENANCE_AUDIT.json', 'SOURCE_BINDING-r8.json', 'SOURCE-r8.tar.gz', 'ATTEMPTS.json')},
        'historical_limitations': 'r8 receipt did not seal the full execution closure; new qualification does not retroactively supply it',
        'generation_uids': sorted(row['assignment']['uid'] for row in records
                                  if row['disposition'] == 'authorized_generation'),
        'unknown_uids': sorted(row['assignment']['uid'] for row in records
                               if row['disposition'] == 'incompatible_unknown'),
        'paths': paths,
        'classifier': artifact_binding(classifier),
        'annotator': campaign['annotator'],
        'reader_runtime_files': campaign['reader_runtime_files'],
        'reuse_indexes': _annotation_indexes(study),
        'class_mapping': list(CLASSES),
        'selection': {'seed': SELECTION_SEED, 'engineering_purpose': 'engineering-blocks',
                      'audit_purpose': 'trajectory-occurrences', 'audit_per_stratum': 32},
        'budget': {'gpu_hours': 64, 'cpu_core_hours': 16, 'protected_analysis_core_hours': 4},
        'predecessor': predecessor,
        'science_cutoff': SCIENCE_CUTOFF,
        'report_deadline': REPORT_DEADLINE,
        'counts': counts,
        'records': records,
    })


def read(path: Path, *, verify_source: bool = True) -> dict:
    """Verify a cohort without deriving identity from mutable execution state.

    Args:
        path: Explicit immutable descriptor.
        verify_source: Also compare frozen code and installed runtime.

    Returns:
        Validated descriptor with the complete planned population.
    """
    value = verified(path)
    require(value['schema'] == 'causal-rescue-cohort-v1', 'unsupported cohort schema')
    records = value['records']
    assigned = [row['assignment']['uid'] for row in records]
    require(len(set(assigned)) == len(assigned), 'duplicate cohort UID')
    require(value['generation_uids'] == sorted(row['assignment']['uid'] for row in records
            if row['disposition'] == 'authorized_generation'), 'fixed generation roster differs')
    require(value['unknown_uids'] == sorted(row['assignment']['uid'] for row in records
            if row['disposition'] == 'incompatible_unknown'), 'fixed unknown roster differs')
    require(value['class_mapping'] == list(CLASSES), 'reader class mapping differs')
    expected = {}
    for manifest in value['manifests'].values():
        original = verified(Path(manifest['path']))
        require(original['sha256'] == manifest['sha256'], 'original manifest changed')
        expected.update({row['uid']: digest(row) for row in original['records']})
    require({row['assignment']['uid']: digest(row['assignment']) for row in records} == expected,
            'cohort assignments differ from original manifest science')
    check_files(value['classifier']['files'])
    check_files(value['historical_files'])
    check_files(value['reader_runtime_files'])
    check_files(value['reuse_indexes'])
    if value.get('predecessor'):
        previous = value['predecessor']
        check_files(previous['files'])
        require(verified(Path(previous['path']))['sha256'] == previous['sha256'], 'predecessor cohort differs')
    require(file_sha(Path(value['annotator']['program'])) == value['annotator']['config']['program_sha256'],
            'frozen reader program bytes changed')
    for name in ('bulk', 'control', 'extraction', 'reader', 'analysis'):
        writable(Path(value['paths'][name]))
    if verify_source:
        from .rescue_source import verify
        source = verify(Path(value['source']['path']))
        require(source['sha256'] == value['source']['sha256'], 'cohort source differs')
        require(source['study_binding_path'] == value['study_binding_path'], 'cohort storage binding differs')
        study = verified(Path(value['study_binding_path']))
        require(study['config']['bulk_root'] == value['paths']['bulk']
                and study['control_root'] == value['paths']['control'], 'cohort roots differ from source binding')
        for manifest in value['manifests'].values():
            require(source['files'].get(manifest['path']) == file_sha(Path(manifest['path'])),
                    'cohort manifest is outside the frozen source closure')
    return value


def assignments(value: dict, model: str | None = None) -> list[dict]:
    """Return original scientific rows, including every unknown assignment.

    Args:
        value: Verified cohort descriptor.
        model: Optional model restriction.

    Returns:
        Original rows with unchanged UIDs, seeds and intervention parameters.
    """
    return [row['assignment'] for row in value['records']
            if model is None or row['assignment']['model'] == model]


def identity(value: dict, model: str, stage: str, uids: list[str]) -> dict:
    """Bind execution to a frozen roster, independent of future completions.

    Args:
        value: Verified immutable cohort.
        model: Stage model.
        stage: Generation, engineering or annotation.
        uids: Full frozen roster for this store, never a pending subset.

    Returns:
        Stable execution identity.
    """
    return {
        'schema': 'causal-rescue-execution-v1',
        'cohort_sha256': value['sha256'],
        'source_sha256': value['source']['sha256'],
        'model': model,
        'stage': stage,
        'roster_sha256': digest(sorted(uids)),
    }


def qualify_capture(assignment: dict, folder: Path, jobs: set[str], *, profile: dict | None = None) -> dict:
    """Verify OUTPUT/RESULT/CAPTURE and every scientific assignment join.

    Args:
        assignment: Original assignment.
        folder: Actual historical or new unit directory.
        jobs: Permitted positive-source job owners.
        profile: Frozen model geometry for full capture validation.

    Returns:
        References to verified evidence, or raises for per-assignment adjudication.
    """
    from .rescue_reader import generation_matches
    raw = verified(folder/'OUTPUT.json')
    result = verified(folder/'RESULT.json')
    capture = verified(folder/'CAPTURE.json')
    uid = assignment['uid']
    require(raw['uid'] == result['uid'] == capture['uid'] == uid, 'capture UID mismatch')
    require(result['status'] == 'complete', 'generation result is not complete')
    require(raw['assignment_sha256'] == result['assignment_sha256'] == digest(assignment), 'assignment mismatch')
    require(raw['sha256'] == result['raw_sha256'] == capture['raw_sha256'], 'raw/capture/result mismatch')
    require(result['generated_tokens'] == len(raw['token_ids']), 'generated token count mismatch')
    require(result['segments'] == len(raw['segments']) and result['finish_reason'] == raw['finish_reason'],
            'generation result segment or termination join differs')
    require(raw['profile_sha256'] == assignment['profile_sha256'], 'generation profile mismatch')
    require(all(type(token) is int and token >= 0 for token in raw['token_ids']), 'invalid saved token IDs')
    require(generation_matches(assignment, raw, jobs), 'source/seed/window mismatch')
    require(raw['finish_reason'] in ('stop', 'length'), 'unqualified termination')
    require(capture['stats']['target_contract'] and capture['stats']['noop_exact'], 'routing contract failed')
    count = len(raw['token_ids'])
    for layer in assignment['capture_layers']:
        data = capture['capture'][str(layer)]
        require(data['positions'] == list(range(count)), 'capture positions do not cover the saved tokens')
        require(all(len(data[name]) == count for name in
                    ('native_ids', 'native_weights', 'executed_ids', 'executed_weights', 'active')),
                'capture arrays differ from token count')
        import math
        for ids_key, weights_key in (('native_ids', 'native_weights'), ('executed_ids', 'executed_weights')):
            for ids, weights in zip(data[ids_key], data[weights_key]):
                require(len(ids) == len(weights) == len(set(ids)), 'capture expert cardinality differs')
                require(all(math.isfinite(weight) and weight >= 0 for weight in weights), 'invalid captured weights')
                if profile is not None:
                    require(len(ids) == profile['native_k'] and all(type(expert) is int and 0 <= expert < profile['experts']
                            for expert in ids), 'capture native expert geometry differs')
        action = next((entry for entry in assignment.get('actions') or [] if entry['layer'] == layer), None)
        for position, active, ids in zip(data['positions'], data['active'], data['executed_ids']):
            require(not active or action and position < action['pulse_tokens'], 'capture active pulse differs')
            if active:
                wanted = set(action['experts'])
                require(wanted <= set(ids) if action['direction'] == 'activate' else not wanted.intersection(ids),
                        'capture intended membership differs')
    return {'output_sha256': raw['sha256'], 'result_sha256': result['sha256'],
            'capture_sha256': capture['sha256'], 'folder': str(folder)}


def _replay_model(arguments: tuple) -> tuple[list[dict], list[dict]]:
    """Qualify one model in an allocated CPU process, caching each frozen prefix."""
    import json
    from transformers import AutoTokenizer
    from .outcomes import record
    from .runtime import closure, prompt_prefix
    from moe_exp.models.token_replay import validate_token_replay
    from .rescue_source import load_study_binding
    value, model, candidates = arguments
    study = load_study_binding(Path(value['study_binding_path']))
    profile = study['models'][model]
    check_files(profile['config_files'])
    tokenizer = AutoTokenizer.from_pretrained(profile['local_path'], local_files_only=True)
    close = closure(tokenizer, model)
    inputs, results, replayed = {}, [], []
    for candidate in candidates:
        assignment = candidate['assignment']
        uid = assignment['uid']
        folder = Path(candidate['folder'])
        phase = 'metadata'
        try:
            require(assignment['profile_sha256'] == profile['profile_sha256'], 'frozen model profile differs')
            evidence = qualify_capture(assignment, folder, set(candidate['jobs']), profile=profile)
            expected = candidate.get('expected', {})
            require(all(evidence[key] == expected[key] for key in expected), 'historical evidence changed')
            phase = 'replay'
            key = digest(assignment['source'])
            if key not in inputs:
                inputs[key] = prompt_prefix(study, model, assignment, tokenizer)
                original = inputs[key][2]
                validate_token_replay(original['metadata']['token_replay'], tokenizer, original['cot_text'])
            raw = verified(folder/'OUTPUT.json')
            rebuilt = record(study, model, assignment, raw, tokenizer, close, frozen_input=inputs[key])
            require(rebuilt['generated_tokens'] == len(raw['token_ids']), 'replay token count differs')
            require(all(0 <= sentence['token_start'] < sentence['token_end'] <= len(raw['token_ids'])
                        for sentence in rebuilt['sentences']), 'replayed sentence bounds differ')
            results.append({'uid': uid, 'status': 'eligible', **evidence, 'replay_sha256': digest(rebuilt)})
            replayed.append({'uid': uid, 'replay_json': json.dumps(rebuilt)})
        except (ValueError, KeyError, OSError, TypeError) as error:
            if phase == 'replay' and not isinstance(error, ValueError):
                raise
            results.append({'uid': uid, 'status': 'qualification_unknown',
                            'error': f'{type(error).__name__}: {error}', 'folder': str(folder)})
    return results, replayed


def qualify_saved(path: Path) -> dict:
    """Replay every saved candidate on allocated CPUs, retaining assignment failures.

    Args:
        path: Explicit immutable cohort descriptor.

    Returns:
        Full qualification index and hashed replay tables; failures stay unknown.
    """
    import os
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    value = read(path)
    require(os.environ.get('SLURM_JOB_ID'), 'saved replay requires an allocated CPU job')
    arguments = []
    for model in MODELS:
        candidates = [{
            'assignment': row['assignment'],
            'folder': row['historical_folder'],
            'jobs': row['historical_jobs'],
            'expected': row['historical_evidence'],
        } for row in value['records'] if row['assignment']['model'] == model
            and row['disposition'] == 'historical_candidate']
        # Split by family so each process decodes a source prefix only once.
        families = sorted({row['assignment']['family'] for row in candidates})
        for part in range(2):
            selected = set(families[part::2])
            arguments.append((value, model, [row for row in candidates if row['assignment']['family'] in selected]))
    records, replay_rows = [], {model: [] for model in MODELS}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for argument, (qualified, replayed) in zip(arguments, pool.map(_replay_model, arguments)):
            records.extend(qualified)
            replay_rows[argument[1]].extend(replayed)
    files = {}
    for model, rows in replay_rows.items():
        destination = Path(value['paths']['extraction'])/model/'saved-replay.parquet'
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        require(not destination.exists(), 'saved replay table already exists; preserve previous qualification')
        pd.DataFrame(rows, columns=['uid', 'replay_json']).to_parquet(destination, index=False)
        files[str(destination)] = file_sha(destination)
    read(path)
    return publish(Path(value['paths']['qualification']), {
        'cohort_sha256': value['sha256'],
        'records': sorted(records, key=lambda row: row['uid']),
        'counts': dict(Counter(row['status'] for row in records)),
        'replay_files': files,
    })


def terminal_index(path: Path) -> dict:
    """Publish every disposition only after all generation owners are quiescent.

    Args:
        path: Explicit immutable cohort descriptor.

    Returns:
        Immutable full population index; unknowns do not prohibit reader admission.
    """
    from contextlib import ExitStack
    from .ownership import TERMINAL_STATES, accounting, store_lock
    value = read(path)
    qualified = verified(Path(value['paths']['qualification']))
    require(qualified['cohort_sha256'] == value['sha256'], 'saved qualification belongs to another cohort')
    check_files(qualified['replay_files'])
    historical = {row['uid']: row for row in qualified['records']}
    generated_root = Path(value['paths']['bulk'])/'driver'/'qwen36'/'rescue'
    with ExitStack() as locks:
        locks.enter_context(store_lock(generated_root))
        owners = [verified(owner) for owner in sorted((generated_root/'owners').glob('*.json'))]
        terminal = [accounting(owner['job']) for owner in owners]
        require(all(row['state'] in TERMINAL_STATES for row in terminal), 'generation index requires quiescent owners')
        records = []
        extraction = {model: verified(Path(value['paths']['extraction'])/model/'EXTRACTION.json') for model in MODELS}
        for receipt in extraction.values():
            require(receipt['cohort_sha256'] == value['sha256'], 'extraction belongs to another cohort')
            check_files(receipt['files'])
        generated = {row['uid']: row for receipt in extraction.values() for row in receipt['new_qualification']}
        require(set(generated) == set(value['generation_uids']), 'new qualification does not cover the fixed generation roster')
        for row in value['records']:
            assignment, disposition = row['assignment'], row['disposition']
            uid = assignment['uid']
            if disposition == 'historical_candidate':
                evidence = historical[uid]
            elif disposition == 'incompatible_unknown':
                evidence = {'uid': uid, 'status': 'incompatible_unknown', 'folder': row['historical_folder']}
            else:
                evidence = generated[uid]
            records.append({**evidence, 'origin': disposition})
        result = publish(Path(value['paths']['generation_index']), {
            'cohort_sha256': value['sha256'],
            'quiescent': True,
            'owners': owners,
            'terminal_accounting': terminal,
            'saved_qualification_sha256': qualified['sha256'],
            'extraction_sha256': {model: receipt['sha256'] for model, receipt in extraction.items()},
            'records': records,
            'counts': dict(Counter(row['status'] for row in records)),
        })
        publish(generated_root/'SEALED.json', {'cohort_sha256': value['sha256'], 'generation_index_sha256': result['sha256']})
        return result


def generation_index(value: dict) -> dict:
    """Validate complete terminal-index coverage and exact frozen output references.

    Args:
        value: Verified cohort descriptor.

    Returns:
        Terminal index, including all unsuccessful assignments.
    """
    index = verified(Path(value['paths']['generation_index']))
    require(index['cohort_sha256'] == value['sha256'] and index['quiescent'], 'invalid terminal generation index')
    require(sorted(row['uid'] for row in index['records']) == sorted(row['uid'] for row in assignments(value)),
            'terminal index does not cover the complete assignment population')
    return index


def extract_terminal(path: Path, model: str) -> dict:
    """Extract the terminal cohort into a fresh model-specific replay table.

    Args:
        path: Required immutable cohort descriptor.
        model: GPT or Qwen3.6 model.

    Returns:
        Verified extraction receipt; failures remain in the final population.
    """
    import pandas as pd
    from .ownership import TERMINAL_STATES, accounting, store_lock
    value = read(path)
    root = Path(value['paths']['bulk'])/'driver'/'qwen36'/'rescue'
    with store_lock(root):
        owners = [verified(owner) for owner in (root/'owners').glob('*.json')]
        require(all(accounting(owner['job'])['state'] in TERMINAL_STATES for owner in owners),
                'saved extraction requires quiescent generation owners')
        candidates = [{
            'assignment': row,
            'folder': str(root/'units'/row['uid']),
            'jobs': [owner['job'] for owner in owners],
        } for row in assignments(value, model) if row['uid'] in set(value['generation_uids'])]
        records, replayed = _replay_model((value, model, candidates)) if candidates else ([], [])
    saved = Path(value['paths']['extraction'])/model/'saved-replay.parquet'
    qualification = verified(Path(value['paths']['qualification']))
    check_files(qualification['replay_files'])
    table = pd.concat([pd.read_parquet(saved), pd.DataFrame(replayed, columns=['uid', 'replay_json'])], ignore_index=True)
    require(table.uid.is_unique, 'duplicate extracted replay UID')
    destination = saved.parent/'terminal-replay.parquet'
    require(not destination.exists(), 'terminal extraction requires a fresh destination')
    table.to_parquet(destination, index=False)
    return publish(saved.parent/'EXTRACTION.json', {
        'cohort_sha256': value['sha256'],
        'model': model,
        'files': {str(destination): file_sha(destination)},
        'new_qualification': records,
        'extracted': len(table),
    })
