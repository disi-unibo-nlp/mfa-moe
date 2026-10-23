"""Prepare the frozen reader from the driver's saved rescue continuations.

The inventory retains every assigned continuation, including incomplete outputs
and continuations without a complete new sentence. Reader tasks use the same
token replay and boundary handling as outcome extraction.

Run ``python -m moe_exp.study.rescue_reader plan --model MODEL`` in an approved
CPU allocation, then ``run`` in an approved GPU allocation.
"""
from __future__ import annotations

from pathlib import Path
from moe_exp.io import REPO, check_files, digest, file_sha, load, publish, require, verified
from .driver import DRIVER, STORES, binding, bulk


TASK_FILE = 'RESCUE_READER_TASKS-r1.json'


def generation_release(model: str) -> dict:
    """Read the preserved generation receipt and its observed job roster."""
    from .rescue_source import RECEIPT, REPORT

    source = verified(RECEIPT)
    attempts = load(REPORT/'ATTEMPTS.json')
    jobs = {
        job
        for attempt in attempts['attempts']
        if attempt.get('source_binding') == RECEIPT.name
        for job in attempt.get('jobs_by_model', {}).get(model, [])
    }
    require(jobs, 'rescue generation receipt has no recorded jobs')
    return {'source_sha256': source['sha256'], 'jobs': sorted(jobs)}


def generation_matches(row: dict, raw: dict, jobs: set[str]) -> bool:
    """Accept uninterrupted current-source tokens, allowing earlier empty admissions."""
    segments = raw.get('segments', [])
    return (
        raw.get('total_allowance') == row['horizon']
        and len(raw['token_ids']) <= row['horizon']
        and bool(segments)
        and all(segment.get('generated', 0) == 0 and segment.get('history') == 0
                for segment in segments[:-1])
        and segments[-1].get('job') in jobs
        and segments[-1].get('history') == 0
        and segments[-1].get('max_tokens') == row['horizon']
        and segments[-1].get('seed') == row['seed']
    )


def context_sources() -> dict[str, str]:
    """Bind extraction and reader execution independently of completed generation."""
    paths = (
        'src/moe_exp/study/rescue_reader.py',
        'src/moe_exp/study/outcomes.py',
        'src/moe_exp/study/contexts.py',
        'src/moe_exp/study/driver.py',
        'src/moe_exp/study/common.py',
        'src/moe_exp/study/runtime.py',
        'src/moe_exp/study/rescue_source.py',
        'src/moe_exp/study/reader.py',
        'src/moe_exp/study/session.py',
        'src/moe_exp/study/observer.py',
        'src/moe_exp/final_routing/common.py',
        'src/moe_exp/final_routing/operator.py',
        'src/moe_exp/final_routing/renderer.py',
        'src/moe_exp/models/token_replay.py',
        'src/moe_exp/correlation_pipeline/scoring.py',
        'src/moe_exp/correlation_pipeline/spans.py',
        'src/moe_exp/correlation_pipeline/dynamics/classes.py',
        'src/moe_exp/research_campaign_v3_r6/annotation.py',
        'src/moe_exp/research_campaign_v3_r6/contracts.py',
        'src/moe_exp/research_campaign_v3_r6/inputs.py',
        'src/moe_exp/research_campaign_v3_r6/reader_contract.py',
        'src/moe_exp/research_campaign_v3_r6/scoring.py',
        'src/moe_exp/verification_study/backend.py',
        'scripts/research_campaign_v3_r6/renderer.py',
        'scripts/slurm/rescue_reader_plan.sbatch',
        'scripts/slurm/rescue_reader.sbatch',
    )
    return {name: file_sha(REPO/name) for name in paths}


def frozen_campaign(study: dict) -> dict:
    """Check the inherited campaign and actual annotator program bytes.

    Args:
        study: Study binding containing the frozen parent campaign identity.

    Returns:
        Verified parent campaign with an unchanged annotation program.
    """
    campaign_path = Path(study['source_campaign'])/'CAMPAIGN.json'
    campaign = verified(campaign_path)
    require(file_sha(campaign_path) == study['source_campaign_sha256'],
            'rescue reader parent campaign differs from the study binding')
    annotator = campaign['annotator']
    require(file_sha(Path(annotator['program'])) == annotator['config']['program_sha256'],
            'frozen rescue reader program bytes changed')
    return campaign


def context_data(study: dict, model: str) -> dict[str, str]:
    """Bind small source tables; referenced trace lines already verify their hashes."""
    campaign = frozen_campaign(study)
    return {
        'study_receipt_sha256': verified(DRIVER.parent/'BINDING.json')['sha256'],
        'source_campaign_receipt_sha256': campaign['sha256'],
        'source_campaign_file_sha256': study['source_campaign_sha256'],
        'sources_sha256': file_sha(Path(study['source_campaign'])/model/'SOURCES.json'),
    }


class RescueContexts:
    """Join frozen assignments, terminal outputs and their exact reader inputs."""

    def __init__(self, study: dict, model: str, tokenizer: object, release: dict):
        from .runtime import closure

        self.study = study
        self.model = model
        self.tokenizer = tokenizer
        self.closure = closure(tokenizer, model)
        self.units = bulk(study)/model/STORES['rescue']
        self.annotator = frozen_campaign(study)['annotator']
        self.generation_jobs = set(release['jobs'])
        self.tasks: dict[str, dict] = {}

    def context(self, row: dict) -> dict:
        """Rebuild one saved continuation; preserve failures as explicit inventory."""
        inventory = {'uid': row['uid'], 'family': row['family'], 'arm': row['arm']}
        folder = self.units/row['uid']
        result_path = folder/'RESULT.json'
        if not result_path.is_file():
            return {**inventory, 'status': 'generation_unavailable'}
        result = verified(result_path)
        if result.get('status') != 'complete':
            return {**inventory, 'status': 'generation_unknown', 'generation_status': result.get('status')}

        # These joins prevent a valid JSON receipt from being attributed to a
        # different arm or to an earlier output of the same continuation.
        raw = verified(folder/'OUTPUT.json')
        assignment_hash = digest(row)
        require(result['assignment_sha256'] == raw['assignment_sha256'] == assignment_hash,
                f"{row['uid']}: rescue assignment and output differ")
        require(result['raw_sha256'] == raw['sha256'],
                f"{row['uid']}: rescue result refers to another output")
        require(raw['uid'] == result['uid'] == row['uid'],
                f"{row['uid']}: rescue output identity differs")
        require(raw['profile_sha256'] == row['profile_sha256']
                == self.study['models'][self.model]['profile_sha256'],
                f"{row['uid']}: rescue generation profile differs")
        inventory.update(raw_sha256=raw['sha256'], result_sha256=result['sha256'])
        if not generation_matches(row, raw, self.generation_jobs):
            return {
                **inventory,
                'status': 'generation_source_mismatch',
                'segment_jobs': [segment.get('job') for segment in raw.get('segments', [])],
                'total_allowance': raw.get('total_allowance'),
                'generated_tokens': len(raw['token_ids']),
            }
        return self._sentence_context(row, raw, inventory)

    def _sentence_context(self, row: dict, raw: dict, inventory: dict) -> dict:
        """Keep the first complete new sentence, with its original context identity."""
        from .outcomes import record
        from moe_exp.study.contexts import context_id

        try:
            rebuilt = record(self.study, self.model, row, raw, self.tokenizer, self.closure)
        except Exception as error:
            return {
                **inventory,
                'status': 'extraction_failed',
                'error': f'{type(error).__name__}: {error}',
            }
        sentences = [sentence for sentence in rebuilt['sentences'] if sentence['first_complete']]
        require(len(sentences) <= 1, f"{row['uid']}: multiple first complete sentences")
        inventory.update(
            finish_reason=rebuilt['finish_reason'],
            generated_tokens=rebuilt['generated_tokens'],
            reasoning_end=rebuilt['reasoning_end'],
        )
        if not sentences:
            return {**inventory, 'status': 'no_complete_sentence'}
        sentence = sentences[0]
        require(sentence['complete'] and not sentence['straddles_prefix'],
                f"{row['uid']}: first complete reader sentence straddles the prefix")
        inputs = sentence['inputs']
        annotation_uid = context_id(inputs, self.annotator)
        task = {'uid': annotation_uid, 'inputs': inputs, 'model': self.model}
        require(annotation_uid not in self.tasks or self.tasks[annotation_uid] == task,
                'reader context identity collision')
        self.tasks[annotation_uid] = task
        return {
            **inventory,
            'status': 'reader_task',
            'annotation_uid': annotation_uid,
            'sentence_index': sentence['sentence_index'],
            'token_start': sentence['token_start'],
            'token_end': sentence['token_end'],
        }


def build_tasks(model: str, *, cohort: Path | None = None) -> dict:
    """Plan the explicit cohort; historical unbound task stores stay read-only.

    Args:
        model: Historical model argument, retained for callers.
        cohort: Required immutable cohort descriptor.

    Returns:
        Full cohort task and availability receipt.
    """
    require(cohort is not None, 'rescue reader planning requires --cohort')
    return cohort_tasks(Path(cohort))


def reader_tasks(model: str, *, cohort: Path | None = None) -> list[dict]:
    """Read exact eligible tasks without requiring every assignment to succeed.

    Args:
        model: Model whose task records are requested.
        cohort: Required immutable cohort descriptor.

    Returns:
        Model-specific exact-context task rows from the validated cohort index.
    """
    from .cohort import read
    require(cohort is not None, 'rescue reader tasks require --cohort')
    return [row for row in verified_tasks(read(Path(cohort)))['records'] if row['model'] == model]


def validate_inventory(receipt: dict, manifest: dict, annotator: dict, model: str) -> None:
    """Require complete assignment coverage and exact frozen context identities.

    Args:
        receipt: Corrected reader tasks and per-assignment inventory.
        manifest: Original rescue assignment manifest.
        annotator: Frozen reader configuration used for context identities.
        model: Model that owns these assignments.

    Returns:
        None; raises on an incomplete or inconsistent join.
    """
    from moe_exp.study.contexts import context_id

    assigned = [row['uid'] for row in manifest['records']]
    inventoried = [row['uid'] for row in receipt['inventory']]
    require(len(set(assigned)) == len(assigned) and sorted(inventoried) == sorted(assigned),
            'rescue reader inventory does not cover the exact assignment roster')
    require(receipt['contexts'] == len(inventoried), 'rescue reader context count differs')
    tasks = receipt['records']
    task_ids = [task['uid'] for task in tasks]
    require(receipt['tasks'] == len(tasks) == len(set(task_ids)), 'reader task count or uniqueness differs')
    linked = {row['annotation_uid'] for row in receipt['inventory'] if row['status'] == 'reader_task'}
    require(linked == set(task_ids), 'rescue reader tasks and assignment inventory do not join')
    for task in tasks:
        require(task['model'] == model and task['uid'] == context_id(task['inputs'], annotator),
                'rescue reader task differs from its frozen context identity')


def run(model: str, deadline: float, *, cohort: Path | None = None, wave: str = 'pilot') -> dict:
    """Execute a cohort-owned reader wave, preserving unknown assignments.

    Args:
        model: Historical model argument; both source models share the frozen wave.
        deadline: Absolute allocated deadline.
        cohort: Required immutable cohort descriptor.
        wave: Fixed pilot or annotation-wave identifier.

    Returns:
        Actual completed or incomplete wave receipt.
    """
    require(cohort is not None, 'rescue reader execution requires --cohort')
    return run_cohort(Path(cohort), wave, deadline)


def _model_inventory(value: dict, index: dict, model: str, tasks: dict) -> tuple:
    """Join one model extraction to all assignments and exact first-sentence inputs."""
    import json
    import pandas as pd
    from .cohort import assignments
    from .measurement import audit_frame
    from moe_exp.study.contexts import context_id
    references = {row['uid']: row for row in index['records']}
    inventory, occurrences, files = [], [], {}
    folder = Path(value['paths']['extraction'])/model
    extraction = verified(folder/'EXTRACTION.json')
    require(extraction['cohort_sha256'] == value['sha256']
                and index['extraction_sha256'][model] == extraction['sha256'], 'cohort extraction differs')
    check_files(extraction['files'])
    files.update(extraction['files'])
    replays = {row['uid']: json.loads(row['replay_json'])
               for row in pd.read_parquet(folder/'terminal-replay.parquet').to_dict('records')}
    for assignment in assignments(value, model):
        uid = assignment['uid']
        replay = replays.get(uid)
        row = {'uid': uid, 'model': model, 'status': 'unknown', 'reason': references[uid]['status']}
        if replay is not None:
            sentences = replay['sentences']
            selected = next((sentence for sentence in sentences if sentence['complete']
                             and not sentence['straddles_prefix'] and 0 <= sentence['token_start'] < 512), None)
            if selected is not None:
                annotation_uid = context_id(selected['inputs'], value['annotator'])
                task = {'uid': annotation_uid, 'model': model, 'inputs': selected['inputs']}
                require(annotation_uid not in tasks or tasks[annotation_uid]['inputs'] == task['inputs'],
                        'cohort context collision')
                tasks.setdefault(annotation_uid, task)
                row.update(status='reader_task', annotation_uid=annotation_uid,
                           sentence_index=selected['sentence_index'], token_start=selected['token_start'])
            elif replay['finish_reason'] == 'stop' or replay['reasoning_end'] is not None:
                row.update(status='no_event', reason='genuine_reasoning_termination')
            else:
                row.update(reason='censored_without_complete_sentence')
            occurrences.extend(audit_frame(value, assignment, replay))
        inventory.append(row)
    return inventory, occurrences, files


def cohort_tasks(path: Path) -> dict:
    """Freeze first-sentence and probability-sampled trajectory tasks label-blind.

    Args:
        path: Required immutable cohort descriptor.

    Returns:
        Full inventory, exact contexts and the frozen occurrence audit frame.
    """
    import pandas as pd
    from collections import Counter
    from .cohort import generation_index, read
    from .lexical import predict
    from .measurement import sample_occurrences
    from moe_exp.study.contexts import context_id
    value = read(path)
    index = generation_index(value)
    tasks, inventory, occurrences, files = {}, [], [], {}
    for model in ('gpt', 'qwen36'):
        model_inventory, model_occurrences, model_files = _model_inventory(value, index, model, tasks)
        inventory.extend(model_inventory)
        occurrences.extend(model_occurrences)
        files.update(model_files)
    if occurrences:
        sentences = pd.DataFrame([{
            'fold': row['fold'], 'sentence_text': row['inputs']['sentence'],
            'prev_sentence': row['inputs']['previous_sentence'], 'next_sentence': row['inputs']['next_sentence'],
        } for row in occurrences])
        # The lexical implementation uses sentinels for absent neighbors.
        sentences.loc[sentences.prev_sentence == '', 'prev_sentence'] = '<START OF RESPONSE>'
        sentences.loc[sentences.next_sentence == '', 'next_sentence'] = '<END OF RESPONSE>'
        predicted = predict(sentences, Path(value['classifier']['root']), frozen=value['classifier'])
        for row, label in zip(occurrences, predicted.lexical_label):
            row['lexical_label'] = label
    audit = sample_occurrences(value, occurrences)
    for occurrence in audit['selected']:
        uid = context_id(occurrence['inputs'], value['annotator'])
        occurrence['annotation_uid'] = uid
        tasks.setdefault(uid, {'uid': uid, 'model': occurrence['model'], 'inputs': occurrence['inputs']})
    records = sorted(tasks.values(), key=lambda task: digest([20261020, 'reader-task-order', task['uid']]))
    return publish(Path(value['paths']['control'])/'READER_TASKS.json', {
        'cohort_sha256': value['sha256'],
        'generation_index_sha256': index['sha256'],
        'annotator_sha256': digest(value['annotator']),
        'input_files': files,
        'inventory': inventory,
        'availability': dict(Counter(row['status'] for row in inventory)),
        'records': records,
        'audit': audit,
        'semantic_endpoint_frozen': '2026-10-20, after historical generation; first fully new complete sentence starting in [0,512)',
    })


def verified_tasks(value: dict) -> dict:
    """Verify exact task inputs, annotator and full-roster inventory joins.

    Args:
        value: Verified immutable cohort.

    Returns:
        Immutable reader-task receipt; unavailable generations remain permitted.
    """
    from .cohort import assignments, generation_index
    from moe_exp.study.contexts import context_id
    receipt = verified(Path(value['paths']['control'])/'READER_TASKS.json')
    require(receipt['cohort_sha256'] == value['sha256']
            and receipt['generation_index_sha256'] == generation_index(value)['sha256'], 'reader cohort/index differs')
    require(receipt['annotator_sha256'] == digest(value['annotator']), 'reader configuration differs')
    check_files(receipt['input_files'])
    require(sorted(row['uid'] for row in receipt['inventory']) == sorted(row['uid'] for row in assignments(value)),
            'reader inventory differs from the planned population')
    ids = [row['uid'] for row in receipt['records']]
    require(len(ids) == len(set(ids)), 'duplicate reader context identity')
    for row in receipt['records']:
        require(row['uid'] == context_id(row['inputs'], value['annotator']), 'reader context input changed')
    require({row['annotation_uid'] for row in receipt['inventory'] if row['status'] == 'reader_task'} <= set(ids),
            'first-sentence inventory does not join annotation tasks')
    check_files(value['reader_runtime_files'])
    return receipt


def annotation_binding(value: dict, row: dict, *, reused: bool = False) -> dict:
    """Bind raw/result annotations to exact context and unchanged reader science.

    Args:
        value: Cohort descriptor.
        row: Frozen exact-context task.
        reused: Historical API annotations specified no per-request seed.

    Returns:
        Immutable annotation identity including seed and class mapping.
    """
    return {
        'cohort_sha256': value['sha256'],
        'inputs_sha256': digest(row['inputs']),
        'annotator_sha256': digest(value['annotator']),
        'seed': None if reused else int(digest([20261008, 'reader', row['uid']])[:8], 16),
        'origin': 'historical_unseeded_api' if reused else 'new_cohort_reader',
        'class_mapping': value['class_mapping'],
    }


def freeze_reuse(value: dict, tasks: dict) -> dict:
    """Snapshot reused label bytes after verifying exact index and source-row joins.

    Args:
        value: Verified cohort with the frozen reader configuration.
        tasks: Frozen label-blind task selection.

    Returns:
        Immutable exact-context label snapshot; later lookups use this receipt.
    """
    import sqlite3
    from moe_exp.study.contexts import context_id
    from .rescue_source import load_study_binding
    study = load_study_binding(Path(value['study_binding_path']))
    parent = Path(study['source_campaign'])
    sources, hashes, records = {}, {}, []
    for model in ('gpt', 'qwen36'):
        index = parent/model/'LABEL_INDEX.sqlite'
        if not index.exists():
            continue
        before = file_sha(index)
        with sqlite3.connect(f'file:{index}?mode=ro', uri=True) as database:
            for task in tasks['records']:
                if task['model'] != model:
                    continue
                match = database.execute('select label,source,row_index from labels where context=?',
                                         (task['uid'],)).fetchone()
                if match is None:
                    continue
                label, source, row_index = match
                if source not in sources:
                    hashes[source] = file_sha(Path(source))
                    sources[source] = load(Path(source))
                row = sources[source][row_index]
                require((label is None or row['label'] == label) and row['inputs'] == task['inputs']
                        and context_id(row['inputs'], value['annotator']) == task['uid'],
                        'annotation index/source row/context join differs')
                from moe_exp.io import CLASSES
                require(label is None or label in CLASSES, 'invalid reused annotation class')
                records.append({
                    'uid': task['uid'], 'label': label, 'inputs_sha256': digest(task['inputs']),
                    'index': str(index), 'index_sha256': before, 'source': source,
                    'source_sha256': hashes[source], 'row_index': row_index,
                    'annotation_binding': annotation_binding(value, task, reused=True),
                })
        require(file_sha(index) == before, 'annotation index bytes changed during lookup')
    check_files(hashes)
    return publish(Path(value['paths']['control'])/'REUSED_LABELS.json', {
        'cohort_sha256': value['sha256'], 'tasks_sha256': tasks['sha256'], 'records': records})


def freeze_waves(value: dict, tasks: dict) -> dict:
    """Price actual input lengths and freeze pilot plus two disjoint waves.

    Args:
        value: Verified cohort.
        tasks: Frozen exact contexts, before reading any new annotation result.

    Returns:
        Longest-input pilot, representative contexts and disjoint remaining order.
    """
    from transformers import AutoTokenizer
    from moe_exp.study.renderer import Renderer
    from moe_exp.study.renderer import token_ids
    from .rescue_source import load_study_binding
    study = load_study_binding(Path(value['study_binding_path']))
    campaign = frozen_campaign(study)
    annotator = value['annotator']
    require(campaign['annotator'] == annotator, 'frozen reader differs from cohort')
    tokenizer = AutoTokenizer.from_pretrained(annotator['path'], local_files_only=True)
    renderer = Renderer(campaign)
    lengths, unsupported = {}, []
    reused = {row['uid'] for row in verified(Path(value['paths']['control'])/'REUSED_LABELS.json')['records']}
    try:
        for task in tasks['records']:
            conversation = renderer.request('render', inputs=task['inputs'])['conversation']
            length = len(token_ids(tokenizer, conversation, annotator['config']))
            if length+16384 > annotator['native_context']:
                unsupported.append(task['uid'])
            else:
                lengths[task['uid']] = length
    finally:
        renderer.close()
    ordered = sorted((uid for uid in lengths if uid not in reused),
                     key=lambda uid: digest([20261020, 'reader-task-order', uid]))
    # The length endpoints are compulsory; the remaining pilot slots are drawn
    # across length ranks and hash order without consulting annotation outcomes.
    by_length = sorted(ordered, key=lambda uid: (lengths[uid], uid))
    pilot = []
    if by_length:
        positions = {round(index*(len(by_length)-1)/max(1, min(64, len(by_length))-1))
                     for index in range(min(64, len(by_length)))}
        pilot = [by_length[index] for index in sorted(positions)]
        pilot += [uid for uid in ordered if uid not in set(pilot)][:max(0, 128-len(pilot))]
    remaining = [uid for uid in ordered if uid not in set(pilot)]
    return publish(Path(value['paths']['control'])/'READER_WAVES.json', {
        'cohort_sha256': value['sha256'], 'tasks_sha256': tasks['sha256'],
        'pilot': pilot, 'wave-0': remaining[::2], 'wave-1': remaining[1::2],
        'reserve': ordered, 'reused': sorted(reused), 'unsupported_unknown': unsupported,
        'input_lengths': lengths, 'frozen_before_new_annotation': True,
    })


def admit_reader(value: dict) -> dict:
    """Admit both frozen reader waves from actual pilot costs with 25% margin.

    Args:
        value: Immutable cohort descriptor.

    Returns:
        Admitted or incomplete forecast; pilot labels remain available either way.
    """
    import math
    plan = verified(Path(value['paths']['control'])/'READER_WAVES.json')
    batches = [verified(path) for path in (Path(value['paths']['reader'])/'batches').glob('*.json')]
    batches = [row for row in batches if set(row['units']) <= set(plan['pilot'])]
    require(all(row['cohort_sha256'] == value['sha256'] for row in batches), 'pilot belongs to another cohort')
    require(batches or not plan['pilot'], 'reader pilot has no measured batch evidence')
    batch_seconds = max((row['seconds'] for row in batches), default=0.)
    loading = max((row['load_seconds'] for row in batches), default=0.)
    terminals = [verified(path) for path in (Path(value['paths']['reader'])/'waves'/'pilot'/'jobs').glob('*.json')]
    require(terminals or not plan['pilot'], 'pilot shutdown/initialization evidence missing')
    initialization = max((row['initialization_seconds'] for row in terminals if row.get('engine_loaded')), default=0.)
    shutdown = max((row['shutdown_seconds'] for row in terminals if row.get('engine_loaded')), default=0.)
    # Parsing/preparation are already included in batch seconds. Initialization
    # and shutdown are measured separately; input extremes were forced into the pilot.
    forecast = {wave: 1.25*(initialization+loading+math.ceil(len(plan[wave])/64)*batch_seconds+shutdown+60)
                for wave in ('wave-0', 'wave-1')}
    lengths = plan['input_lengths'].values()
    return publish(Path(value['paths']['control'])/'READER_ADMISSION.json', {
        'cohort_sha256': value['sha256'], 'tasks_sha256': plan['tasks_sha256'],
        'status': 'admitted' if all(seconds <= 6*3600 for seconds in forecast.values()) else 'incomplete',
        'forecast_seconds': forecast, 'forecast_margin': 1.25,
        'batch_seconds': 1.25*batch_seconds, 'input_min': min(lengths, default=0),
        'input_max': max(lengths, default=0), 'pilot_is_not_a_worst_case_guarantee': True,
        'initialization_seconds': initialization, 'load_seconds': loading, 'shutdown_seconds': shutdown,
    })


def context_labels(value: dict, receipt: dict) -> dict:
    """Load only exact-context, valid-result reader labels.

    Args:
        value: Verified cohort descriptor.
        receipt: Verified frozen task receipt.

    Returns:
        Context labels with unavailable annotations omitted for unknown handling.
    """
    from moe_exp.io import CLASSES
    units = Path(value['paths']['reader'])/'annotations'
    snapshot = verified(Path(value['paths']['control'])/'REUSED_LABELS.json')
    require(snapshot['cohort_sha256'] == value['sha256'] and snapshot['tasks_sha256'] == receipt['sha256'],
            'reused label snapshot differs')
    tasks = {row['uid']: row for row in receipt['records']}
    labels = {}
    for row in snapshot['records']:
        require(row['uid'] in tasks and row['inputs_sha256'] == digest(tasks[row['uid']]['inputs'])
                and (row['label'] is None or row['label'] in CLASSES)
                and row['annotation_binding'] == annotation_binding(value, tasks[row['uid']], reused=True),
                'reused label snapshot task join differs')
        if row['label'] in CLASSES:
            labels[row['uid']] = row['label']
    for task in receipt['records']:
        path = units/task['uid']/'RESULT.json'
        if not path.is_file():
            continue
        result = verified(path)
        require(result['uid'] == task['uid'] and result['assignment_sha256'] == digest(task), 'reader result task join differs')
        reused = result.get('source', {}).get('annotation_binding', {}).get('origin') == 'historical_unseeded_api'
        require(result['annotation_binding'] == annotation_binding(value, task, reused=reused), 'reader annotation provenance differs')
        if result['status'] == 'complete':
            require(result['label'] in CLASSES, 'reader result class is invalid')
            if 'raw_sha256' in result:
                raw = verified(path.parent/'OUTPUT.json')
                require(result['raw_sha256'] == raw['sha256'] and raw['assignment_sha256'] == digest(task)
                        and raw['annotation_binding'] == result['annotation_binding'], 'reader output/result join differs')
            labels[task['uid']] = result['label']
    return labels


def semantic_labels(value: dict, receipt: dict) -> dict:
    """Join validated context labels to every assignment, preserving unknowns.

    Args:
        value: Verified cohort descriptor.
        receipt: Verified immutable task receipt.

    Returns:
        Full assignment inventory with exact-context semantic labels.
    """
    labels = context_labels(value, receipt)
    return {row['uid']: {**row, 'label': labels.get(row.get('annotation_uid'))} for row in receipt['inventory']}


def run_cohort(path: Path, wave: str, deadline: float) -> dict:
    """Reuse, parse or admit only the fixed disjoint reader wave within its budget.

    Args:
        path: Immutable cohort descriptor.
        wave: Pilot, one of two disjoint waves, or the single operational reserve.
        deadline: Allocation deadline, clipped to the frozen science cutoff.

    Returns:
        Immutable wave receipt with all task dispositions.
    """
    import time
    from .cohort import identity, read
    from .ownership import check_time, worker
    from .reader import Reader
    from .rescue_source import load_study_binding
    value = read(path)
    deadline = min(deadline, check_time(value))
    tasks = verified_tasks(value)
    plan = verified(Path(value['paths']['control'])/'READER_WAVES.json')
    require(plan['tasks_sha256'] == tasks['sha256'], 'reader wave tasks differ')
    selected = set(plan[wave])
    rows = [row for row in tasks['records'] if row['uid'] in selected]
    require({row['uid'] for row in rows} == selected, 'reader wave roster differs')
    root = Path(value['paths']['reader'])
    execution = identity(value, 'reader', 'annotation', sorted(row['uid'] for row in tasks['records']))
    with worker(root, execution) as owner:
        operation = owner['shape'].get('operation', owner['shape']['stage'])
        require(operation == wave, 'accepted reader owner belongs to another frozen wave')
        import os
        from moe_exp.io import Units
        units = Units(root/'annotations')
        terminal = root/'waves'/wave/'jobs'/(os.environ['SLURM_JOB_ID']+'.json')
        existing = [units.state(row['uid']) for row in rows]
        if not any(state is None or state['status'] == 'output' for state in existing):
            return publish(terminal, {
                'cohort_sha256': value['sha256'], 'wave': wave, 'engine_loaded': False,
                'tasks': len(rows), 'status': 'complete'})
        require(time.time()+900 < deadline, 'reader has insufficient model loading reserve')
        admission = None
        if wave != 'pilot':
            admission = verified(Path(value['paths']['control'])/'READER_ADMISSION.json')
            require(admission['status'] == 'admitted' and admission['tasks_sha256'] == tasks['sha256'], 'reader work not admitted')
        started = time.monotonic()
        reader = Reader(load_study_binding(Path(value['study_binding_path'])), deadline, load_model=False)
        initialization = time.monotonic()-started
        reader.cohort = value
        try:
            try:
                batches = reader.run(rows, admission=admission)
            except TimeoutError:
                batches = []
        finally:
            started = time.monotonic()
            reader.close()
            shutdown = time.monotonic()-started
        states = [units.state(row['uid']) for row in rows]
        unfinished = sum(state is None or state['status'] == 'output' for state in states)
        read(path)
        return publish(terminal, {
            'cohort_sha256': value['sha256'], 'wave': wave, 'engine_loaded': reader.llm is not None,
            'tasks': len(rows), 'batches': batches, 'status': 'incomplete' if unfinished else 'complete',
            'untouched_or_unparsed': unfinished, 'initialization_seconds': initialization, 'shutdown_seconds': shutdown})


def main(argv=None):
    """Require an immutable cohort for every semantic planning or execution command."""
    import argparse
    import json
    import os
    parser = argparse.ArgumentParser(prog='moe_exp.study.rescue_reader')
    parser.add_argument('command', choices=('plan', 'run'))
    parser.add_argument('--model', required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--wave', choices=('pilot', 'wave-0', 'wave-1', 'reserve'), default='pilot')
    args = parser.parse_args(argv)
    if args.command == 'plan':
        from .rescue import stage_worker
        print(json.dumps(stage_worker(args.cohort, 'prepare', None)['sha256']))
        return
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS'),
            'rescue reader run requires a Slurm GPU allocation')
    from .rescue import stage_worker
    print(json.dumps(stage_worker(args.cohort, args.wave, None)['sha256']))


if __name__ == '__main__':
    main()
