"""Content-bound cohort planning for campaign v3; no scheduling or inference.

Capture manifests bind ONLY the cohort manifest (embedded manifest_sha256 plus
file sha256), the trace bytes named by each cohort row, and the frozen code
binding. Corrected outcomes are joined at ANALYSIS time from an explicitly
passed sidecar; planning may read one as a consistency reference but never
writes an outcome field into a capture manifest, so cohorts A and B share one
verify receipt and a new outcome version never invalidates a capture.

JSON outcome rows are supported for small fixtures/export interchange. Production
parquet is read with pyarrow, without coercing nullable outcomes to booleans.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

from measurement_study.production_plan import atomic, digest, sha

INPUT_CAPS = {'gpt': 131072, 'glm': 133120, 'qwen36': 133120,
              'nemotron': 133120, 'gemma': 133120, 'qwen35': 133120,
              'qwen330b': 40960}
BACKENDS = {'gpt': 'hf', 'qwen330b': 'hf', 'glm': 'vllm',
            'qwen36': 'vllm', 'nemotron': 'vllm', 'gemma': 'vllm', 'qwen35': 'vllm'}
OUTCOME_FIELDS = ('strict_correct', 'adjudicated_correct', 'minerva_tolerance_correct')
STATUSES = {'math_verify_true', 'judge_equivalent', 'judge_not_equivalent',
            'uncertain', 'no_answer', 'capped', 'unscored', 'pending'}
UNRESOLVED_STATUSES = {'uncertain', 'unscored', 'pending'}
CORPORA = {'historical', 'card-v3', 'continuation'}
UNKNOWN_TERMINATION = 'unknown'
COHORT_CORE_FIELDS = ('schema_version', 'model', 'seed', 'label_total', 'per_b', 'floor', 'traces_sha256')
MIN_VERIFICATION_SECONDS = 5 * 3600
MIN_DRAIN_MARGIN_SECONDS = 900
UNMEASURED_DRAIN_MARGIN_SECONDS = 1800


def outcome_key(row):
    """Join identity used by the analysis-time outcome join (trace hash excluded)."""
    return (row['corpus'], row['model'], row['dataset'], row['problem_id'],
            _nullable_integer(row.get('sample_id'), 'sample_id'))


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError('Invalid ' + name)
    return value


def _nullable_integer(value, name, minimum=0):
    if value is None:
        return None
    return _integer(value, name, minimum)


def _hex_digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(character not in '0123456789abcdef' for character in value):
        raise ValueError('Invalid ' + name)
    return value


def attempt_id(model, dataset, problem_id, sample_id, trace_sha256):
    return digest([model, dataset, problem_id, sample_id, trace_sha256])


def cohort_core_digest(cohort):
    """Recompute the digest sample_cohorts.write embeds as manifest_sha256."""
    missing = [key for key in COHORT_CORE_FIELDS if key not in cohort]
    if missing:
        raise ValueError('Cohort manifest core fields missing: ' + ','.join(missing))
    return digest({key: cohort[key] for key in COHORT_CORE_FIELDS})


def load_cohort_manifest(path):
    """Load a v1 card-v3 cohort manifest and bind its content identity."""
    path = Path(path)
    cohort = json.loads(path.read_text())
    if not isinstance(cohort, dict) or cohort.get('schema_version') != 1 or cohort.get('corpus') != 'card-v3':
        raise ValueError('Expected v1 card-v3 cohort manifest')
    if not isinstance(cohort.get('model'), str) or not cohort['model']:
        raise ValueError('Cohort manifest model required')
    embedded = _hex_digest(cohort.get('manifest_sha256'), 'cohort manifest_sha256')
    if cohort_core_digest(cohort) != embedded:
        raise ValueError('Cohort manifest_sha256 does not match its core fields')
    return cohort, dict(schema_version=1, manifest_sha256=embedded, file_sha256=sha(path),
                        path=str(path.resolve()))


def validate_cohort_binding(binding, path=None):
    """Re-read the bound cohort manifest; used by capture and gate before any work."""
    if not isinstance(binding, dict) or not binding.get('path'):
        raise ValueError('Missing cohort binding')
    if path is not None and Path(path).resolve() != Path(binding['path']).resolve():
        raise ValueError('Cohort manifest path differs from its binding')
    cohort_path = Path(binding['path'])
    if not cohort_path.is_file() or sha(cohort_path) != binding.get('file_sha256'):
        raise ValueError('Cohort manifest changed since planning')
    cohort, reread = load_cohort_manifest(cohort_path)
    if reread['manifest_sha256'] != binding.get('manifest_sha256'):
        raise ValueError('Cohort manifest identity changed since planning')
    return cohort


def cohort_duplicate_report(cohort):
    """State the capture cost of planning cohorts A and B as separate sets.

    A and B memberships live in one cohort manifest, but the campaign plans B
    before labels exist and A after the label gate. An attempt selected by both
    cohorts would then be captured twice with different cohort variants, so the
    duplicate cost is measured here instead of being silently absorbed. Planning
    a single label-complete `--cohort all` set captures every attempt once.
    """
    attempts = {'A': set(), 'B': set()}
    tokens = {'A': {}, 'B': {}}
    for row in cohort.get('rows', []):
        if row.get('cohort') not in attempts:
            raise ValueError('Unknown cohort row')
        aid = attempt_id(cohort['model'], row['dataset'], row['problem_id'], row['sample_id'], row['trace_sha256'])
        attempts[row['cohort']].add(aid)
        tokens[row['cohort']][aid] = _integer(row['completion_tokens'], 'completion_tokens')
    shared = attempts['A'] & attempts['B']
    return dict(
        cohort_a_attempts=len(attempts['A']), cohort_b_attempts=len(attempts['B']),
        union_attempts=len(attempts['A'] | attempts['B']), shared_attempts=len(shared),
        cohort_a_completion_tokens=sum(tokens['A'].values()),
        cohort_b_completion_tokens=sum(tokens['B'].values()),
        duplicated_completion_tokens=sum(tokens['B'][aid] for aid in shared),
        single_capture_set_plan='--cohort all captures each attempt once and lets both cohorts reference the same packet',
        separate_plan_cost='separate A/B plans re-capture the shared attempts; duplicated_completion_tokens is the extra GPU work')


def read_outcomes(path, manifest_path=None, *, corpus=None, model=None):
    """Return validated rows and the exact sidecar/manifest provenance binding.

    `path` may be the immutable parquet/JSON sidecar or the mutable pointer that
    names it plus its sha256. Only the requested corpus/model slice is validated,
    so historical rows with inherited null completion tokens and capped/pending
    statuses can never block a card-v3 plan. Null terminations are normalized to
    the explicit string 'unknown'; nullable Int64 columns stay None.
    """
    if corpus is not None and corpus not in CORPORA:
        raise ValueError('Unknown outcome corpus filter')
    path = Path(path)
    pointer_binding = None
    document = None
    if path.suffix != '.parquet':
        try:
            document = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            document = None
    if isinstance(document, dict) and 'rows' not in document and 'path' in document and 'sha256' in document:
        resolved = Path(document['path'])
        if not resolved.is_absolute():
            raise ValueError('Outcome pointer must name an absolute sidecar path')
        manifest_path = Path(manifest_path) if manifest_path else resolved.with_suffix('.manifest.json')
        pointer_binding = dict(pointer_path=str(path.resolve()), pointer_sha256=sha(path))
    else:
        resolved = path
        document = None
        manifest_path = Path(manifest_path) if manifest_path else resolved.with_suffix('.manifest.json')
    checksum = sha(resolved)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('sha256') != checksum:
        raise ValueError('Outcome sidecar checksum mismatch')
    if pointer_binding is not None and document['sha256'] != checksum:
        raise ValueError('Outcome pointer/checksum mismatch')
    if pointer_binding is not None and pointer_binding['pointer_sha256'] != sha(path):
        raise ValueError('Outcome pointer changed during read')
    if resolved.suffix == '.parquet':
        import pyarrow.parquet as pq
        rows = pq.read_table(resolved).to_pylist()
    else:
        document = json.loads(resolved.read_text())
        rows = document if isinstance(document, list) else document['rows']
    seen = set()
    required = {'corpus', 'model', 'dataset', 'problem_id', 'source_problem_id',
                'sample_id', 'trace_sha256', 'completion_tokens', 'termination',
                'adjudication_status', 'scorer_contract_version', *OUTCOME_FIELDS}
    selected = []
    for original in rows:
        if corpus is not None and original.get('corpus') != corpus:
            continue
        if model is not None and original.get('model') != model:
            continue
        if not isinstance(original, dict):
            raise ValueError('Outcome row must be an object')
        row = dict(original)
        if required - row.keys():
            raise ValueError('Missing outcome columns: ' + ','.join(sorted(required - row.keys())))
        if row['corpus'] not in CORPORA:
            raise ValueError('Unknown outcome corpus')
        for name in ('model', 'dataset', 'problem_id'):
            if not isinstance(row[name], str) or not row[name]:
                raise ValueError('Invalid outcome ' + name)
        if row['source_problem_id'] is not None and not isinstance(row['source_problem_id'], str):
            raise ValueError('Invalid outcome source_problem_id')
        if row['trace_sha256'] is not None:
            _hex_digest(row['trace_sha256'], 'trace_sha256')
        row['sample_id'] = _nullable_integer(row['sample_id'], 'sample_id')
        row['completion_tokens'] = _nullable_integer(row['completion_tokens'], 'completion_tokens')
        if row['scorer_contract_version'] != 3:
            raise ValueError('Outcome scorer contract must be v3')
        termination = row['termination']
        if termination is None:
            termination = UNKNOWN_TERMINATION
        elif not isinstance(termination, str) or not termination:
            raise ValueError('Invalid termination')
        row['termination'] = termination
        if row['adjudication_status'] not in STATUSES:
            raise ValueError('Unknown adjudication status')
        if any(row[key] is not None and type(row[key]) is not bool for key in OUTCOME_FIELDS):
            raise ValueError('Outcome fields must be nullable booleans')
        if row['adjudication_status'] in UNRESOLVED_STATUSES and row['adjudicated_correct'] is not None:
            raise ValueError('Unresolved adjudication cannot supply definitive correctness')
        key = outcome_key(row)
        if key in seen:
            raise ValueError('Duplicate outcome identity: ' + repr(key))
        seen.add(key)
        selected.append(row)
    # Catch a concurrently replaced sidecar, including a change during parquet reads.
    manifest_checksum = sha(manifest_path)
    if sha(resolved) != checksum or (pointer_binding is not None and sha(path) != pointer_binding['pointer_sha256']):
        raise ValueError('Outcome sidecar changed during read')
    binding = dict(path=str(resolved.resolve()), sha256=checksum,
                   manifest_path=str(manifest_path.resolve()), manifest_sha256=manifest_checksum,
                   corpus=corpus, model=model, rows=len(selected),
                   scorer_contract_versions=sorted({r['scorer_contract_version'] for r in selected}))
    if pointer_binding is not None:
        binding.update(pointer_binding)
    return selected, binding


def trace_content_hash(trace):
    value = {'prompt': trace.get('prompt'), 'messages': trace.get('generation_messages'),
             'system_prompt': trace.get('system_prompt'), 'cot_text': trace.get('cot_text')}
    if 'token_replay' in trace.get('metadata', {}):
        value['token_replay'] = trace['metadata']['token_replay']
    return digest(value)


def source_line(row):
    """Resolve a cohort byte location and independently verify text/token content."""
    path = Path(row['trace_path'])
    stat = path.stat()
    offset = _integer(row['byte_offset'], 'byte_offset')
    with path.open('rb') as stream:
        stream.seek(offset)
        raw = stream.readline()
    if hashlib.sha256(raw).hexdigest() != row['line_sha256']:
        raise ValueError('Source line checksum changed')
    trace = json.loads(raw)
    if trace_content_hash(trace) != row['trace_sha256']:
        raise ValueError('Source trace content identity changed')
    after = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Source changed while reading')
    location = dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                    byte_offset=offset, line_bytes=len(raw), line_sha256=row['line_sha256'],
                    line_index=_integer(row['line_index'], 'line_index'))
    return trace, location


def prepare_spec(source, spec):
    """Require a resolved native checkpoint; never substitute qwen35's proxy."""
    if source not in INPUT_CAPS:
        raise ValueError('Unknown campaign source')
    spec = copy.deepcopy(spec)
    checkpoint = spec.get('checkpoint')
    if not checkpoint or not Path(checkpoint).is_dir() or not spec.get('revision'):
        raise ValueError('Native checkpoint pending resolution: ' + source)
    config_path = Path(checkpoint) / 'config.json'
    if not config_path.is_file() or spec.get('config_sha256') != sha(config_path):
        raise ValueError('Native checkpoint config/hash missing or changed')
    config = json.loads(config_path.read_text())
    text_config = config.get('text_config', config)
    families = {'gpt': {'gpt_oss'}, 'glm': {'glm4_moe_lite'}, 'gemma': {'gemma4','gemma4_text'},
                'nemotron': {'nemotron_h'}, 'qwen330b': {'qwen3_moe'},
                'qwen35': {'qwen3_5_moe','qwen3_5_moe_text'},
                'qwen36': {'qwen3_5_moe','qwen3_5_moe_text','qwen3_6_moe','qwen3_6_moe_text'}}
    if text_config.get('model_type') not in families[source]:
        raise ValueError('Native model family does not match campaign source')
    index = Path(checkpoint) / 'model.safetensors.index.json'
    shards = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
    for name in shards:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or not (Path(checkpoint) / relative).is_file():
            raise ValueError('Native checkpoint weight shard missing/invalid: ' + str(name))
    if source == 'qwen35':
        quant = config.get('quantization_config', text_config.get('quantization_config', {})) or {}
        if quant.get('quant_method') != 'gptq':
            raise ValueError('Qwen3.5 requires its native GPTQ checkpoint')
        receipt_path = spec.get('checkpoint_receipt')
        if not receipt_path or spec.get('checkpoint_receipt_sha256') != sha(receipt_path):
            raise ValueError('Qwen3.5 verified download receipt required')
        receipt = json.loads(Path(receipt_path).read_text())
        if receipt.get('status') not in {'verified', 'complete'} or receipt.get('revision') != spec['revision']:
            raise ValueError('Qwen3.5 download receipt not accepted for this revision')
    from measurement_study.contract import routed_layers
    actual_layers = routed_layers(config)
    layers = spec.get('layers')
    if not isinstance(layers, list) or not layers or len(layers) != len(set(layers)):
        raise ValueError('Missing/duplicate routed layers')
    if any(type(layer) is not int or layer < 0 for layer in layers):
        raise ValueError('Invalid routed layers')
    if layers != actual_layers:
        raise ValueError('Routed topology differs from native checkpoint')
    tokenizer_files = spec.get('tokenizer_files')
    if not isinstance(tokenizer_files, dict) or not {'tokenizer.json', 'tokenizer_config.json'} <= tokenizer_files.keys():
        raise ValueError('Exact tokenizer file hashes required')
    for name, expected in tokenizer_files.items():
        if Path(name).name != name or sha(Path(checkpoint)/name) != expected:
            raise ValueError('Native tokenizer changed')
    for key in ('experts', 'top_k'):
        _integer(spec[key], key, 1)
    if spec['top_k'] > spec['experts']:
        raise ValueError('top_k exceeds expert count')
    native_experts = text_config.get('num_experts', text_config.get('n_routed_experts', text_config.get('num_local_experts')))
    native_top_k = text_config.get('num_experts_per_tok', text_config.get('top_k_experts'))
    if (spec['experts'], spec['top_k']) != (native_experts, native_top_k):
        raise ValueError('Expert dimensions differ from native checkpoint')
    hidden = spec.get('hidden_size', text_config.get('hidden_size'))
    if hidden != text_config.get('hidden_size'):
        raise ValueError('Hidden dimension differs from native checkpoint')
    spec['hidden_size'] = _integer(hidden, 'hidden_size', 1)
    spec['backend'] = BACKENDS[source]
    spec['capture_version'] = 2
    spec['corpus'] = 'card-v3'
    spec['num_experts'] = spec['experts']
    spec['replay_variant'] = ''
    if source == 'gemma':
        spec['kv_cache_dtype'] = 'bfloat16'
    cap = spec.get('max_input_tokens', INPUT_CAPS[source])
    _integer(cap, 'max_input_tokens', 1)
    if cap > INPUT_CAPS[source]:
        raise ValueError('Input cap exceeds approved per-model context')
    spec['max_input_tokens'] = cap
    spec['max_model_len'] = cap
    return spec


def build_campaign_specs(campaign, overrides=None):
    """Seed v3 from existing metadata without treating unresolved weights as ready.

    Caller supplies qwen35's exact downloaded checkpoint/revision/receipt through
    overrides. Missing entries remain explicitly pending and cannot be planned.
    This function reads configuration/tokenizer metadata, never model weights.
    """
    from measurement_study.contract import routed_layers
    sources = copy.deepcopy(campaign.get('sources', {}))
    for source in INPUT_CAPS:
        spec = sources.setdefault(source, {})
        spec.update(copy.deepcopy((overrides or {}).get(source, {})))
        spec.update(capture_version=2, corpus='card-v3', backend=BACKENDS[source],
                    max_input_tokens=INPUT_CAPS[source], max_model_len=INPUT_CAPS[source], replay_variant='')
        if source == 'gemma':
            spec['kv_cache_dtype'] = 'bfloat16'
        checkpoint = Path(spec['checkpoint']) if spec.get('checkpoint') else None
        if checkpoint is None or not (checkpoint/'config.json').is_file():
            spec['native_backend_readiness'] = 'pending_native_checkpoint'
            continue
        config = json.loads((checkpoint/'config.json').read_text())
        text_config = config.get('text_config', config)
        spec['config_sha256'] = sha(checkpoint/'config.json')
        spec['layers'] = routed_layers(config)
        spec['hidden_size'] = text_config['hidden_size']
        spec['experts'] = text_config.get('num_experts', text_config.get('n_routed_experts', text_config.get('num_local_experts')))
        spec['num_experts'] = spec['experts']
        spec['top_k'] = text_config.get('num_experts_per_tok', text_config.get('top_k_experts'))
        spec['tokenizer_files'] = {name: sha(checkpoint/name) for name in ('tokenizer.json', 'tokenizer_config.json') if (checkpoint/name).is_file()}
        try:
            sources[source] = prepare_spec(source, spec)
            sources[source]['native_backend_readiness'] = 'requires_capture_v2_gpu_validation'
        except (ValueError, OSError, KeyError) as error:
            spec['native_backend_readiness'] = 'pending_native_checkpoint'
            spec['pending_reason'] = str(error)
    return sources


def fit_attempt_runtime(attempts, startup_seconds):
    """Fit fixed per-attempt cost plus per-token cost from measured attempts.

    Shard runtime is `startup + sum(per_attempt + seconds_per_token * tokens)`, and the
    measured attempt seconds exclude startup (load.json times it separately); the
    old token-only budget silently charged one startup per shard and nothing per
    attempt, so many short attempts overran their window. Two measured traces
    (the verification pair) determine the slope and intercept; if the intercept
    is negative the fit clamps it to zero and keeps the conservative rate that
    covers every observed attempt instead of inventing a negative fixed cost.
    """
    if not math.isfinite(startup_seconds) or startup_seconds < 0:
        raise ValueError('Measured startup required')
    points = []
    for row in attempts:
        tokens, seconds = row.get('tokens'), row.get('seconds')
        if type(tokens) is not int or tokens <= 0 or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Invalid measured attempt runtime')
        points.append((tokens, float(seconds)))
    if len(points) < 2 or len({tokens for tokens, _ in points}) < 2:
        raise ValueError('At least two distinct-token measured attempts required')
    points.sort()
    mean_tokens = sum(tokens for tokens, _ in points) / len(points)
    mean_seconds = sum(seconds for _, seconds in points) / len(points)
    covariance = sum((tokens - mean_tokens) * (seconds - mean_seconds) for tokens, seconds in points)
    variance = sum((tokens - mean_tokens) ** 2 for tokens, _ in points)
    slope = covariance / variance
    # Attempt timings exclude model startup (load.json measures it separately), so the
    # intercept is the fixed per-attempt cost itself; startup is charged once per shard.
    intercept = mean_seconds - slope * mean_tokens
    if not math.isfinite(slope) or slope <= 0:
        raise ValueError('Measured attempts do not show a positive per-token cost')
    conservative = intercept < 0
    if conservative:
        slope = max(slope, max(seconds / tokens for tokens, seconds in points))
        intercept = 0.0
    if any(intercept + slope * tokens < seconds * (1 - 1e-9) for tokens, seconds in points):
        raise ValueError('Fitted runtime does not cover the measured attempts')
    return dict(startup_seconds=float(startup_seconds), seconds_per_token=float(slope),
                per_attempt_seconds=float(intercept),
                points=[dict(tokens=tokens, seconds=seconds) for tokens, seconds in points],
                method='nonnegative_clamp_conservative_rate' if conservative else 'least_squares_with_measured_startup',
                conservative=bool(conservative))


def runtime_cost(runtime, tokens):
    return float(runtime['startup_seconds']) + float(runtime['per_attempt_seconds']) + float(runtime['seconds_per_token'])*tokens


def token_budget_shards(rows, seconds_per_token, startup_seconds, per_attempt_seconds=0.0, *, factor=1.25, max_hours=16):
    """Deterministic first-fit decreasing on the measured per-attempt runtime."""
    rate = float(seconds_per_token)
    startup = float(startup_seconds)
    per_attempt = float(per_attempt_seconds)
    if not math.isfinite(rate) or rate <= 0 or not math.isfinite(startup) or startup < 0:
        raise ValueError('Measured rate/startup required')
    if not math.isfinite(per_attempt) or per_attempt < 0:
        raise ValueError('Measured nonnegative per-attempt cost required')
    if not 1.25 <= factor or not math.isfinite(factor) or not 0 < max_hours <= 16:
        raise ValueError('Invalid runtime safety envelope')
    envelope = max_hours * 3600 / factor
    if envelope <= startup + per_attempt:
        raise ValueError('Startup exceeds shard envelope')
    groups, sizes = [], []
    for row in sorted(rows, key=lambda r: (-r['completion_tokens'], r['id'])):
        tokens = _integer(row['completion_tokens'], 'completion_tokens', 1)
        if startup + per_attempt + rate * tokens > envelope:
            raise ValueError('One attempt exceeds the 16-hour token budget; stage it')
        slot = next((i for i, size in enumerate(sizes)
                     if startup + per_attempt * (len(groups[i]) + 1) + rate * (size + tokens) <= envelope), None)
        if slot is None:
            groups.append([])
            sizes.append(0)
            slot = len(groups) - 1
        groups[slot].append(row)
        sizes[slot] += tokens
    return [dict(attempts=group, tokens=tokens,
                 expected_attempts=len(group),
                 requested_seconds=math.ceil(factor * (startup + per_attempt * len(group) + rate * tokens) / 1800) * 1800)
            for group, tokens in zip(groups, sizes)]


def validate_storage_root(storage_root):
    path = Path(storage_root).resolve()
    allowed = (Path('/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3'),
               Path('/leonardo_work/IscrC_MIOSR/lmolfett'),
               Path('/leonardo/home/userexternal/lmolfett'))
    if 'campaign-v3' not in path.parts or not any(path.is_relative_to(root) for root in allowed):
        raise ValueError('Explicit user-owned campaign-v3 storage root required')
    return path


def build_rows(cohort, outcome_rows, source, spec, selection='all'):
    """Merge cohort memberships into one row per attempt; never embed outcomes.

    An optional outcome reference is used only to fail closed when the sidecar
    disagrees with the cohort trace identity. Capture planning stays valid when
    the outcome version does not exist yet; analysis joins the sidecar itself.
    """
    if cohort.get('schema_version') != 1 or cohort.get('corpus') != 'card-v3':
        raise ValueError('Expected v1 card-v3 cohort manifest')
    if selection not in {'A', 'B', 'all'}:
        raise ValueError('Unknown cohort selection')
    if cohort['model'] not in {source, spec['generation_model']}:
        raise ValueError('Cohort model mismatch')
    lookup = {outcome_key(row): row for row in (outcome_rows or [])}
    merged = {}
    for item in cohort['rows']:
        if item['cohort'] not in {'A', 'B'}:
            raise ValueError('Unknown cohort row')
        if selection != 'all' and item['cohort'] != selection:
            continue
        _integer(item['sample_id'], 'sample_id')
        trace, location = source_line(item)
        identity = (trace['dataset'], trace['problem_id'], trace.get('sample_id', 0))
        if identity != (item['dataset'], item['problem_id'], item['sample_id']) or trace['model_id'] != spec['generation_model']:
            raise ValueError('Cohort/source identity mismatch')
        if trace.get('source_problem_id') != item.get('source_problem_id'):
            raise ValueError('Source question identity mismatch')
        outcome = lookup.get(('card-v3', cohort['model'], *identity))
        replay = trace.get('metadata', {}).get('token_replay', {})
        prompt, completion = replay.get('prompt_token_ids'), replay.get('completion_token_ids')
        if not isinstance(prompt, list) or not prompt or not isinstance(completion, list) or not completion:
            raise ValueError('Exact saved prompt/completion token IDs required')
        if any(type(i) is not int or i < 0 for ids in (prompt, completion) for i in ids):
            raise ValueError('Invalid saved token ID')
        n = len(completion)
        if n != item['completion_tokens']:
            raise ValueError('Completion token coverage mismatch')
        if len(prompt) + n > spec['max_input_tokens']:
            raise ValueError('Saved trace exceeds per-model input cap')
        termination = item['termination'] if item['termination'] is not None else UNKNOWN_TERMINATION
        if outcome is not None:
            # Optional planning-time reference only; nothing below is embedded.
            if outcome.get('trace_sha256') is not None and outcome['trace_sha256'] != item['trace_sha256']:
                raise ValueError('Outcome trace identity differs from cohort trace')
            if outcome.get('source_problem_id') not in (None, item['source_problem_id']):
                raise ValueError('Outcome question identity mismatch')
            if outcome.get('completion_tokens') is not None and outcome['completion_tokens'] != n:
                raise ValueError('Outcome completion tokens differ from cohort trace')
            if outcome['termination'] != termination:
                raise ValueError('Cohort termination differs from outcome sidecar')
        selection_doc = item['sentence_selection']
        indices = selection_doc.get('indices')
        if selection_doc.get('schema_version') != 1 or not isinstance(indices, list) or len(indices) != len(set(indices)):
            raise ValueError('Invalid cohort sentence selection')
        if any(type(i) is not int or i < 0 for i in indices):
            raise ValueError('Invalid selected sentence index')
        probability = item['inclusion_probability']
        if item['cohort'] == 'A':
            if not isinstance(probability, (int, float)) or isinstance(probability, bool) or not math.isfinite(probability) or not 0 < probability <= 1:
                raise ValueError('Invalid cohort A inclusion probability')
        elif probability is not None:
            raise ValueError('Cohort B inclusion probability must be null (paired case-control selection)')
        arm_probability = item.get('arm_inclusion_probability')
        if item['cohort'] == 'B' and arm_probability is not None and (
                not isinstance(arm_probability, (int, float)) or isinstance(arm_probability, bool)
                or not math.isfinite(arm_probability) or not 0 < arm_probability <= 1):
            raise ValueError('Invalid cohort B arm inclusion probability')
        aid = attempt_id(trace['model_id'], *identity, item['trace_sha256'])
        cohort_location = {**location, 'sentence_selection': copy.deepcopy(selection_doc)}
        membership = dict(id=item['cohort'], locations=[cohort_location],
                          sentence_selection=copy.deepcopy(selection_doc),
                          stratum=copy.deepcopy(item['stratum']), inclusion_probability=probability,
                          arm_inclusion_probability=arm_probability if item['cohort'] == 'B' else None)
        if item.get('annotation') is not None:
            membership['annotation'] = copy.deepcopy(item['annotation'])
        question_id = digest([trace['dataset'], trace.get('source_problem_id') or trace['problem_id']])
        core = dict(model=trace['model_id'], dataset=trace['dataset'], problem_id=trace['problem_id'],
                    source_problem_id=trace.get('source_problem_id'), sample_id=trace.get('sample_id', 0),
                    trace_id=digest([trace['model_id'], *identity]), question_id=question_id,
                    trace_sha256=item['trace_sha256'], completion_tokens=n,
                    termination=termination, capped=termination == 'length')
        row = merged.setdefault(aid, dict(id=aid, trace_sha256=item['trace_sha256'], model=trace['model_id'],
                                         dataset=trace['dataset'], identity=core,
                                         completion_tokens=n, total_input_tokens=len(prompt) + n,
                                         exact_token_replay=True, source_location=location, cohorts=[]))
        if row['identity'] != core:
            raise ValueError('Conflicting duplicate attempt metadata')
        if any(c['id'] == item['cohort'] for c in row['cohorts']):
            raise ValueError('Duplicate attempt membership in cohort')
        row['cohorts'].append(membership)
    if not merged:
        raise ValueError('Selected cohort is empty')
    return sorted(merged.values(), key=lambda row: row['id'])


def verification_pair(rows):
    """Choose the longest input and a distinct short differential-safe input.

    The short attempt must differ in completion tokens as well: the two accepted
    runtimes are the only points used to fit the fixed per-attempt plus
    per-token cost, so equal completion counts cannot identify both parameters.
    """
    longest = max(rows, key=lambda r: (r['total_input_tokens'], r['id']))
    shorts = [row for row in rows if 256 <= row['completion_tokens'] <= 4096
              and row['total_input_tokens'] <= 32768 and row['id'] != longest['id']
              and row['completion_tokens'] != longest['completion_tokens']]
    if not shorts:
        raise ValueError('Verification requires independent short (total input <=32768, distinct completion tokens) and longest attempts')
    # the longest eligible short: the HF/vLLM differential then covers >=~4k tokens of
    # position-dependent behaviour, and the runtime fit gets two well-separated points
    short = max(shorts, key=lambda r: (r['total_input_tokens'], r['id']))
    return [longest, short]


def bind_inline_annotations(rows, layouts, paths):
    """Freeze selected labels from shared A/B exports, with exact span checks."""
    from measurement_study.attach_labels import assemble_annotation, load_annotation_records
    from measurement_study.production_plan import read_location
    from measurement_study.contract import CLASSES
    records, inputs = load_annotation_records(paths)
    inputs = sorted({entry['path']: entry for entry in inputs}.values(), key=lambda entry: entry['path'])
    bindings = []
    for row in rows:
        units = layouts[row['id']]['units']
        for membership in row['cohorts']:
            source = read_location(membership['locations'][0])
            annotation = assemble_annotation(row, membership, records, source)
            selected = membership['sentence_selection']['indices']
            if len(annotation['units']) != len(selected) or {unit['index'] for unit in annotation['units']} != set(selected):
                raise ValueError('Inline annotation selection coverage mismatch')
            for unit in annotation['units']:
                expected = units[unit['index']]
                if any(unit.get(key) != value for key, value in expected.items()):
                    raise ValueError('Inline annotation source offsets/text changed')
                if unit.get('status') == 'unknown':
                    if unit.get('label') is not None:
                        raise ValueError('Unknown annotation has a canonical label')
                elif unit.get('label') not in CLASSES:
                    raise ValueError('Invalid inline class label')
            checksum = digest(annotation)
            membership['annotation'] = dict(inline=annotation, sha256=checksum, inputs=inputs)
            bindings.append((row['id'], membership['id'], checksum))
    return dict(inputs=inputs, annotations_sha256=digest(sorted(bindings)))


def explicit_runtime(startup_seconds, seconds_per_token, per_attempt_seconds):
    """Validate caller-supplied measured runtime for the fixed+per-token model."""
    values = dict(startup_seconds=startup_seconds, seconds_per_token=seconds_per_token,
                  per_attempt_seconds=per_attempt_seconds)
    for name, value in values.items():
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Measured ' + name + ' required for production shards')
    if values['startup_seconds'] < 0 or values['per_attempt_seconds'] < 0 or values['seconds_per_token'] <= 0:
        raise ValueError('Invalid measured runtime model')
    return dict(startup_seconds=float(values['startup_seconds']),
                seconds_per_token=float(values['seconds_per_token']),
                per_attempt_seconds=float(values['per_attempt_seconds']),
                points=[], method='explicit_measured_values', conservative=False)


def measured_runtime_from_receipt(receipt_path, validation_binding, maximum_input_tokens):
    """Fit the runtime model from the accepted verify-only receipt (two traces)."""
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('status') != 'passed' or not receipt.get('validation_binding'):
        raise ValueError('Accepted verification receipt required for production sizing')
    if receipt['validation_binding'] != validation_binding:
        raise ValueError('Verification receipt does not bind this spec/cohort/runtime envelope')
    if receipt.get('maximum_input_tokens') != maximum_input_tokens:
        raise ValueError('Verification receipt did not measure the corpus-longest input')
    measurements = receipt.get('measurements')
    if not isinstance(measurements, dict) or not measurements.get('attempts'):
        raise ValueError('Verification receipt has no measured attempt runtimes')
    startup = measurements.get('startup_seconds')
    if isinstance(startup, bool) or not isinstance(startup, (int, float)) or not math.isfinite(startup) or startup < 0:
        raise ValueError('Verification receipt has no measured startup')
    runtime = fit_attempt_runtime(measurements['attempts'], float(startup))
    footprint = measurements.get('bytes_per_token')
    if isinstance(footprint, bool) or not isinstance(footprint, (int, float)) or not math.isfinite(footprint) or footprint <= 0:
        raise ValueError('Verification receipt has no measured compressed bytes/token')
    return runtime, float(footprint), dict(path=str(receipt_path.resolve()), sha256=sha(receipt_path))


def build_manifests(*, cohort_path, spec, source, storage_root, code_binding,
                    selection='all', outcomes_path=None, outcome_manifest_path=None,
                    layout_fn=None, verification_only=False, verification_seconds=None,
                    annotation_paths=None, p90_seconds_per_token=None, startup_seconds=None,
                    per_attempt_seconds=None, bytes_per_token=None, verification_receipt_path=None):
    """Build selected production shards and one shared, label-free model verifier.

    Capture binds ONLY the cohort manifest, the cohort trace bytes and the code
    binding. `outcomes_path` (sidecar or pointer) is an optional planning-time
    consistency reference; it is never written into a manifest or receipt, so
    cohorts A and B share one verifier and outcomes may be versioned later.
    Production shards size with the measured fixed-plus-per-token runtime taken
    from `verification_receipt_path` (or explicit measured values).
    """
    storage = validate_storage_root(storage_root)
    spec = prepare_spec(source, spec)
    cohort_path = Path(cohort_path)
    cohort, cohort_binding = load_cohort_manifest(cohort_path)
    if selection not in {'A', 'B', 'all'}:
        raise ValueError('Unknown cohort selection')
    outcomes = []
    if outcomes_path is not None:
        outcomes, _reference = read_outcomes(outcomes_path, outcome_manifest_path,
                                             corpus='card-v3', model=cohort['model'])
    # One verification input universe for the model, even when B is planned first.
    all_rows = build_rows(cohort, outcomes, source, spec, 'all')
    duplicate_report = cohort_duplicate_report(cohort)
    if layout_fn is None:
        from measurement_study.production_plan import read_location
        from measurement_study.sentence_schema import install
        install()
        from moe_exp.schemas import TraceRecord
        from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
        def layout_fn(row):
            return saved_layout(TraceRecord(**read_location(row['source_location'])))
    corpus, layouts = [], {}
    for row in all_rows:
        layout = layout_fn(row)
        if layout.get('reasoning_tokens') is None or layout.get('token_count') != row['completion_tokens']:
            raise ValueError('Missing exact saved token/character alignment')
        layouts[row['id']] = layout
        row['reasoning_tokens'] = len(layout['reasoning_tokens'])
        units = layout['units']
        for membership in row['cohorts']:
            if any(i >= len(units) for i in membership['sentence_selection']['indices']):
                raise ValueError('Cohort sentence selection outside source')
            # The external cohort schema has no labels. The shared verifier never
            # consumes an optional legacy annotation extension from its input.
            membership.pop('annotation', None)
        corpus.append((row['dataset'], row['identity']['problem_id'], row['trace_sha256'], row['reasoning_tokens']))
    reference = dict(model_id=spec['generation_model'], mean_reasoning_tokens=sum(r['reasoning_tokens'] for r in all_rows)/len(all_rows),
                     traces=len(all_rows), bins=10, corpus_sha256=digest(sorted(corpus)),
                     rule='ceil(model_corpus_mean * i / bins); noncumulative; overflow after mean',
                     population='deduplicated union of cohort A and B', limit_applied_to_reference=False)
    rows = []
    for original in all_rows:
        row = copy.deepcopy(original)
        row['cohorts'] = [member for member in row['cohorts'] if selection == 'all' or member['id'] == selection]
        if row['cohorts']:
            rows.append(row)
    if not rows:
        raise ValueError('Selected cohort is empty')
    if verification_only and annotation_paths:
        raise ValueError('Bootstrap verification is label-free; omit --annotations')
    annotation_binding = None
    if not verification_only:
        if any(member['id'] == 'A' for row in rows for member in row['cohorts']) and not annotation_paths:
            raise ValueError('Cohort A production requires --annotations')
        if annotation_paths:
            annotation_binding = bind_inline_annotations(rows, layouts, annotation_paths)
    maximum = max(r['total_input_tokens'] for r in all_rows)
    validation_binding = digest(dict(spec=spec, code_binding=code_binding,
                                     cohort_binding=cohort_binding, maximum_input_tokens=maximum))
    receipt_binding = None
    if verification_only:
        if type(verification_seconds) is not int or not 0 < verification_seconds <= 16*3600 or verification_seconds % 1800:
            raise ValueError('Explicit verification walltime must be a positive half-hour up to 16 hours')
        if verification_seconds < MIN_VERIFICATION_SECONDS:
            raise ValueError('Verification walltime below the 5-hour envelope for a 131k trace plus reference')
        footprint = None
        runtime = None
        groups = []
        sizing = dict(status='unmeasured_verification', runtime_factor=1.25, max_shard_hours=16,
                      drain_margin_seconds=UNMEASURED_DRAIN_MARGIN_SECONDS,
                      artifact_cap_rule='sum conservative native snapshots plus metric headroom; no measured compression claim')
    else:
        if verification_receipt_path is not None:
            if any(value is not None for value in (startup_seconds, per_attempt_seconds, bytes_per_token, p90_seconds_per_token)):
                raise ValueError('Pass either the verification receipt or explicit measured values, not both')
            runtime, footprint, receipt_binding = measured_runtime_from_receipt(
                verification_receipt_path, validation_binding, maximum)
        else:
            runtime = explicit_runtime(startup_seconds, p90_seconds_per_token, per_attempt_seconds)
            footprint = float(bytes_per_token) if bytes_per_token is not None else None
            if footprint is None or not math.isfinite(footprint) or footprint <= 0:
                raise ValueError('Measured compressed bytes/token required')
        groups = token_budget_shards(rows, runtime['seconds_per_token'], runtime['startup_seconds'],
                                     runtime['per_attempt_seconds'])
        sizing = dict(status='measured', startup_seconds=runtime['startup_seconds'],
                      seconds_per_token=runtime['seconds_per_token'],
                      per_attempt_seconds=runtime['per_attempt_seconds'],
                      runtime_model=runtime, p90_seconds_per_token=p90_seconds_per_token,
                      runtime_factor=1.25, max_shard_hours=16,
                      drain_margin_seconds=MIN_DRAIN_MARGIN_SECONDS,
                      compressed_bytes_per_token=footprint)
    base = dict(schema_version=3, source=source, cohort=selection, corpus='card-v3', spec=spec,
                storage_root=str(storage), cohort_binding=cohort_binding, cohort_model=cohort['model'],
                cohort_manifest_path=str(cohort_path.resolve()), dedup=duplicate_report,
                code_binding=code_binding, cohorts_separate=True, position_reference=reference,
                maximum_input_tokens=maximum, annotation_binding=annotation_binding, sizing=sizing)
    base['validation_binding'] = validation_binding
    base['verification_receipt_binding'] = receipt_binding
    base['verification_path'] = str(storage/'capture'/source/'verify'/'verification.json')
    base['memory_limit_bytes'] = 240*1024**3

    def manifest(index, attempts, seconds, *, shared_verifier=False):
        longest = max(r['completion_tokens'] for r in attempts)
        from measurement_study.capture_buffers import capture_memory_budget
        memory = capture_memory_budget(longest, len(spec['layers']), spec['hidden_size'], spec['experts'], spec['top_k'])
        if footprint is None:
            artifact_cap = sum(capture_memory_budget(r['completion_tokens'], len(spec['layers']), spec['hidden_size'],
                               spec['experts'], spec['top_k'])['transient_bytes'] for r in attempts)
            artifact_cap = math.ceil(artifact_cap*1.25) + 64*1024**2
        else:
            artifact_cap = math.ceil(sum(r['completion_tokens'] for r in attempts)*footprint*1.25) + 64*1024**2
        m = dict(base, shard=index, attempts=attempts, expected_attempts=len(attempts),
                 expected_tokens=sum(r['completion_tokens'] for r in attempts), requested_seconds=seconds,
                 capture_dir=str(storage/'capture'/source/selection/str(index)), artifact_cap_bytes=artifact_cap,
                 memory_budget=memory, estimated_snapshot_bytes=memory['native_tensor_bytes'],
                 transient_cap_bytes=memory['transient_bytes'], buffer_cap_bytes=memory['worker_bytes'],
                 seconds_per_token=None if verification_only else runtime['seconds_per_token'],
                 per_attempt_seconds=None if verification_only else runtime['per_attempt_seconds'])
        if shared_verifier:
            m.update(cohort='all', annotation_binding=None, capture_dir=str(storage/'capture'/source/'verify'))
        m['binding'] = digest(m)
        return m

    manifests = [manifest(i, group['attempts'], group['requested_seconds']) for i, group in enumerate(groups)]
    verify_rows = verification_pair(all_rows)
    if verification_only:
        verify_seconds = verification_seconds
    else:
        expected = 1.25*(runtime['startup_seconds'] + len(verify_rows)*runtime['per_attempt_seconds']
                         + runtime['seconds_per_token']*sum(r['completion_tokens'] for r in verify_rows))
        verify_seconds = max(MIN_VERIFICATION_SECONDS, math.ceil(expected/1800)*1800)
    if verify_seconds > 16*3600:
        raise ValueError('Verification pair exceeds runtime envelope')
    verification = manifest('verify', verify_rows, verify_seconds, shared_verifier=True)
    ready = dict(base, status='needs_measurement' if verification_only else 'ready', attempts=len(rows), tokens=sum(r['completion_tokens'] for r in rows),
                 verification_manifest='../verify.json',
                 shards=[dict(index=m['shard'], attempts=m['expected_attempts'], tokens=m['expected_tokens'], requested_seconds=m['requested_seconds'], binding=m['binding']) for m in manifests])
    return manifests, verification, ready


def publish_manifests(storage_root, source, selection, manifests, verification, ready):
    model_root = validate_storage_root(storage_root)/'manifests'/source
    root = model_root/selection
    outputs = {str(m['shard']) + '.json': m for m in manifests}
    outputs['ready.json'] = ready
    previous_ready = json.loads((root/'ready.json').read_text()) if (root/'ready.json').exists() else None
    upgrading = (previous_ready is not None and previous_ready.get('status') == 'needs_measurement'
                 and ready['status'] == 'ready' and previous_ready.get('validation_binding') == ready['validation_binding'])
    verifier_path = model_root/'verify.json'
    if verifier_path.exists():
        previous = json.loads(verifier_path.read_text())
        if (digest({k:v for k,v in previous.items() if k != 'binding'}) != previous.get('binding')
                or previous.get('validation_binding') != verification['validation_binding']
                or previous.get('attempts') != verification['attempts']
                or previous.get('annotation_binding') is not None
                or previous.get('cohort') != 'all'
                or previous.get('capture_dir') != verification['capture_dir']):
            raise ValueError('Prior shared verification manifest cannot be rebound')
        # Its startup estimate/walltime may predate measurement. It remains the
        # immutable parent of the one verification receipt shared by A and B.
    for name, value in outputs.items():
        path = root/name
        if not path.exists():
            continue
        previous = json.loads(path.read_text())
        if previous == value or (name == 'ready.json' and upgrading):
            continue
        raise ValueError('Existing campaign-v3 manifest differs; choose a new root')
    if root.exists() and any(p.name not in outputs for p in root.glob('*.json')):
        raise ValueError('Unexpected prior manifest/shard in output root')
    if not verifier_path.exists():
        atomic(verifier_path, verification)
    for name, value in outputs.items():
        if not (root/name).exists() or (name == 'ready.json' and upgrading):
            atomic(root/name, value)
    return root
