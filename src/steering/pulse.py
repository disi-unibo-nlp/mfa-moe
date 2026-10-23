"""Reduced final-routing driver: one engine per model shard with continuous admission.

Design (debate round 3, 2026-10-17): eight models, five arms, central count, one
discovery-selected layer per fold/class. Every assigned continuation runs to EOS
or its original allowance (`runtime.remaining`) in one hooked eager engine that
admits requests as others finish. A request still running at the allocation
deadline is checkpointed with all its tokens and resumed in a later job by
re-prefilling prompt + prefix + generated tokens through the same hooks, so the
pulse is re-applied at its original positions. Nothing is redrawn.

Commands (python -m steering.pulse ...):
  plan                          CPU: selection + reduced manifests for prepared models
  plan-strong --model M,M       CPU: the post-hoc strong-dose manifests
  run --model M --shard I/N     GPU: assigned continuations of one shard (--design main|strong)
  native --shard I/N            GPU: LFM native population (unhooked engine)
  annotate --shard I/N          GPU: LFM discovery judge labels (unhooked engine)
  lfm-materialize | lfm-derive  CPU: labeling.lfm_inputs / steering.nominations
  lfm-readout | lfm-dynamics    CPU: LFM native readout (routing.lfm_readout) and native-dynamics panel
  lfm-replay --shard I/N        GPU: capture.lfm_replay (Backend.capture_trace)
  status                        login: dispositions per model
"""
from __future__ import annotations
import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from provenance import code_provenance
from store import CLASSES, MODELS, REPO, digest, file_sha, load, private, publish, require, verified

ROOT = Path(os.environ.get('STUDY_DATA_ROOT', REPO/'report/final-routing-20261015-r1'))
DRIVER = ROOT/'driver'
SELECTOR = 'one-layer-per-fold-class: training association, then stability, then layer ID'


def binding():
    return load(ROOT/'BINDING.json')


# Additive float telemetry; every routing decision and integer counter must still agree exactly.
FLOAT_DOSE_KEYS = ('native_target_weight', 'executed_target_weight', 'native_activate_weight',
                   'executed_activate_weight', 'native_deactivate_weight', 'executed_deactivate_weight',
                   'native_activate_share', 'executed_activate_share', 'l1',
                   'native_shift_weight', 'executed_shift_weight')


def ranks_agree(first, second, rtol=1e-6):
    """Compare two tensor-parallel release payloads.

    Args:
        first, second: Per-rank `release()` payloads for one request.
        rtol: Relative tolerance for float dose telemetry only.

    Returns:
        True when audit routing, integer counters and stream state match exactly and float
        telemetry matches within `rtol`.
    """
    import math

    def exact(capture):
        dose = {layer: {k: v for k, v in counters.items() if k not in FLOAT_DOSE_KEYS}
                for layer, counters in capture['stats'].get('dose', {}).items()}
        return {**{k: v for k, v in capture.items() if k not in ('sha256', 'stats')},
                'stats': {**capture['stats'], 'dose': dose}}
    if exact(first) != exact(second):
        return False
    return all(math.isclose(counters[k], second['stats']['dose'][layer][k], rel_tol=rtol, abs_tol=1e-6)
               for layer, counters in first['stats'].get('dose', {}).items()
               for k in FLOAT_DOSE_KEYS if k in counters)


def bulk(b=None):
    return Path((b or binding())['config']['bulk_root'])/'driver'


def checkpoint(path, value):
    """Atomic overwrite for mutable progress records (never for results)."""
    path = private(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(prefix='.partial-', dir=path.parent)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- selection

def select_layers(nominations):
    """One layer per (fold, class) among the regional winners; no confirmation or outcome data."""
    best, evidence = {}, []
    for n in nominations:
        if n['status'] != 'nominated':
            continue
        key = (n['fold'], n['class'])
        rank = (-n['association'], -n['stability'], n['layer'])
        evidence.append({'fold': n['fold'], 'class': n['class'], 'depth_third': n['depth_third'],
                         'layer': n['layer'], 'association': n['association'], 'stability': n['stability']})
        if key not in best or rank < best[key][0]:
            best[key] = (rank, n['depth_third'], n['layer'])
    return {k: {'depth_third': v[1], 'layer': v[2]} for k, v in best.items()}, evidence


def reduce_rows(rows, chosen):
    """Native rows plus central-count rows at the selected depth; unavailable prefixes kept apart."""
    layers_by_fold = {}
    for (fold, _), v in chosen.items():
        layers_by_fold.setdefault(fold, set()).add(v['layer'])
    kept, unavailable = [], []
    for r in rows:
        if r['arm'] != 'native':
            pick = chosen.get((r['fold'], r['class']))
            if r['dose'] != 'central' or pick is None or r['depth_third'] != pick['depth_third']:
                continue
            require(r['action'] is None or r['action']['layer'] == pick['layer'], 'selected layer differs from assignment')
        if r['source'] is None or r['prefix_status'] != 'enrolled' or (r['arm'] != 'native' and r['action'] is None):
            unavailable.append(r['uid'])
            continue
        capture = sorted(layers_by_fold.get(r['fold'], ())) if r['arm'] == 'native' else [r['action']['layer']]
        kept.append({**r, 'capture_layers': capture})
    kept.sort(key=lambda r: (r['family'], r['arm'], r['class'] or '', r['uid']))
    return kept, unavailable


def plan(models=None):
    out = {}
    for model in models or [m for m in MODELS if (ROOT/m/'ASSIGNMENTS-full.json').exists()]:
        source = verified(ROOT/model/'ASSIGNMENTS-full.json')
        nominations = verified(ROOT/model/'COUNT_NOMINATIONS.json')
        chosen, evidence = select_layers(nominations['records'])
        require(len(chosen) == 35, f'{model}: a fold/class lacks a nominated layer')
        rows, unavailable = reduce_rows(source['records'], chosen)
        arms = {}
        for r in rows:
            arms[r['arm']] = arms.get(r['arm'], 0)+1
        value = publish(DRIVER/model/'ASSIGNMENTS.json', {
            'model': model, 'selector': SELECTOR, 'design': {'dose': 'central', 'arms': arms},
            'selection': [{'fold': f, 'class': c, **v} for (f, c), v in sorted(chosen.items())],
            'selection_evidence': evidence, 'outcomes_used': False,
            'source_assignments_sha256': source['sha256'], 'nominations_sha256': nominations['sha256'],
            'unavailable': unavailable, 'records': rows})
        out[model] = {'executable': len(rows), 'unavailable': len(unavailable), 'arms': arms, 'sha256': value['sha256']}
    return out


MANIFESTS = {'main': 'ASSIGNMENTS.json', 'strong': 'STRONG.json'}
STORES = {'main': 'units', 'strong': 'strong/units'}


def plan_strong(models, seed_tag='strong-dose-v1'):
    """Post-hoc dose escalation (decided 2026-10-17 after the null pre-registered preview; labelled exploratory).
    For each (model, family, class): the pre-registered HIGH-count nominated expert sets of all three depth thirds
    (COUNT_NOMINATIONS, training folds only) applied together, held for the whole 16·Q75 window; matched control =
    the pre-registered high-count control sets of the same layers. Same prefix and seed as the family's native row
    (common random numbers)."""
    out = {}
    for model in models:
        nominations = verified(ROOT/model/'COUNT_NOMINATIONS.json')
        by = {}
        for n in nominations['records']:
            if n['status'] == 'nominated' and n['expert_sets'][2] and n['control_sets'][2]:
                by.setdefault((n['fold'], n['class']), []).append(n)
        natives = [r for r in verified(DRIVER/model/'ASSIGNMENTS.json')['records'] if r['arm'] == 'native']
        rows = []
        for native in natives:
            for c in CLASSES:
                noms = sorted(by.get((native['fold'], c), []), key=lambda n: n['depth_third'])
                require(noms and all(n['horizon'] == native['horizon'] for n in noms), 'fold horizon differs')
                for arm, key in (('strong_target_activate', 'expert_sets'), ('strong_random_activate', 'control_sets')):
                    actions = [{'direction': 'activate', 'experts': sorted(n[key][2]), 'layer': n['layer'],
                                'pulse_tokens': native['horizon'], 'start': 0} for n in noms]
                    row = {k: native[k] for k in ('benchmark', 'family', 'fold', 'model', 'horizon', 'seed', 'source',
                                                  'profile_sha256', 'prefix_status')}
                    row.update({'class': c, 'arm': arm, 'dose': 'high-all-depths-window', 'action': None,
                                'actions': actions, 'pulse_tokens': native['horizon'], 'depth_third': None,
                                'nomination_status': 'nominated', 'replicate': 0,
                                'capture_layers': sorted(a['layer'] for a in actions)})
                    row['uid'] = digest([seed_tag, row])
                    rows.append(row)
        rows.sort(key=lambda r: (r['family'], r['arm'], r['class'], r['uid']))
        counts = {}
        for r in rows:
            counts[r['arm']] = counts.get(r['arm'], 0)+1
        value = publish(DRIVER/model/'STRONG.json', {'model': model, 'design': seed_tag, 'post_hoc': True,
            'decided_after': 'preview of pre-registered primary contrasts (gpt, nemotron, qwen330b): null',
            'arms': counts, 'nominations_sha256': nominations['sha256'], 'records': rows})
        out[model] = {'rows': len(rows), 'arms': counts, 'sha256': value['sha256']}
    return out


def shard_rows(rows, shard):
    """The rows of the families with rank % count == index, families ranked by name."""
    index, count = shard
    families = sorted({r['family'] for r in rows})
    mine = {f for i, f in enumerate(families) if i % count == index}
    return [r for r in rows if r['family'] in mine]


def design_rows(model, design, shard):
    """One shard of a model's rows in the main or strong design."""
    return shard_rows(verified(DRIVER/model/MANIFESTS[design])['records'], shard)


# ----------------------------------------------------------------------------- engine

class Store:
    """Per-uid durable state: PARTIAL (mutable progress) -> OUTPUT/CAPTURE -> RESULT (immutable)."""
    def __init__(self, root):
        self.root = private(root)

    def folder(self, uid):
        require(len(uid) == 64 and all(c in '0123456789abcdef' for c in uid), 'invalid execution ID')
        return self.root/uid

    def state(self, uid):
        f = self.folder(uid)
        if (f/'RESULT.json').exists():
            return verified(f/'RESULT.json')
        if (f/'OUTPUT.json').exists():  # crashed between output and result: finalize, never re-admit
            return {'status': 'output'}
        if (f/'PARTIAL.json').exists():
            return {'status': 'partial', **load(f/'PARTIAL.json')}
        return None


def deadline_from_slurm(margin=420):
    import subprocess
    end = subprocess.run(['squeue', '-h', '-j', os.environ['SLURM_JOB_ID'], '-o', '%e'], capture_output=True,
                         text=True, check=True).stdout.strip()
    from datetime import datetime
    return datetime.strptime(end, '%Y-%m-%dT%H:%M:%S').timestamp()-margin


class Engine:
    """Continuous admission over a synchronous vLLM engine with durable resumable state."""

    def __init__(self, llm, store, deadline, *, hooked, max_inflight, checkpoint_seconds=600, drain_seconds=900):
        self.llm, self.engine, self.store = llm, llm.llm_engine, store
        self.deadline, self.hooked, self.max_inflight = deadline, hooked, max_inflight
        self.checkpoint_seconds, self.drain_seconds = checkpoint_seconds, drain_seconds
        self.inflight, self.job = {}, os.environ.get('SLURM_JOB_ID', 'local')
        cache = getattr(getattr(llm.llm_engine, 'vllm_config', None), 'cache_config', None)
        blocks = getattr(cache, 'num_gpu_blocks', None)
        self.kv_tokens = blocks*cache.block_size if blocks else None
        self.metrics = {'admitted': 0, 'resumed': 0, 'finished': 0, 'drained': 0, 'generated_tokens': 0,
                        'prefill_tokens': 0, 'steps': 0, 'step_seconds': 0., 'peak_inflight': 0,
                        'mixed_eight_verified': False}
        require(not self.engine.has_unfinished_requests(), 'engine has neighboring requests')

    def admit(self, row, build):
        """build(row, history, segment) -> dict(prompt_ids, params, request, total) or None when complete."""
        uid = row['uid']
        state = self.store.state(uid)
        require(state is None or state['status'] == 'partial', 'admitting a terminal unit')
        history = list(state['tokens']) if state else []
        segments = list(state['segments']) if state else []
        # Seed index = number of distinct history lengths reached before this admission: a segment drained
        # before emitting a token keeps the same seed, so common random numbers survive empty interruptions.
        spec = build(row, history, len({x['history'] for x in segments} | {len(history)})-1)
        if spec is None:
            return False
        if self.kv_tokens and self.inflight:
            used = sum(st['context'] for st in self.inflight.values())
            if used+len(spec['prompt_ids'])+2048 > .85*self.kv_tokens:
                return 'deferred'
        segments.append({'job': self.job, 'admitted': time.time(), 'history': len(history),
                         'seed': spec['params'].seed, 'max_tokens': spec['params'].max_tokens})
        progress = {'uid': uid, 'assignment_sha256': digest(row), 'tokens': history, 'segments': segments,
                    'total_allowance': spec['total']}
        checkpoint(self.store.folder(uid)/'PARTIAL.json', progress)
        internal = self.engine.add_request(uid, {'prompt_token_ids': spec['prompt_ids']}, spec['params'])
        require(isinstance(internal, str) and internal, 'vLLM returned no internal ID')
        if self.hooked:
            reports = self.llm.collective_rpc('fs_add', kwargs={'requests': [spec['request']], 'aliases': {internal: uid}})
            require(len(reports) == 2 and len({r['mapping_sha256'] for r in reports}) == 1,
                    'tensor-parallel request maps disagree')
        self.inflight[uid] = {'row': row, 'progress': progress, 'internal': internal, 'new': 0,
                              'context': len(spec['prompt_ids'])}
        self.metrics['admitted'] += 1
        self.metrics['peak_inflight'] = max(self.metrics['peak_inflight'], len(self.inflight))
        active_rows = [state['row'] for state in self.inflight.values()]
        if len(active_rows) == 8:
            self.metrics['mixed_eight_verified'] |= (
                len({row.get('family') for row in active_rows}) > 1
                and len({row.get('arm') for row in active_rows}) > 1
                and any(row.get('action') or row.get('actions') for row in active_rows))
        self.metrics['resumed'] += bool(history)
        self.metrics['prefill_tokens'] += len(spec['prompt_ids'])
        return True

    def release(self, uids):
        if not self.hooked or not uids:
            return {}
        reports = self.llm.collective_rpc('fs_release', kwargs={'uids': list(uids)})
        require(len(reports) == 2, 'routing release lacks both tensor-parallel ranks')
        # fs_release consumes worker state, including when comparison below fails.
        for uid in uids:
            self.inflight[uid]['capture_released'] = True
        for uid in uids:
            if len({r[uid]['sha256'] for r in reports}) != 1:
                if ranks_agree(reports[0][uid], reports[1][uid]):
                    self.metrics['float_telemetry_rank_differences'] = self.metrics.get(
                        'float_telemetry_rank_differences', 0)+1
                    log({'event': 'tp-float-telemetry-difference', 'uid': uid})
                    continue
                folder = self.store.folder(uid)/'TP_DISAGREEMENT'
                for rank, report in enumerate(reports):
                    capture = report[uid]
                    saved = compress_audit_capture(folder/str(rank), capture)
                    publish(folder/str(rank)/'CAPTURE.json',
                            {**saved, 'routing_capture_sha256': capture['sha256']})
                raise ValueError(f'tensor-parallel captures disagree: {folder}')
        return reports[0]

    def save_progress(self):
        for uid, st in self.inflight.items():
            checkpoint(self.store.folder(uid)/'PARTIAL.json', st['progress'])

    def step(self, finish):
        tick = time.monotonic()
        outputs = self.engine.step()
        self.metrics['steps'] += 1
        self.metrics['step_seconds'] += time.monotonic()-tick
        done = []
        for response in outputs:
            uid = response.request_id
            require(uid in self.inflight and len(response.outputs) == 1, 'response attribution differs')
            out = response.outputs[0]
            st = self.inflight[uid]
            st['progress']['tokens'].extend(out.token_ids)
            if out.logprobs:
                st['progress'].setdefault('logprobs', []).extend(
                    {str(t): float(v.logprob) for t, v in step.items()} for step in out.logprobs)
            st['new'] += len(out.token_ids)
            st['context'] += len(out.token_ids)
            self.metrics['generated_tokens'] += len(out.token_ids)
            if response.finished:
                done.append((uid, out.finish_reason, out.stop_reason))
        if done:
            captures = self.release([u for u, _, _ in done])
            for uid, reason, stop in done:
                st = self.inflight.pop(uid)
                finish(st['row'], st['progress'], reason, stop, captures.get(uid))
                self.metrics['finished'] += 1

    def drain(self):
        uids = list(self.inflight)
        if uids:
            self.save_progress()  # persist every token before touching the engine
            self.engine.abort_request(uids, internal=False)
            self.release([uid for uid in uids if not self.inflight[uid].get('capture_released')])
            for uid in uids:
                st = self.inflight.pop(uid)
                st['progress']['segments'][-1].update(drained=time.time(), generated=st['new'])
                checkpoint(self.store.folder(uid)/'PARTIAL.json', st['progress'])
                self.metrics['drained'] += 1
        require(not self.engine.has_unfinished_requests(), 'abort left unfinished engine requests')

    def run(self, rows, build, finish, *, stop_admission=None, stop_execution=None):
        queue = [r for r in rows if (self.store.state(r['uid']) or {'status': 'partial'})['status'] == 'partial']
        # Resume partial units first: they hold the longest outstanding work.
        queue.sort(key=lambda r: self.store.state(r['uid']) is None)
        last_save, started = time.time(), time.time()
        try:
            while queue or self.inflight:
                now = time.time()
                if now > self.deadline or (stop_execution is not None and stop_execution()):
                    break
                admitting = now < self.deadline-self.drain_seconds and (stop_admission is None or not stop_admission())
                while admitting and queue and len(self.inflight) < self.max_inflight:
                    row = queue.pop(0)
                    admitted = self.admit(row, build)
                    if admitted == 'deferred':
                        queue.insert(0, row)
                        self.metrics['deferred'] = self.metrics.get('deferred', 0)+1
                        break
                    if not admitted:
                        finish(row, None, None, None, None)
                if not self.inflight:
                    if not admitting:
                        break
                    continue
                self.step(finish)
                if time.time()-last_save > self.checkpoint_seconds:
                    self.save_progress()
                    last_save = time.time()
                    elapsed = max(1., last_save-started)
                    log({'event': 'progress', 'inflight': len(self.inflight), 'queue': len(queue),
                         'tokens_per_second': self.metrics['generated_tokens']/elapsed,
                         **{k: self.metrics[k] for k in ('admitted', 'resumed', 'finished', 'generated_tokens')}})
        except BaseException:
            try:
                self.drain()
            except Exception as error:
                log({'event': 'engine-error-cleanup-failed', 'error': str(error)})
            raise
        else:
            self.drain()
        self.metrics['wall_seconds'] = time.time()-started
        self.metrics['remaining_queue'] = len(queue)
        return self.metrics


# ----------------------------------------------------------------------------- interventions

def contract_for(b, model, rows):
    p = b['models'][model]
    hook = sorted({l for r in rows for l in r['capture_layers']} |
                  {a['layer'] for r in rows for a in (r.get('actions') or ([r['action']] if r.get('action') else []))})
    return {'layers': p['layers'], 'native_k': p['native_k'], 'experts': p['experts'], 'hook_layers': hook}


def hooked_llm(b, model, max_num_seqs, *, batched_tokens=8192, memory=.88):
    from vllm import LLM
    from models.engine import engine_kwargs
    from store import check_files
    p = b['models'][model]
    check_files(p['config_files'])
    require((Path(p['local_path'])/'config.json').is_file(), 'pinned checkpoint is unavailable')
    kwargs = engine_kwargs(p, context=p['card']['max_model_len'])
    kwargs.update(max_num_seqs=max_num_seqs, enable_chunked_prefill=True, max_num_batched_tokens=batched_tokens,
                  gpu_memory_utilization=memory, worker_extension_cls='steering.stream.StreamWorkerExtension',
                  logits_processors=['steering.fast_penalties:FastPrefixPenalties'])
    if model == 'lfm25':
        kwargs['kv_cache_memory_bytes'] = p['kv_cache_memory_bytes']
        kwargs.pop('gpu_memory_utilization')
    tick = time.monotonic()
    llm = LLM(**kwargs)
    require(type(llm.llm_engine.engine_core).__name__ == 'InprocClient', 'unqualified executor')
    return llm, kwargs, time.monotonic()-tick


def segment_seed(seed, segment):
    return seed if segment == 0 else int(digest([seed, 'continuation-segment', segment])[:8], 16)


def sampling(profile, seed, history, cap, *, greedy=False, logprobs=None):
    """runtime.sampling_parameters with streamed token deltas and no detokenization."""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    sampler = {k: v for k, v in profile['sampler'].items() if v is not None}
    if sampler.get('top_k') == 0:
        sampler['top_k'] = -1
    penalties = {'research_prefix_penalties': {'prefix': list(history), 'presence': sampler.get('presence_penalty', 0.),
                 'frequency': sampler.get('frequency_penalty', 0.), 'repetition': sampler.get('repetition_penalty', 1.)}}
    sampler.update(presence_penalty=0., frequency_penalty=0., repetition_penalty=1.)
    if greedy:
        sampler.update(temperature=0., top_p=1., top_k=-1, min_p=0.)
    return SamplingParams(**sampler, seed=seed, max_tokens=cap, skip_special_tokens=False, extra_args=penalties,
                          detokenize=False, logprobs=logprobs, output_kind=RequestOutputKind.DELTA)


def intervention_builder(b, model, tokenizer, closure, *, greedy=False, limit=None, logprobs=None):
    from steering.runtime import prompt_prefix, remaining
    p = b['models'][model]
    cache = {}

    def build(row, history, segment):
        # A length-limit (budget) row continues a cut copy of a native run, so its prefix is its own.
        key = (row['source']['attempt_id'], row['source']['prefix_tokens'], digest(row.get('budget_prefix')))
        if key not in cache:
            cache.clear()
            cache[key] = prompt_prefix(b, model, row, tokenizer)[:2]
        prompt, prefix = cache[key]
        total = remaining(p, prompt, prefix)
        if row.get('budget_prefix'):
            total = min(total, int(row['horizon']))
        if limit is not None:
            total = min(total, limit)
        left = total-len(history)
        if left <= 0:
            return None
        params = sampling(p, segment_seed(row['seed'], segment), prefix+history, left, greedy=greedy, logprobs=logprobs)
        request = {'uid': row['uid'], 'prompt': prompt, 'prefix': prefix, 'closure': closure,
                   'action': row['action'], 'actions': row.get('actions'), 'capture_layers': row['capture_layers'],
                   'window': row['horizon']}
        if row.get('whole'):
            request.update(whole=True, prompt_scope=row['prompt_scope'], audit_capture=row['audit_capture'])
        return {'prompt_ids': prompt+prefix+history, 'params': params, 'request': request, 'total': total}
    return build


def work_remaining_bytes() -> int:
    """Room left under the WORK project quota (the smaller of its soft and hard limits)."""
    import subprocess
    result = subprocess.run(['lfs', 'quota', '-p', '20149688', '/leonardo_work'],
                            capture_output=True, text=True, check=True)
    fields = next(line.split() for line in result.stdout.splitlines() if line.strip().startswith('/leonardo_work '))
    used, soft, hard = (int(value.rstrip('*')) for value in fields[1:4])
    limits = [value for value in (soft, hard) if value]
    require(limits, 'native WORK project quota limit is unavailable')
    return (min(limits)-used)*1024


def compress_audit_capture(folder: Path, capture: dict) -> dict:
    """Keep full audit arrays losslessly without per-token JSON expansion.

    Args:
        folder: This unit's private output directory.
        capture: Tensor-parallel-agreed full routing capture.

    Returns:
        Capture receipt with a hashed compressed-array reference.
    """
    import numpy as np
    arrays = {}
    for layer, record in capture['capture'].items():
        for key, values in record.items():
            dtype = (np.uint8 if key.endswith('_ids') else np.float32 if key.endswith('_weights')
                     else np.bool_ if key == 'active' else np.int32)
            arrays['layer_'+layer+'_'+key] = np.asarray(values, dtype=dtype)
    require(work_remaining_bytes() > 13*2**30+sum(array.nbytes for array in arrays.values()),
            'full audit would breach the existing WORK reserve and concurrent-write margin')
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder/'AUDIT.npz'
    require(not destination.exists(), 'whole audit capture already exists')
    with destination.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    return {**capture, 'capture': {}, 'full_capture': {
        'path': str(destination), 'sha256_file': file_sha(destination),
        'bytes': destination.stat().st_size, 'format': 'numpy-compressed-exact-float32-uint8'}}


def intervention_finisher(store, b, model):
    """Bind the existing output publication callback to this model and store."""
    def finish(row, progress, reason, stop, capture):
        """Commit a completed draw and its agreed routing telemetry atomically by receipt."""
        uid = row['uid']
        folder = store.folder(uid)
        if progress is None:  # allowance already exhausted by saved tokens
            progress = load(folder/'PARTIAL.json')
            reason = 'length'
        raw = publish(folder/'OUTPUT.json', {'uid': uid, 'assignment_sha256': digest(row),
            'token_ids': progress['tokens'], 'finish_reason': reason, 'stop_reason': stop,
            'segments': progress['segments'], 'total_allowance': progress['total_allowance'],
            'profile_sha256': b['models'][model]['profile_sha256']})
        status, why = 'complete', None
        capture_valid = capture is not None and (
            all(str(l) in capture['stats']['dose'] for l in row['capture_layers']) if row.get('whole') else
            all(str(l) in capture['capture'] and capture['capture'][str(l)]['positions'] for l in row['capture_layers']))
        if not capture_valid:
            status, why = 'unknown', 'required_routing_capture_missing'
        else:
            if row.get('whole') and row.get('audit_capture'):
                capture = compress_audit_capture(folder, capture)
            publish(folder/'CAPTURE.json', {'uid': uid, 'raw_sha256': raw['sha256'], **capture})
        publish(folder/'RESULT.json', {'uid': uid, 'assignment_sha256': digest(row), 'status': status,
            'reason': why, 'raw_sha256': raw['sha256'], 'generated_tokens': len(progress['tokens']),
            'finish_reason': reason, 'segments': len(progress['segments']), 'stochastic_redraw': False})
    return finish


def recover_outputs(store, rows):
    """Finalize units whose OUTPUT was published but whose RESULT was not (crash in between)."""
    recovered = 0
    for row in rows:
        if (store.state(row['uid']) or {}).get('status') != 'output':
            continue
        folder = store.folder(row['uid'])
        raw = verified(folder/'OUTPUT.json')
        capture = folder/'CAPTURE.json'
        ok = capture.exists() and verified(capture).get('raw_sha256') == raw['sha256']
        publish(folder/'RESULT.json', {'uid': row['uid'], 'assignment_sha256': digest(row),
            'status': 'complete' if ok else 'unknown', 'reason': None if ok else 'routing_capture_missing_after_crash',
            'raw_sha256': raw['sha256'], 'generated_tokens': len(raw['token_ids']), 'finish_reason': raw['finish_reason'],
            'segments': len(raw['segments']), 'stochastic_redraw': False, 'recovered_after_crash': True})
        recovered += 1
    return recovered


def shard_completion(store: Store, rows: list[dict]) -> dict:
    """Count each assignment's state, whether or not the engine drained its queue.

    Complete only when every assignment in the shard has a committed complete RESULT.
    """
    from collections import Counter

    counts = Counter()
    complete_results = 0
    for row in rows:
        state = store.state(row['uid'])
        status = state['status'] if state is not None else 'untouched'
        counts[status] += 1
        if status == 'complete' and (store.folder(row['uid'])/'RESULT.json').is_file():
            complete_results += 1
    return {
        'status': 'complete' if complete_results == len(rows) else 'incomplete',
        'rows': len(rows),
        'state_counts': dict(sorted(counts.items())),
    }


def run_model(model, shard, deadline, *, max_inflight=64, design='main'):
    b = binding()
    rows = design_rows(model, design, shard)
    store = Store(bulk(b)/model/STORES[design])
    return execute_rows(b, model, rows, store, deadline, max_inflight=max_inflight,
                        design=design, shard=shard)


def execute_rows(b, model, rows, store, deadline, *, max_inflight, design, shard=(0, 1)):
    """Execute a caller-owned roster with the maintained engine and publication path.

    Args:
        b: Verified study binding.
        model: Loaded scientific model.
        rows: Full assigned store roster.
        store: Store holding the roster's durable state.
        deadline: Absolute engine deadline.
        max_inflight: Maximum concurrent requests.
        design: Main or strong.
        shard: Historical shard identity.

    Returns:
        Terminal job receipt.
    """
    from steering.runtime import closure
    recovered = recover_outputs(store, rows)
    pending = [r for r in rows if (store.state(r['uid']) or {'status': 'partial'})['status'] == 'partial']
    log({'event': 'start', 'model': model, 'design': design, 'shard': shard, 'rows': len(rows), 'pending': len(pending),
         'recovered_outputs': recovered})
    if not pending:
        return {'status': 'complete', 'rows': len(rows)}
    require(time.time()+900 < deadline, 'insufficient time for model loading before deadline')
    llm, kwargs, load_seconds = hooked_llm(b, model, max_inflight)
    try:
        contract = contract_for(b, model, rows)
        reports = llm.collective_rpc('fs_begin', kwargs={'contract': contract})
        require(len(reports) == 2, 'study requires two tensor-parallel ranks')
        tokenizer = llm.get_tokenizer()
        engine = Engine(llm, store, deadline, hooked=True, max_inflight=max_inflight)
        log({'event': 'engine', 'model': model, 'kv_tokens': engine.kv_tokens, 'load_seconds': load_seconds,
             'hook_layers': contract['hook_layers']})
        metrics = engine.run(pending, intervention_builder(b, model, tokenizer, closure(tokenizer, model)),
                             intervention_finisher(store, b, model))
        validation = llm.collective_rpc('fs_validation')
        llm.collective_rpc('fs_end')
    finally:
        llm.llm_engine.engine_core.shutdown()
    receipt = {'model': model, 'shard': list(shard), 'job': os.environ.get('SLURM_JOB_ID'), 'load_seconds': load_seconds,
               'engine': {k: v for k, v in kwargs.items() if k != 'logits_processors'}, 'metrics': metrics,
               'native_validation': validation[0], 'code': code_provenance()}
    publish(bulk(b)/model/'jobs'/(design+'-'+os.environ.get('SLURM_JOB_ID', 'local')+'.json'), receipt)
    log({'event': 'end', **receipt})
    return receipt


# ----------------------------------------------------------------------------- LFM chain

def plain_llm(kwargs):
    from vllm import LLM
    tick = time.monotonic()
    llm = LLM(**kwargs)
    return llm, kwargs, time.monotonic()-tick


def native_kwargs(b, model, max_num_seqs=128):
    """The model's study engine profile with CUDA graphs; no hooks, so no intervention state exists."""
    from models.engine import engine_kwargs
    from store import check_files
    p = b['models'][model]
    check_files(p['config_files'])
    kwargs = engine_kwargs(p, context=p['card']['max_model_len'])
    kwargs.update(max_num_seqs=max_num_seqs, enable_chunked_prefill=True, max_num_batched_tokens=8192,
                  enforce_eager=False, gpu_memory_utilization=.88)
    if model == 'lfm25':
        kwargs['kv_cache_memory_bytes'] = p['kv_cache_memory_bytes']
    return kwargs


def reader_kwargs(a, max_num_seqs=128):
    """Reader.__init__ profile with CUDA graphs and a larger running batch."""
    return {'model': a['path'], 'tokenizer': a['path'], 'tensor_parallel_size': 2, 'dtype': 'bfloat16',
            'max_model_len': a['native_context'], 'max_num_seqs': max_num_seqs, 'enforce_eager': False,
            'enable_prefix_caching': True, 'async_scheduling': False, 'enable_chunked_prefill': True,
            'max_num_batched_tokens': 8192, 'gpu_memory_utilization': .90, 'kv_cache_dtype': 'bfloat16',
            'attention_config': {'backend': 'FLASH_ATTN'}, 'language_model_only': True, 'generation_config': 'vllm'}


def lfm_native(shard, deadline):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from labeling.lfm_inputs import population
    from steering.runtime import prompt_prefix, remaining
    from store import Units
    b = binding()
    rows = population(b)['records']
    rows = [r for i, r in enumerate(sorted(rows, key=lambda r: r['uid'])) if i % shard[1] == shard[0]]
    p = b['models']['lfm25']
    store = Store(bulk(b)/'lfm25/native-progress')
    units = Units(Path(b['config']['bulk_root'])/'lfm25/native')
    pending = [r for r in rows if units.state(r['uid']) is None]
    log({'event': 'start', 'stage': 'lfm-native', 'rows': len(rows), 'pending': len(pending)})
    if not pending:
        return {'status': 'complete'}
    llm, kwargs, load_seconds = plain_llm(native_kwargs(b, 'lfm25'))
    tokenizer = llm.get_tokenizer()
    sampler = {k: v for k, v in p['sampler'].items() if v is not None}

    def build(row, history, segment):
        prompt, _, _ = prompt_prefix(b, 'lfm25', row, tokenizer)
        total = remaining(p, prompt, [])
        left = total-len(history)
        if left <= 0:
            return None
        # Repetition over prompt|output is the same token set after re-admission; presence/frequency are zero.
        params = SamplingParams(**sampler, seed=segment_seed(row['seed'], segment), max_tokens=left,
                                skip_special_tokens=False, detokenize=False, output_kind=RequestOutputKind.DELTA)
        return {'prompt_ids': prompt+history, 'params': params, 'request': None, 'total': total}

    def finish(row, progress, reason, stop, capture):
        if progress is None:
            progress, reason = load(store.folder(row['uid'])/'PARTIAL.json'), 'length'
        require(units.intent(row, os.environ['SLURM_JOB_ID']), 'native unit claimed twice')
        raw = units.output(row, {'status': 'complete', 'token_ids': progress['tokens'],
            'text': tokenizer.decode(progress['tokens'], skip_special_tokens=False), 'finish_reason': reason,
            'stop_reason': stop, 'segments': progress['segments'], 'cap': progress['total_allowance']})
        units.commit(row, {'status': 'complete', 'raw_sha256': raw['sha256'], 'stochastic_redraw': False})

    require(sampler.get('presence_penalty', 0) == 0 and sampler.get('frequency_penalty', 0) == 0,
            'native resume requires history-free penalties')
    try:
        metrics = Engine(llm, store, deadline, hooked=False, max_inflight=kwargs['max_num_seqs']).run(pending, build, finish)
    finally:
        llm.llm_engine.engine_core.shutdown()
    receipt = {'stage': 'lfm-native', 'shard': list(shard), 'job': os.environ.get('SLURM_JOB_ID'),
               'load_seconds': load_seconds, 'metrics': metrics, 'code': code_provenance()}
    publish(bulk(b)/'lfm25/jobs'/('native-'+os.environ.get('SLURM_JOB_ID', 'local')+'.json'), receipt)
    log({'event': 'end', **receipt})
    return receipt


def lfm_annotate(shard, deadline):
    """Discovery judge labels with the original renderer, prompt tokens, decoding and parser."""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from labeling.renderer import Renderer
    from labeling.renderer import token_ids
    from labeling.reader import Reader
    from store import Units
    b = binding()
    parent = load(Path(b['source_campaign'])/'CAMPAIGN.json')
    a = parent['annotator']
    require(a['config']['max_tokens'] == 16384, 'fixed reader allowance changed')
    rows = verified(ROOT/'lfm25/ANNOTATION_TASKS.json')['records']
    rows = [r for i, r in enumerate(rows) if i % shard[1] == shard[0]]
    units = Units(Path(b['config']['bulk_root'])/'reader/annotations')
    store = Store(bulk(b)/'reader-progress')
    pending = [r for r in rows if units.state(r['uid']) is None]
    log({'event': 'start', 'stage': 'lfm-annotate', 'rows': len(rows), 'pending': len(pending)})
    if not pending:
        return {'status': 'complete'}
    llm, kwargs, load_seconds = plain_llm(reader_kwargs(a))
    renderer = Renderer(parent)
    parser = Reader.__new__(Reader)
    parser.annotator, parser.renderer, parser.tokenizer = a, renderer, llm.get_tokenizer()

    def build(row, history, segment):
        prompt = token_ids(parser.tokenizer, renderer.request('render', inputs=row['inputs'])['conversation'], a['config'])
        if len(prompt)+16384 > a['native_context']:
            # As the original reader inventory: an input that cannot keep the fixed allowance is an unjudged
            # unknown (e.g. a repetition loop rendered as one giant sentence), never truncated or redrawn.
            if units.state(row['uid']) is None and units.intent(row, os.environ['SLURM_JOB_ID']):
                units.commit(row, {'status': 'unknown', 'label': None, 'reason': 'input_exceeds_reader_context',
                                   'input_tokens': len(prompt), 'stochastic_redraw': False})
            return None
        left = 16384-len(history)
        if left <= 0:
            return None
        params = SamplingParams(**a['decoding'], max_tokens=left,
                                seed=segment_seed(int(digest([20261008, 'reader', row['uid']])[:8], 16), segment),
                                skip_special_tokens=False, detokenize=False, output_kind=RequestOutputKind.DELTA)
        return {'prompt_ids': prompt+history, 'params': params, 'request': None, 'total': 16384}

    def finish(row, progress, reason, stop, capture):
        if progress is None:
            if units.state(row['uid']) is not None:  # recorded as unjudged unknown in build
                return
            progress, reason = load(store.folder(row['uid'])/'PARTIAL.json'), 'length'
        require(units.intent(row, os.environ['SLURM_JOB_ID']), 'reader unit claimed twice')
        raw = units.output(row, {'token_ids': progress['tokens'], 'finish_reason': reason,
                                 'segments': progress['segments']})
        units.commit(row, parser.parse(row, raw))

    try:
        metrics = Engine(llm, store, deadline, hooked=False, max_inflight=kwargs['max_num_seqs']).run(pending, build, finish)
    finally:
        try:
            llm.llm_engine.engine_core.shutdown()
        finally:
            renderer.close()
    receipt = {'stage': 'lfm-annotate', 'shard': list(shard), 'job': os.environ.get('SLURM_JOB_ID'),
               'load_seconds': load_seconds, 'metrics': metrics, 'code': code_provenance()}
    publish(bulk(b)/'lfm25/jobs'/('annotate-'+os.environ.get('SLURM_JOB_ID', 'local')+'.json'), receipt)
    log({'event': 'end', **receipt})
    return receipt


def lfm_replay(shard, deadline):
    from capture.lfm_replay import replay
    b = binding()
    sources = verified(ROOT/'lfm25/SOURCES.json')['records']
    rows = [s for i, s in enumerate(sources) if i % shard[1] == shard[0]]
    rows = [s for s in rows if not (Path(b['config']['bulk_root'])/'lfm25/replay'/(s['attempt_id']+'.json')).exists()]
    log({'event': 'start', 'stage': 'lfm-replay', 'pending': len(rows)})
    if not rows:
        return {'status': 'complete'}
    done = []
    from steering.runtime import Backend
    backend = Backend(b, 'lfm25', 1, deadline, replay=True)
    try:
        from store import read_source
        for row in rows:
            if time.time()+600 > deadline:
                break
            done.append(backend.capture_trace(read_source(row['source_location']),
                                              Path(b['config']['bulk_root'])/'lfm25/replay'/(row['attempt_id']+'.npz')))
    finally:
        backend.close()
    log({'event': 'end', 'stage': 'lfm-replay', 'captured': len(done), 'remaining': len(rows)-len(done)})
    return {'captured': len(done)}


# ----------------------------------------------------------------------------- bookkeeping

def log(value):
    print(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%S'), **value}, default=str), flush=True)


def status():
    b = binding()
    out = {}
    for model in MODELS:
        path = DRIVER/model/'ASSIGNMENTS.json'
        if not path.exists():
            continue
        rows = verified(path)['records']
        store = Store(bulk(b)/model/'units')
        counts = {}
        tokens = 0
        for r in rows:
            s = store.state(r['uid'])
            key = 'untouched' if s is None else s['status']
            counts[key] = counts.get(key, 0)+1
            if s and s['status'] != 'partial':
                tokens += s.get('generated_tokens', 0)
        out[model] = {'rows': len(rows), **counts, 'generated_tokens_complete': tokens}
    return out


def parse_shard(text):
    i, n = map(int, text.split('/'))
    require(0 <= i < n, 'invalid shard')
    return i, n


def main(argv=None):
    ap = argparse.ArgumentParser(prog='steering.pulse')
    ap.add_argument('command', choices=['plan', 'run', 'native', 'annotate', 'lfm-materialize', 'lfm-replay',
                                        'lfm-derive', 'status', 'plan-strong',
                                        'lfm-repair-contexts',
                                        'lfm-readout', 'lfm-dynamics'])
    ap.add_argument('--model')
    ap.add_argument('--shard', default='0/1', type=parse_shard)
    ap.add_argument('--max-inflight', type=int, default=64)
    ap.add_argument('--design', choices=sorted(MANIFESTS), default='main')
    args = ap.parse_args(argv)
    if args.command == 'plan':
        log({'event': 'plan', 'models': plan([args.model] if args.model else None)})
        return
    if args.command == 'plan-strong':
        log({'event': 'plan-strong', 'models': plan_strong(args.model.split(','))})
        return
    if args.command == 'status':
        print(json.dumps(status(), indent=1))
        return
    if args.command == 'lfm-materialize':
        from labeling.lfm_inputs import materialize
        log({'event': 'materialize', **materialize(binding())})
        return
    if args.command == 'lfm-repair-contexts':
        from labeling.lfm_inputs import repair_annotation_inputs
        value = repair_annotation_inputs(binding())
        log({'event': 'repaired', 'contexts': len(value['records']), 'sha256': value['sha256']})
        return
    if args.command == 'lfm-derive':
        from routing.selection import derive
        from steering.nominations import derive_model
        b = binding()
        if not (ROOT/'lfm25/DISCOVERY_INPUTS.json').exists():
            derive(b)
        derive_model(b, 'lfm25')
        log({'event': 'derived', 'plan': plan(['lfm25'])})
        return
    if args.command == 'lfm-readout':
        from routing.lfm_readout import lfm
        value = lfm(binding())
        log({'event': 'readout', 'sha256': value['sha256'],
             **{mode: {k: r[k] for k in ('routing', 'routing_excess_over_lexical_null')}
                for mode, r in value['models']['lfm25'].items()}})
        return
    if args.command == 'lfm-dynamics':
        from routing.dynamics import dynamics_summary
        log({'event': 'dynamics', 'sha256': dynamics_summary(binding())['sha256']})
        return
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS'), 'GPU commands require a Slurm GPU allocation')
    deadline = deadline_from_slurm()
    if args.command == 'run':
        run_model(args.model, args.shard, deadline, max_inflight=args.max_inflight, design=args.design)
    elif args.command == 'native':
        lfm_native(args.shard, deadline)
    elif args.command == 'annotate':
        lfm_annotate(args.shard, deadline)
    elif args.command == 'lfm-replay':
        lfm_replay(args.shard, deadline)


if __name__ == '__main__':
    main()
