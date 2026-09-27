"""Run with python -m moe_exp.moe_cache_streaming.run {generate,analyze}."""
import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
import time

from .cache import simulate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def select_rows(rows, limit):
    """Deterministic dataset round-robin, one attempt per problem, no outcome selection."""
    if limit < 1:
        raise ValueError('limit must be positive')
    pools, seen = defaultdict(list), set()
    for row in sorted(rows, key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest()):
        key = (row.get('dataset'), row.get('source_problem_id', row.get('problem_id', row['id'])))
        if key not in seen:
            seen.add(key)
            pools[row.get('dataset', 'custom')].append(row)
    result = []
    while pools and len(result) < limit:
        for name in sorted(list(pools)):
            result.append(pools[name].pop())
            if not pools[name]:
                del pools[name]
            if len(result) == limit:
                break
    return result


def generate(args):
    from moe_exp.moe_guiding.run import load_prompts
    from moe_exp.moe_identity_guiding.calibration import validate_policy, problem_key
    from moe_exp.moe_identity_guiding.execution import render_inputs, atomic_json
    from moe_exp.moe_identity_guiding.sampling import request_sampling
    from moe_exp.correlation_pipeline.scoring import score_completion
    from moe_exp.correlation_pipeline.model_profiles import model_profile
    from vllm import LLM, SamplingParams

    if not math.isfinite(args.strength):
        raise ValueError('strength must be finite')
    policy = read(args.policy)
    validate_policy(policy, args.model)
    rows = select_rows(load_prompts(args.prompts), args.limit)
    if not rows or set(policy['calibration_problems']) & {problem_key(r) for r in rows}:
        raise ValueError('Empty evaluation or calibration overlap')
    if args.prefill_chunk_size < 1:
        raise ValueError('prefill-chunk-size must be positive')
    if args.max_tokens < 1 or args.max_model_len <= args.max_tokens:
        raise ValueError('Need 0 < max_tokens < max_model_len')
    sampling = dict(max_tokens=args.max_tokens, temperature=0.6, top_p=0.95, top_k=-1, seed=0)
    requests = request_sampling(rows, sampling, model=args.model)
    engine = dict(language_model_only=model_profile(args.model).language_model_only,
                  model=args.model, revision=args.revision, tokenizer_revision=args.revision,
                  max_model_len=args.max_model_len, max_num_seqs=1,
                  max_num_batched_tokens=args.prefill_chunk_size,
                  enforce_eager=True, enable_prefix_caching=False, enable_chunked_prefill=True, async_scheduling=False,
                  tensor_parallel_size=1, seed=0, generation_config='vllm',
                  gpu_memory_utilization=0.9,
                  worker_extension_cls='moe_exp.moe_cache_streaming.routing.CacheWorkerExtension')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', condition=args.condition, policy=policy,
                    policy_sha256=digest(args.policy), prompts_sha256=digest(args.prompts),
                    engine=engine, sampling=sampling, strength=args.strength,
                    template_date=args.template_date, selected_ids=[r['id'] for r in rows],
                    versions={k: version(k) for k in ('torch', 'vllm', 'transformers')},
                    scope='decode only, excludes first token predicted by prefill',
                    timing_scope='instrumented resident-weight generation, NOT SSD inference',
                    scoring_contract='correlation_pipeline.score_completion')
    atomic_json(args.output / 'manifest.json', manifest)
    llm = LLM(**engine)
    rendered, inputs = render_inputs(llm.get_tokenizer(), rows, args.template_date)
    info = llm.collective_rpc('cache_configure', kwargs=dict(
        policy=policy, strength=args.strength, condition=args.condition))
    if len(info) != 1:
        raise ValueError('Expected one worker')
    manifest['routing'] = info[0]
    # Warm the same eager path without active recorder state.
    llm.generate([inputs[0]], SamplingParams(max_tokens=1, temperature=0), use_tqdm=False)
    with (args.output / 'generations.jsonl').open('x') as out:
        for index, (row, prompt, params) in enumerate(zip(rows, inputs, requests, strict=True)):
            llm.collective_rpc('cache_begin', kwargs=dict(prompt_tokens=len(prompt['prompt_token_ids'])))
            start = time.perf_counter()
            output = llm.generate([prompt], SamplingParams(**params), use_tqdm=False)[0]
            elapsed = time.perf_counter() - start
            completion = output.outputs[0]
            count = len(completion.token_ids)
            try:
                traces = llm.collective_rpc('cache_finish', kwargs=dict(generated_tokens=count))
            except Exception:
                atomic_json(args.output / 'failed_completion.json', dict(
                    id=row['id'], generated_token_count=count,
                    generated_token_ids=list(completion.token_ids), text=completion.text,
                    finish_reason=completion.finish_reason, stop_reason=completion.stop_reason))
                raise
            trace_name = f'routing_{index:05d}.json.gz'
            with gzip.open(args.output / trace_name, 'wt') as f:
                json.dump(traces[0], f)
            answer, correct, method = score_completion(
                row, answer_type=row.get('answer_type', 'math'), model_text=completion.text)
            record = dict(id=row['id'], input=row, rendered_prompt=rendered[index],
                          prompt_token_ids=output.prompt_token_ids, sampling_args=params,
                          text=completion.text, is_correct=correct, model_answer=answer,
                          scoring_method=method, generated_token_count=count,
                          finish_reason=completion.finish_reason, instrumented_seconds=elapsed,
                          routing=trace_name, routing_sha256=digest(args.output / trace_name))
            out.write(json.dumps(record) + '\n')
            out.flush()
            print(f'{args.condition}: {index+1}/{len(rows)}; {count} tokens', flush=True)
    manifest['status'] = 'complete'
    manifest['generations_sha256'] = digest(args.output / 'generations.jsonl')
    atomic_json(args.output / 'manifest.json', manifest)


def load_run(path):
    manifest = read(path / 'manifest.json')
    if manifest['status'] != 'complete':
        raise ValueError('Incomplete run')
    if digest(path / 'generations.jsonl') != manifest['generations_sha256']:
        raise ValueError('Generation hash mismatch')
    records = [json.loads(line) for line in (path / 'generations.jsonl').read_text().splitlines()]
    if [r['id'] for r in records] != manifest['selected_ids']:
        raise ValueError('Missing, duplicate, or reordered attempts')
    return manifest, records


def analyze(args):
    a, baseline = load_run(args.baseline)
    b, guided = load_run(args.guided)
    if (a['condition'], b['condition']) != ('baseline', 'guided'):
        raise ValueError('Expected baseline and guided runs')
    for key in ('engine', 'sampling', 'policy_sha256', 'prompts_sha256', 'selected_ids',
                'versions', 'template_date', 'strength', 'routing', 'scoring_contract'):
        if a[key] != b[key]:
            raise ValueError(f'Unmatched {key}')
    for x, y in zip(baseline, guided, strict=True):
        for key in ('id', 'input', 'prompt_token_ids', 'sampling_args'):
            if x[key] != y[key]:
                raise ValueError(f'Unmatched prompt {key}')
    policy = a['policy']
    budgets = sorted(set(args.budgets))
    max_pins = max(sum(s > 0 for s in p['scores']) for p in policy['layers'].values())
    if not budgets or min(budgets) < policy['top_k'] + max_pins or max(budgets) > policy['num_experts']:
        raise ValueError('Budgets must fit top-k plus pins, and cannot exceed expert count')
    experiments = [('baseline_lru', args.baseline, baseline, 'none'),
                   ('baseline_frequency_pins', args.baseline, baseline, 'frequency'),
                   ('guided_target_pins', args.guided, guided, 'targets'),
                   ('guided_lru', args.guided, guided, 'none')]
    summaries = []
    for name, directory, records, pin_rule in experiments:
        totals = {budget: defaultdict(int) for budget in budgets}
        for record in records:
            if digest(directory / record['routing']) != record['routing_sha256']:
                raise ValueError('Routing hash mismatch')
            with gzip.open(directory / record['routing'], 'rt') as f:
                trace = json.load(f)
            if set(trace) != set(a['routing']['layers']):
                raise ValueError('Missing routing layer')
            for layer, sequence in trace.items():
                if len(sequence) != record['generated_token_count'] - 1:
                    raise ValueError('Decode length mismatch')
                if any(len(ids) != policy['top_k'] or any(i >= policy['num_experts'] for i in ids)
                       for ids in sequence):
                    raise ValueError('Invalid routing identities')
                p = policy['layers'].get(layer)
                targets = [i for i, s in enumerate(p['scores']) if s > 0] if p else []
                pins = []
                if pin_rule == 'targets':
                    pins = targets
                elif pin_rule == 'frequency' and p:
                    pins = [e['expert'] for e in sorted(p['experts'],
                            key=lambda e: (-e['frequency_mass'], e['expert']))[:len(targets)]]
                for budget in budgets:
                    counts = simulate(sequence, budget, pins)
                    for key, value in counts.items():
                        totals[budget][key] += value
        scored = [r['is_correct'] for r in records]
        if any(type(v) is not bool for v in scored):
            raise ValueError('Unscored answer')
        for budget, counts in totals.items():
            summaries.append(dict(condition=name, capacity_per_layer=budget, **counts,
                accuracy=sum(scored)/len(scored), attempts=len(records),
                mean_generated_tokens=sum(r['generated_token_count'] for r in records)/len(records),
                loads_per_answer=counts['total_loads']/len(records),
                loads_per_decode_token=counts['total_loads']/max(1, sum(r['generated_token_count']-1 for r in records)),
                hit_fraction=counts['hits']/max(1, counts['requests'])))
    quality = {}
    for condition, records in [('baseline', baseline), ('guided', guided)]:
        groups = defaultdict(list)
        for r in records:
            groups[r['input'].get('dataset', 'custom')].append(r)
        quality[condition] = {name: dict(
            attempts=len(group), accuracy=sum(r['is_correct'] for r in group)/len(group),
            truncated=sum(r.get('finish_reason') == 'length' for r in group),
            mean_generated_tokens=sum(r['generated_token_count'] for r in group)/len(group))
            for name, group in groups.items()}
    result = dict(quality_by_dataset=quality, cache_scope='cold per response, decode only, per-layer equal expert slots',
                  first_token_excluded=True, actual_ssd_speed_measured=False,
                  pinning_scope='policy layers only; other layers use LRU',
                  source_manifests={str(p): digest(p / 'manifest.json') for p in (args.baseline, args.guided)},
                  summaries=summaries)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Equal-memory cache simulation', '',
             'Decode-only, cold per response. Pin initialization counts as loads. No SSD timing claim.', '',
             '| Condition | Slots/layer | Accuracy | Loads/answer | Loads/decode token | Hit rate |',
             '|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['condition']} | {r['capacity_per_layer']} | {r['accuracy']:.2%} | {r['loads_per_answer']:.1f} | {r['loads_per_decode_token']:.2f} | {r['hit_fraction']:.2%} |")
    (args.output / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('generate')
    gen.add_argument('--policy', type=Path, required=True)
    gen.add_argument('--prompts', type=Path, required=True)
    gen.add_argument('--output', type=Path, required=True)
    gen.add_argument('--condition', choices=['baseline', 'guided'], required=True)
    gen.add_argument('--model', default='openai/gpt-oss-20b')
    gen.add_argument('--revision')
    gen.add_argument('--strength', type=float, default=1)
    gen.add_argument('--limit', type=int, default=24)
    gen.add_argument('--max-tokens', type=int, default=32768)
    gen.add_argument('--max-model-len', type=int, default=49152)
    gen.add_argument('--prefill-chunk-size', type=int, default=2048)
    gen.add_argument('--template-date', default='2026-09-22')
    ana = sub.add_parser('analyze')
    ana.add_argument('--baseline', type=Path, required=True)
    ana.add_argument('--guided', type=Path, required=True)
    ana.add_argument('--output', type=Path, required=True)
    ana.add_argument('--budgets', type=int, nargs='+', default=[8, 16, 32])
    args = parser.parse_args()
    (generate if args.command == 'generate' else analyze)(args)


if __name__ == '__main__':
    main()
