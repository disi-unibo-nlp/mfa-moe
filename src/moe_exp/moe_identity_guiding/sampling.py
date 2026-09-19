"""Recover exact per-attempt seeds and template options from original traces."""
import json
from pathlib import Path

from moe_exp.jsonl import iter_jsonl
from moe_exp.moe_guiding.run import load_prompts


def prepare_sampling(prompts: Path, generations, output: Path):
    rows = load_prompts(prompts)
    wanted = {r['id']: r for r in rows}
    found = {}
    for path in generations:
        for source in iter_jsonl(path):
            identity = json.dumps([source['dataset'], source['problem_id'], source.get('sample_id', 0)])
            if identity not in wanted:
                continue
            if identity in found:
                raise ValueError(f'Duplicate source attempt: {identity}')
            row = wanted[identity]
            for key in ('dataset', 'problem_id', 'source_problem_id', 'prompt', 'gold_answer',
                        'system_prompt', 'generation_messages'):
                if row.get(key) != source.get(key):
                    raise ValueError(f'Source prompt mismatch for {identity}: {key}')
            config = source.get('metadata', {}).get('generation_config', {})
            if type(config.get('seed')) is not int:
                raise ValueError(f'Missing original seed: {identity}')
            found[identity] = {**row, 'original_generation_config': {
                key: config.get(key) for key in
                ('seed', 'temperature', 'top_p', 'top_k', 'max_tokens', 'chat_template_kwargs')},
                'original_model_id': source['model_id']}
    missing = wanted.keys() - found.keys()
    if missing:
        raise ValueError(f'Missing {len(missing)} original attempts')
    payload = ''.join(json.dumps(found[r['id']], ensure_ascii=False)+'\n' for r in rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_text() != payload:
            raise ValueError('Existing sampling prompts differ; choose a new output file')
    else:
        with output.open('x') as handle:
            handle.write(payload)
    return dict(prompts=len(rows), output=str(output))


def request_sampling(rows, sampling, *, require_original=False, model=None):
    """Use original request seeds; CLI seed is a shared offset (default zero)."""
    params = []
    for row in rows:
        config = row.get('original_generation_config')
        if require_original and config is None:
            raise ValueError('Missing original seeds; run prepare-sampling first')
        if config is not None:
            if type(config.get('seed')) is not int:
                raise ValueError('Invalid original attempt seed')
            if require_original:
                if row.get('original_model_id') != model:
                    raise ValueError('Original generation model differs from --model')
                for key in ('temperature', 'top_p', 'max_tokens'):
                    if config.get(key) != sampling[key]:
                        raise ValueError(f'Original {key} differs from requested sampling')
                original_topk = config.get('top_k')
                if (-1 if original_topk == 0 else original_topk) != sampling['top_k']:
                    raise ValueError('Original top_k differs from requested sampling')
            seed = config['seed'] + sampling['seed']
        else:
            # Stable independent streams for custom prompts, without source metadata.
            import hashlib
            seed = (int.from_bytes(hashlib.sha256(row['id'].encode()).digest()[:4], 'big')
                    + sampling['seed']) % (2**32)
        params.append({**sampling, 'seed': seed})
    return params
