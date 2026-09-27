"""Rescore the completed model-specific cache run; preserve original artifacts and labels.
Run: PYTHONPATH=/tmp/moe-rescore-deps:src python3 report/analyze_cache_100.py
Uses dependencies pinned in report/rescoring_requirements.txt.
"""
import argparse
import hashlib
import json
from pathlib import Path
from importlib.metadata import version
from rescore_guiding import comparison
from moe_exp.correlation_pipeline.offline_scoring import rescore
from moe_exp.moe_cache_streaming.run import load_run

ROOT = Path(__file__).resolve().parents[1]
MODEL='oss'
RUN = ROOT / f'results/moe_cache_streaming/{MODEL}_100_20260925T071519Z'


def main():
    groups, manifests, hashes = [], [], {}
    for condition in ['baseline', 'guided']:
        path = RUN / condition
        manifest, records = load_run(path)
        assert manifest['condition'] == condition and len(records) == 100
        manifests.append(manifest)
        group = {}
        for r in records:
            answer, correct, method = rescore(r['text'], str(r['input']['gold_answer']),
                                             r['input'].get('answer_type', 'math'))
            group[r['id']] = dict(input=r['input'], sampling_args=r['sampling_args'],
                prompt_sha256=hashlib.sha256(json.dumps(r['prompt_token_ids']).encode()).hexdigest(),
                is_correct=correct, original_is_correct=r['is_correct'], answer=answer,
                method=method, generated_token_count=r['generated_token_count'],
                finish_reason=r['finish_reason'])
        groups.append(group)
        for name in ['manifest.json', 'generations.jsonl']:
            p=path/name;hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for key in ['engine','sampling','policy_sha256','prompts_sha256','selected_ids','versions','template_date','strength','routing']:
        assert manifests[0][key] == manifests[1][key], key
    assert groups[0].keys() == groups[1].keys()
    result=dict(run=str(RUN.relative_to(ROOT)), comparison=comparison(*groups),
        original_correct=[sum(r['original_is_correct'] for r in g.values()) for g in groups],
        source_sha256=hashes,
        scoring_versions={k:version(k) for k in ['math-verify','latex2sympy2_extended','antlr4-python3-runtime','sympy','mpmath']},
        records=dict(zip(['baseline','guided'],groups)))
    for source in ['report/analyze_cache_100.py','report/rescore_guiding.py',
                   'src/moe_exp/correlation_pipeline/offline_scoring.py',
                   'src/moe_exp/moe_cache_streaming/cache.py','src/moe_exp/moe_cache_streaming/run.py']:
        result['source_sha256'][source]=hashlib.sha256((ROOT/source).read_bytes()).hexdigest()
    (ROOT/f'report/{MODEL}_cache_100_rescoring.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','source_sha256']},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['oss','qwen','gemma'],default='oss')
    MODEL=parser.parse_args().model
    RUN=ROOT/f'results/moe_cache_streaming/{MODEL}_100_20260925T071519Z'
    main()
