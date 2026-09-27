"""Analyze the full OSS diagnostic rerun without changing original artifacts.

PYTHONPATH=/tmp/moe-rescore-deps:src python3 report/analyze_gemma_cache_diagnostics.py
Dependencies: report/rescoring_requirements.txt.
"""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from analyze_pinned_expert_cache import analyze
from rescore_guiding import comparison
from moe_exp.correlation_pipeline.offline_scoring import rescore, CONTRACT

RUN = 'results/moe_identity_guiding/gemma_detailed_20260924T232401Z/sampling/full'


def main():
    manifests, groups, hashes = [], [], {}
    for condition in ['baseline', 'guided']:
        directory = ROOT / RUN / condition
        m = json.loads((directory / 'manifest.json').read_text())
        assert m['status'] == 'complete' and m['condition'] == condition
        assert not m['calibration_overlap']
        manifests.append(m)
        records = {}
        for line in (directory / 'generations.jsonl').open():
            r = json.loads(line)
            assert r['id'] not in records
            answer, correct, method = rescore(r['text'], str(r['input']['gold_answer']),
                                             r['input'].get('answer_type', 'math'))
            records[r['id']] = dict(input=r['input'], sampling_args=r['sampling_args'],
                prompt_sha256=hashlib.sha256(json.dumps(r['prompt_token_ids']).encode()).hexdigest(),
                is_correct=correct, original_is_correct=r['is_correct'], answer=answer,
                method=method, generated_token_count=r['generated_token_count'],
                finish_reason=r['finish_reason'])
        assert len(records) == m['completed_count'] == m['expected_count'] == 1395
        groups.append(records)
        for name in ['manifest.json', 'generations.jsonl']:
            path = directory / name
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for key in ['policy_sha256','prompts_sha256','engine_args','sampling_args','versions','scoring_contract']:
        assert manifests[0][key] == manifests[1][key], key
    assert groups[0].keys() == groups[1].keys()
    provenance = manifests[0].get('reused_baseline')
    if provenance:
        assert provenance['generations_sha256'] == hashes[RUN + '/baseline/generations.jsonl']
    import copy
    m=copy.deepcopy(manifests[1])
    sessions=[s['routing_diagnostics'] for s in m.get('previous_sessions', [])]+[m['routing_diagnostics']]
    merged=copy.deepcopy(sessions[0])
    for reports in sessions[1:]:
        assert len(reports)==len(merged)==1
        assert reports[0]['layers'].keys()==merged[0]['layers'].keys()
        for layer,d in reports[0]['layers'].items():
            for key,value in d.items():
                if isinstance(value,list):
                    merged[0]['layers'][layer][key]=[a+b for a,b in zip(merged[0]['layers'][layer][key],value,strict=True)]
                elif isinstance(value,(int,float)):
                    merged[0]['layers'][layer][key]+=value
                else:
                    assert merged[0]['layers'][layer][key]==value
    m['routing_diagnostics']=merged
    rows, total = analyze(m)
    result = dict(run=RUN, scoring_contract=CONTRACT,
        versions={p:version(p) for p in ['math-verify','latex2sympy2_extended','antlr4-python3-runtime','sympy','mpmath']},
        source_sha256=hashes, comparison=comparison(*groups), layers=rows, aggregate=total, diagnostic_sessions=len(sessions),
        session_completion_counts=[s.get("completed_count") for s in manifests[1].get("previous_sessions",[])]+[manifests[1]["completed_count"]],
        original_correct=[sum(r['original_is_correct'] for r in g.values()) for g in groups],
        records=dict(zip(['baseline','guided'], groups)),
        limitations=['Before/after routing uses guided hidden states, not separate baseline routing',
                     'Seven policy layers; combined session counters include prefill, decode, padding and discarded unfinished work before interruption',
                     'Fixed resident-cache proxy; no ordered loads, cold-start costs, SSD bytes or latency',
                     'Frequency control uses calibration statistics; oracle uses evaluation histograms',
                     'Diagnostic rerun is separate from historical minimal-diagnostics evaluation'])
    for relative in ['report/analyze_gemma_cache_diagnostics.py','report/analyze_pinned_expert_cache.py',
                     'src/moe_exp/correlation_pipeline/offline_scoring.py',
                     'src/moe_exp/moe_identity_guiding/bootstrap.py']:
        result['source_sha256'][relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    (ROOT / 'report/gemma_cache_diagnostics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['comparison','aggregate','original_correct']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
