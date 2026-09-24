"""Create a reproducible, stratified qualitative sample of saved right-to-wrong flips.

Run from the repository root: python3 report/inspect_paper_failures.py
No model calls; original generations are read only. This is an assistant-assisted
excerpt review, not independent human annotation or a prevalence estimate.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260924
RUNS = {
    'oss_positive': 'results/moe_identity_guiding/oss_global/variants/positive_paper_eps_0.01/strength_1/sampling/full',
    'oss_negative': 'results/moe_identity_guiding/oss_global/variants/negative_paper_eps_0.01/strength_1/sampling/full',
    'qwen_positive_partial': 'results/moe_identity_guiding/qwen_global/variants/positive_paper_eps_0.01/strength_1/sampling/full',
}


def text_diagnostics(row):
    text = row['text']
    words = re.findall(r'\S+', text)
    ngrams = Counter(tuple(words[i:i+16]) for i in range(max(0, len(words)-15)))
    most = ngrams.most_common(1)
    phrase, count = (' '.join(most[0][0]), most[0][1]) if most else ('', 0)
    # Count positions covered by at least one exact 16-word span occurring >=3 times.
    covered = bytearray(len(words))
    for i in range(max(0, len(words)-15)):
        if ngrams[tuple(words[i:i+16])] >= 3:
            covered[i:i+16] = b'\1' * 16
    mid = len(text)//2
    return {
        'is_correct': row['is_correct'], 'model_answer': row['model_answer'],
        'generated_tokens': row['generated_token_count'], 'finish_reason': row['finish_reason'],
        'scoring_method': row['scoring_method'], 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'box_markers': len(re.findall(r'\\boxed\s*\{', text)),
        'word_count': len(words), 'most_repeated_16word_span': phrase,
        'most_repeated_16word_span_count': count,
        'words_in_16word_spans_occurring_at_least_3_times': sum(covered),
        'head': text[:800], 'middle': text[max(0, mid-400):mid+400], 'tail': text[-1800:],
    }


def main():
    hashes = {}
    for name in ['src/moe_exp/utils.py', 'src/moe_exp/correlation_pipeline/scoring.py']:
        hashes[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    result = {
        'seed': SEED,
        'selection': 'Within each run, restrict to saved baseline-correct/guided-incorrect matched IDs. Sample sorted IDs without replacement using random.Random(f"{seed}:{run}:{finish_reason}"). For each complete OSS run select four length and four stop failures; for partial Qwen select six length failures. This is attempt sampling, not problem-balanced sampling.',
        'review_scope': 'Assistant-assisted inspection of first 800 characters, central 800 characters, final 1800 characters and most frequent exact 16-whitespace-word span, plus targeted surrounding context recorded separately. No independent human annotation, exhaustive mathematical verification, or estimate of population failure-mode prevalence.',
        'repetition_measure': 'Whitespace-delimited, case-sensitive words including punctuation. Report maximum occurrence count of an exact 16-word span and number of word positions covered by spans occurring >=3 times. Descriptive measures; overlap is allowed and no looping classifier is fitted.',
        'source_sha256': hashes, 'runs': {},
    }
    for name, folder in RUNS.items():
        groups = []
        manifests = []
        for condition in ('baseline', 'guided'):
            manifest_path = ROOT / folder / condition / 'manifest.json'
            raw = manifest_path.read_bytes(); hashes[str(manifest_path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
            manifest = json.loads(raw); manifests.append(manifest)
            if condition == 'guided':
                assert manifest['policy']['guiding_method'] == 'paper' and manifest['strength'] == 1
                assert manifest['status'] == ('failed' if 'partial' in name else 'complete')
            records = {}; digest = hashlib.sha256()
            path = ROOT / folder / condition / 'generations.jsonl'
            with path.open('rb') as handle:
                for line in handle:
                    digest.update(line); row = json.loads(line)
                    assert row['id'] not in records and type(row['is_correct']) is bool
                    records[row['id']] = {k: row[k] for k in ['id', 'input', 'text', 'is_correct', 'model_answer', 'generated_token_count', 'finish_reason', 'scoring_method']}
            hashes[str(path.relative_to(ROOT))] = digest.hexdigest(); groups.append(records)
        baseline, guided = groups
        b, g = manifests
        assert b['status'] == 'complete' and not g['calibration_overlap']
        for field in ['prompts_sha256', 'policy_sha256', 'engine_args', 'sampling_args',
                      'versions', 'scoring_contract', 'seed_strategy', 'token_scope', 'template_date']:
            assert b[field] == g[field], field
        assert len(guided) == g['completed_count']
        assert len(baseline) == b['completed_count'] == b['expected_count'] == g['expected_count']
        strata = {'length': [], 'stop': []}
        for key, row in guided.items():
            assert key in baseline and row['input'] == baseline[key]['input']
            if baseline[key]['is_correct'] and not row['is_correct']:
                assert row['finish_reason'] in strata
                strata[row['finish_reason']].append(key)
        samples = []
        for reason, ids in strata.items():
            count = (6 if reason == 'length' else 0) if 'partial' in name else 4
            assert len(ids) >= count
            chosen = random.Random(f'{SEED}:{name}:{reason}').sample(sorted(ids), count)
            for key in chosen:
                row = guided[key]
                samples.append({'id': key, 'dataset': row['input']['dataset'], 'source_problem_id': row['input']['source_problem_id'], 'gold_answer': row['input']['gold_answer'], 'prompt': row['input']['prompt'], 'stratum': reason, 'baseline': text_diagnostics(baseline[key]), 'guided': text_diagnostics(row)})
        result['runs'][name] = {'folder': folder, 'saved_attempts': len(guided), 'right_to_wrong': sum(map(len, strata.values())), 'eligible_by_finish_reason': {k: len(v) for k, v in strata.items()}, 'sample_count': len(samples), 'distinct_sample_problems': len({(s['dataset'], s['source_problem_id']) for s in samples}), 'samples': samples}
        print(name, 'eligible', result['runs'][name]['eligible_by_finish_reason'], 'sampled', len(samples), flush=True)
    # Verify inputs against the already audited numerical snapshots.
    reference = json.loads((ROOT / 'report/current_results_sources.json').read_text())['sha256']
    reference.update(json.loads((ROOT / 'report/partial_guiding_statistics.json').read_text())['source_sha256'])
    for name, digest in hashes.items():
        if name.startswith('results/'):
            assert reference[name] == digest, name
    review_path = ROOT / 'report/paper_failure_review.json'
    if review_path.exists():
        notes = json.loads(review_path.read_text())['cases']
        keyed = {(r, x['id']): x for r, run in result['runs'].items() for x in run['samples']}
        assert len(notes) == len(keyed) == len({(x['run'], x['id']) for x in notes})
        for note in notes:
            sample = keyed[(note['run'], note['id'])]
            assert note['guided_text_sha256'] == sample['guided']['text_sha256']
            assert note['category'] in ('repetition', 'arithmetic_error', 'formatting_disagreement')
        result['review_counts'] = {r: dict(Counter(x['category'] for x in notes if x['run'] == r))
                                  for r in result['runs']}
        result['review_sha256'] = hashlib.sha256(review_path.read_bytes()).hexdigest()
    (ROOT / 'report/paper_failure_inspection.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
