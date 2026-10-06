"""Bounded CPU-only decoding of seven stratified saved continuation cases."""
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    assert value['sha256'] == digest({k: v for k, v in value.items() if k != 'sha256'}), str(path)
    return value


def main():
    from tokenizers import Tokenizer
    manifest_path = REPO / 'report/experimental-resume-v1/OVERNIGHT_FRESH_COMPARISON_MANIFEST_v2.json'
    manifest = sealed(manifest_path)
    frame_path = Path(manifest['source_frame_path'])
    frame = sealed(frame_path)
    assert frame['sha256'] == manifest['source_frame_sha256']
    source = {r['uid']: r for r in frame['records']}
    extraction = sealed(ROOT / 'operator-panel-v1/legacy/CONTINUATION_LENGTHS.json')
    csv_path = ROOT / 'operator-panel-v1/legacy/continuation_lengths.csv'
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == extraction['artifacts'][csv_path.name]
    native = {r['uid']: r for r in csv.DictReader(csv_path.open()) if r['lane'] == 'fresh' and r['arm'] == 'native'}
    strata = defaultdict(list)
    for index, row in enumerate(manifest['rows']):
        for seed in manifest['seeds']:
            uid = 'overnight-v2|' + digest([manifest['sha256'], row['uid'], seed, 'native'])[:24]
            r = native[uid]
            strata[row['transition'], r['boundary_status'], r['finish']].append((index, seed))
    selected, seen = [], set()
    for key, block in sorted(strata.items()):
        remaining = 2 if key[1] == 'UNCLOSED_CAPPED' else 1
        block.sort(key=lambda pair: hashlib.sha256(('completion-audit-2026-10-05|' + manifest['rows'][pair[0]]['uid']).encode()).hexdigest())
        for index, seed in block:
            prefix_uid = manifest['rows'][index]['uid']
            if prefix_uid in seen:
                continue
            selected.append((index, seed, key))
            seen.add(prefix_uid)
            remaining -= 1
            if not remaining:
                break
    tokenizer_path = REPO.parent / 'cache/hf/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d08a9490559dae23d0cff1d9466213d989/tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    assert tokenizer.token_to_id('</think>') == 248069
    cases = []
    for case_index, (index, seed, stratum) in enumerate(selected):
        prefix = manifest['rows'][index]
        shard_index = next(i for i, s in enumerate(manifest['shards']) if s['start_row'] <= index < s['end_row'])
        folder = ROOT / 'overnight-routing-v2-6b5aa286e46cb818' / f'shard-{shard_index:03d}'
        summary = sealed(folder / 'SUMMARY.json')
        outputs, receipts = [], []
        for b in range(summary['batches']):
            batch_path = folder / f'batch-{b:03d}.json'
            batch = sealed(batch_path)
            assert batch['manifest_sha256'] == manifest['sha256']
            outputs.extend(batch['outputs'])
            receipts.append({'path': str(batch_path), 'seal': batch['sha256'], 'file_sha256': hashlib.sha256(batch_path.read_bytes()).hexdigest()})
        arms = ['native', 'native_duplicate', 'bias', 'force', 'reweight', ['random_bias', 'random_force', 'random_reweight'][case_index % 3]]
        completions = []
        for arm in arms:
            row = next(r for r in outputs if r['prefix_uid'] == prefix['uid'] and r['seed'] == seed and r['arm'] == arm)
            assert row['uid'] == 'overnight-v2|' + digest([manifest['sha256'], prefix['uid'], seed, arm])[:24]
            ids = row['tokens']
            closure = ids.index(248069) if 248069 in ids else None
            completions.append({'uid': row['uid'], 'arm': arm, 'policy': row['policy'], 'generated_tokens': len(ids),
                'finish': row['finish'], 'error': row['error'], 'reasoning_tokens': closure if closure is not None else len(ids),
                'answer_tokens': len(ids) - closure - 1 if closure is not None else 0, 'reasoning_closed': closure is not None,
                'text': tokenizer.decode(ids, skip_special_tokens=False), 'token_ids_sha256': digest(ids)})
        cases.append({'case': case_index + 1, 'question': prefix['question'], 'family': prefix['family'], 'prefix_uid': prefix['uid'],
            'transition': prefix['transition'], 'seed': seed, 'native_stratum': list(stratum), 'prefix_tokens': prefix['prefix_tokens'],
            **source[prefix['uid']]['reader_input'], 'decoded_saved_prefix': tokenizer.decode(prefix['prefix_ids'], skip_special_tokens=False),
            'raw_batch_receipts': receipts, 'completions': completions})
    value = {'schema': 'saved-completion-qualitative-sample-v1', 'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'selection_rule': 'Hash-ranked distinct native prefixes: two per unclosed/capped transition stratum, one per other observed stratum; no selection by intervention contents.',
        'native_stratum_counts': {str(k): len(v) for k, v in strata.items()}, 'manifest_path': str(manifest_path),
        'manifest_sha256': manifest['sha256'], 'context_frame_path': str(frame_path), 'context_frame_sha256': frame['sha256'],
        'tokenizer_file_sha256': hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        'extraction_sha256': extraction['sha256'], 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cases': cases,
        'scope': 'Small qualitative audit of saved outputs. Stratified sample cannot estimate population incidence. No new inference, classification, grade, policy or cohort change.'}
    value['sha256'] = digest(value)
    out = REPO / 'report/experimental-resume-v1/completion-audit-v1'
    out.mkdir(exist_ok=True)
    path = out / 'SAMPLE.json'
    if path.exists():
        prior = sealed(path)
        assert prior['script_sha256'] == value['script_sha256'] and prior['cases'] == cases
        value = prior
    else:
        with path.open('x') as stream:
            json.dump(value, stream, indent=1, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    assert sealed(path)['sha256'] == value['sha256']
    print(json.dumps({'path': str(path), 'sha256': value['sha256'], 'cases': len(cases),
        'completions': sum(len(c['completions']) for c in cases),
        'overview': [{'case': c['case'], 'question': c['question'], 'problem': c['problem'],
            'prefix_tokens': c['prefix_tokens'], 'lengths': {r['arm']: [r['reasoning_tokens'], r['generated_tokens'], r['finish']] for r in c['completions']}} for c in cases]}, indent=1))


if __name__ == '__main__':
    main()
