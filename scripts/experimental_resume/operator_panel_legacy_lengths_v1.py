"""CPU-only generated length supplement for sealed A/B/C/fresh continuations."""
from collections import Counter
import csv
from pathlib import Path

import operator_panel_v1 as P
import operator_panel_outcomes_v1 as O


def run(panel):
    from transformers import AutoTokenizer
    score, engine = O.scoring()
    tokenizer = AutoTokenizer.from_pretrained(engine.snapshot_path(), local_files_only=True)
    boundary = O.boundary_contract(tokenizer, panel['tokenizer_sha256'])
    out = P.ROOT / 'legacy'; out.mkdir(parents=True, exist_ok=True)
    manifests = {'A': 'OVERNIGHT_DISCOVERY_A_MANIFEST_v1.json', 'B': 'OVERNIGHT_DISCOVERY_B_MANIFEST_v1.json',
                 'C': 'OVERNIGHT_DISCOVERY_C_MANIFEST_v2.json', 'fresh': 'OVERNIGHT_FRESH_COMPARISON_MANIFEST_v2.json'}
    summaries, all_rows = {}, []
    for lane, name in manifests.items():
        path = P.DOC / name; manifest = P.U.sealed(path)
        version = 'v1' if lane in ('A', 'B') else 'v2'
        run_out = P.ROOT.parent / ('overnight-routing-' + version + '-' + manifest['sha256'][:16])
        stage = P.U.sealed(run_out / 'STAGE_COMPLETION.json')
        P.require(stage['manifest_sha256'] == manifest['sha256'] and
                  stage['counts']['assigned'] == manifest['expected_requests'] and
                  all(P.U.file_sha(Path(p)) == h for p, h in manifest['code_files'].items()),
                  'legacy sealed stage/source changed')
        if version == 'v1':
            import overnight_routing_runner_v1 as generation
        else:
            import overnight_routing_runner_v2 as generation
        expected = [meta for _, meta in generation.request_metadata(manifest, manifest['rows'], manifest['arms'])]
        prefix = {r['uid']: r for r in manifest['rows']}
        receipts, outputs = [], []
        for i, shard in enumerate(manifest['shards']):
            directory = run_out / f'shard-{i:03d}'
            binding, summary = P.U.sealed(directory / 'BINDING.json'), P.U.sealed(directory / 'SUMMARY.json')
            completion = P.U.sealed(directory / 'OVERNIGHT_COMPLETION.json')
            P.require(binding['manifest_sha256'] == manifest['sha256'] and
                      summary['binding_sha256'] == binding['sha256'] and
                      completion['sha256'] == stage['shard_completion_sha256s'][i] and
                      completion['source_summary_sha256'] == summary['sha256'], 'legacy shard binding differs')
            for b in range(summary['batches']):
                batch_path = directory / f'batch-{b:03d}.json'; batch = P.U.sealed(batch_path)
                assignment = P.U.sealed(directory / f'batch-{b:03d}-assignment.json')
                P.require(batch['manifest_sha256'] == manifest['sha256'] and
                          batch['binding_sha256'] == binding['sha256'] and
                          [r['uid'] for r in batch['outputs']] == [r['uid'] for r in assignment['requests']] and
                          P.U.file_sha(directory / f'batch-{b:03d}.npz') == batch['array_sha256'],
                          'legacy batch/route bytes differ')
                outputs.extend(batch['outputs'])
                receipts.append({'path': str(batch_path), 'sha256': batch['sha256'], 'routes_sha256': batch['array_sha256']})
        P.require([r['uid'] for r in outputs] == [r['uid'] for r in expected], 'legacy assignment universe differs')
        rows = []
        for assigned, output in zip(expected, outputs, strict=True):
            P.require(all(output.get(k) == v for k, v in assigned.items()), 'legacy output metadata differs')
            original = prefix[assigned['prefix_uid']]
            open_reasoning = P.THINK_END_ID not in original['prefix_ids']
            row = {**assigned, 'lane': lane, 'continuation_cap': manifest['horizon'],
                   'generated_tokens': len(output['tokens']), 'finish': output['finish'],
                   'error': output['error'], **O.lengths(output['tokens'], output['finish'],
                       prompt_reasoning_open=open_reasoning, cap=manifest['horizon'])}
            rows.append(row); all_rows.append(row)
        arms = {}
        for arm in sorted({r['arm'] for r in rows}):
            block = [r for r in rows if r['arm'] == arm]
            observed = [r['reasoning_tokens'] for r in block if r['reasoning_tokens'] is not None and not r['error']]
            arms[arm] = {'assigned': len(block), 'observed_reasoning': len(observed),
                'mean_generated_reasoning': sum(observed) / len(observed) if observed else None,
                'mean_total_generated': sum(r['generated_tokens'] for r in block) / len(block),
                'finish_counts': dict(Counter(r['finish'] for r in block)),
                'boundary_counts': dict(Counter(r['boundary_status'] for r in block))}
        summaries[lane] = {'manifest_path': str(path), 'manifest_sha256': manifest['sha256'],
            'stage_completion_sha256': stage['sha256'], 'assigned': len(rows), 'horizon': manifest['horizon'],
            'arms': arms, 'batch_receipts': receipts,
            'interpretation': 'Generated continuation region only; pretreatment prefix excluded. Shorter horizon than original-prompt 16384-token panel; no common full-answer accuracy endpoint.'}
    csv_path = out / 'continuation_lengths.csv'
    fields = ['lane', 'uid', 'family', 'seed', 'arm', 'continuation_cap', 'finish', 'error',
              'generated_tokens', 'reasoning_tokens', 'answer_tokens', 'boundary_status', 'closure']
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(all_rows)
    value = P.save(out / 'CONTINUATION_LENGTHS.json', {'schema': 'operator-panel-legacy-lengths-v1',
        'panel_plan_sha256': panel['sha256'], 'boundary': boundary, 'stages': summaries,
        'artifacts': {csv_path.name: P.U.file_sha(csv_path)}})
    print('LEGACY_LENGTHS', value['sha256'], len(all_rows), 'assigned continuations', flush=True)
