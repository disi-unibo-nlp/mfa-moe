"""Independently confirm saved native captures and replay diagnostic GPU selections.

Historical failure reproduction is reported separately from corrected capture
acceptance. The original failed receipt and every saved artifact stay immutable.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import socket
from pathlib import Path

from measurement_study.production_plan import ROOT, atomic, digest, sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_workers(workers, record):
    require(len(workers) == 2 and {w['rank'] for w in workers} == {0, 1}, 'Missing TP worker')
    expected = {f'{layer}/{signal}' for layer in record['layers'] for signal in ('ids', 'weights', 'router', 'selection')}
    for worker in workers:
        require(worker['tokens'] == record['completion_tokens'], 'Incomplete token coverage')
        require(worker['layers'] == record['layers'], 'Incomplete routed layer coverage')
        require(set(worker['hashes']) == expected, 'Missing native signal hashes')
        require(set(worker['dispatch_calls']) == {str(x) for x in record['layers']}, 'Missing dispatch evidence')
        require(min(worker['dispatch_calls'].values()) > 0, 'Missing native dispatch')
    require(workers[0]['hashes'] == workers[1]['hashes'], 'TP native signals disagree')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    require(bool(os.environ.get('SLURM_JOB_ID')) and socket.gethostname().startswith('lrdn'), 'GPU Slurm allocation required')
    import torch
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import fused_grouped_topk
    from measurement_study.native_selection import selection_scores, require_native_boundary, boundary_gaps
    from measurement_study.vllm_capture import digest_tensor
    require(torch.cuda.device_count() == 2, 'Exactly two allocated GPUs required')
    for rel, expected in json.loads((args.bundle / 'SHA256.json').read_text()).items():
        require(sha(args.bundle / rel) == expected, 'Frozen confirmation code changed')
    inputs = json.loads((args.bundle / 'nemotron-confirmation-inputs.json').read_text())
    for path, expected in inputs['input_hashes'].items():
        require(sha(path) == expected, 'Confirmation input or installed selector changed: ' + path)
    parent = Path(inputs['parent_bundle'])
    for rel, expected in json.loads((parent / 'SHA256.json').read_text()).items():
        require(sha(parent / rel) == expected, 'Original validation bundle changed')
    campaign = json.loads((parent / 'campaign.json').read_text())
    for path, expected in campaign['runtime_code_hashes'].items():
        require(sha(path) == expected, 'Shared capture runtime changed')
    spec = campaign['sources']['nemotron']
    config = json.loads((Path(spec['checkpoint']) / 'config.json').read_text())
    require(sha(Path(spec['checkpoint']) / 'config.json') == spec['config_sha256'], 'Checkpoint config changed')
    require(config['n_group'] == config['topk_group'] == 1 and config['norm_topk_prob'], 'Unsupported selector configuration')
    previous = json.loads(Path(inputs['parent_acceptance']).read_text())
    require(previous['bundle_sha256'] == sha(parent / 'SHA256.json'), 'Parent acceptance binding changed')
    fixtures = json.loads((parent / 'nemotron-repro.json').read_text())
    require(len(previous['records']) == len(fixtures) == 4, 'Missing shard fixture')
    records = []
    probes = []
    reproduced = []
    for record, fixture in zip(previous['records'], fixtures):
        require(record['id'] == fixture['row']['id'] and record['shard'] == fixture['shard'], 'Wrong failed-attempt fixture')
        require(record['corrected_min_gap'] >= -1e-6 and 'error' not in record, 'Original corrected capture failed')
        audit_path = Path(inputs['parent_acceptance']).parent / ('repro-' + record['id'] + '.native-audit.json')
        workers = json.loads(audit_path.read_text())['workers']
        validate_workers(workers, record)
        packed = {}
        for info in record['files']:
            path = Path(info['path'])
            require(path.stat().st_size == info['bytes'] and sha(path) == info['sha256'], 'Capture file changed')
            raw = gzip.decompress(path.read_bytes())
            require(hashlib.sha256(raw).hexdigest() == info['raw_sha256'], 'Raw capture hash changed')
            if path.name.endswith('.pt.gz'):
                packed['tensor'] = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
            else:
                packed['packet'] = json.loads(raw)
            del raw
        tensors, packet = packed['tensor'], packed['packet']
        require(packet['id'] == record['id'] and packet['native_dispatch_audit'] == workers, 'Packet audit mismatch')
        manifest = fixture['manifest'].copy()
        manifest['code_binding'] = sha(parent / 'SHA256.json')
        manifest['binding'] = digest({k: v for k, v in manifest.items() if k != 'binding'})
        require(packet['manifest_binding'] == manifest['binding'], 'Capture manifest mismatch')
        ids, weights, gaps = (tensors[k] for k in ('ids', 'weights', 'native_boundary_gaps'))
        require(tuple(ids.shape) == (len(spec['layers']), record['completion_tokens'], spec['top_k']), 'Incomplete IDs')
        require(weights.shape == ids.shape and gaps.shape == ids.shape[:2], 'Incomplete weights or gaps')
        require(bool(((ids >= 0) & (ids < spec['experts'])).all()), 'Expert ID outside topology')
        require(bool((ids.sort(-1).values.diff(dim=-1) > 0).all()), 'Duplicate dispatched expert')
        require(bool(torch.isfinite(weights).all() and torch.isfinite(gaps).all() and (gaps >= 0).all()), 'Nonfinite weights or invalid gaps')
        for i, layer in enumerate(spec['layers']):
            require(any(digest_tensor(ids[i].to(dtype)) == workers[0]['hashes'][f'{layer}/ids'] for dtype in (torch.int32, torch.int64)), 'Packed native IDs changed')
            require(digest_tensor(weights[i]) == workers[0]['hashes'][f'{layer}/weights'], 'Packed native weights changed')
        features = packet['trace']['metadata']['correlation_features']
        require(features['token_count'] == record['completion_tokens'] and features['layer_indices'] == spec['layers'], 'Reduced feature coverage mismatch')
        for device in range(2):
            for label, diagnostic in workers[0]['selector_diagnostics'].items():
                require(diagnostic['fused_grouped'] and diagnostic['score_dtype'] == diagnostic['bias_dtype'] == 'torch.float32', 'Unexpected native selector dtype')
                logits = torch.tensor([diagnostic['raw_logits']], dtype=torch.float32, device=f'cuda:{device}')
                bias = torch.tensor(diagnostic['bias'], dtype=torch.float32, device=logits.device)
                # Nemotron passes apply_routed_scale_to_output=True: its 2.5
                # model scale acts after the expert kernel; dispatch weights sum to 1.
                actual_weights, actual_ids = fused_grouped_topk(
                    torch.zeros((1, config['hidden_size']), device=logits.device, dtype=torch.bfloat16),
                    logits, spec['top_k'], True, bias, num_expert_group=1,
                    topk_group=1, scoring_func='sigmoid', routed_scaling_factor=1.0)
                expected_ids = torch.tensor([diagnostic['native_ids']], device=logits.device)
                expected_weights = torch.tensor([diagnostic['native_weights']], device=logits.device)
                require(torch.equal(actual_ids.long(), expected_ids), 'Native kernel did not reproduce saved dispatch IDs')
                error = float((actual_weights - expected_weights).abs().max())
                require(error <= 1e-7, 'Native kernel did not reproduce saved dispatch weights')
                scores = selection_scores(logits, bias, scoring='sigmoid', fused_grouped=True)
                require_native_boundary(scores, actual_ids)
                old_gap = float(boundary_gaps(logits.cpu().sigmoid() + bias.cpu(), actual_ids.cpu())[0])
                new_gap = float(boundary_gaps(scores, actual_ids)[0])
                require(old_gap == diagnostic['old_boundary_discrepancy'] and new_gap == diagnostic['corrected_boundary_discrepancy'], 'Boundary diagnostic changed')
                probes.append(dict(shard=record['shard'], device=device, label=label,
                    layer=diagnostic['layer'], completion_token=diagnostic['completion_token'],
                    absolute_token=diagnostic['absolute_token'], old_gap=old_gap,
                    corrected_gap=new_gap, weights_max_abs_error=error, native_ids=diagnostic['native_ids']))
                if old_gap < -1e-6 and device == 0:
                    reproduced.append(dict(shard=record['shard'], diagnostic=diagnostic))
        records.append({**record, 'status': 'passed', 'historical_reproduction_status': 'reproduced' if record['old_mismatch_reproduced'] else 'not_reproduced'})
        del packed, tensors, packet
    require(bool(reproduced), 'No historical numerical counterexample reproduced')
    result = dict(status='passed', job_id=os.environ['SLURM_JOB_ID'],
        records=records, startup_seconds=previous['startup_seconds'],
        capture_bundle_sha256=previous['bundle_sha256'], bundle_sha256=sha(args.bundle / 'SHA256.json'),
        parent_acceptance=inputs['parent_acceptance'], parent_acceptance_sha256=sha(inputs['parent_acceptance']),
        threshold_unchanged=1e-6, gpu_selector_probes=probes,
        reproduced_offenders=reproduced,
        historical_reproduction_complete=all(r['old_mismatch_reproduced'] for r in records),
        acceptance_basis='All four exact-token native captures pass unchanged numerical gates; saved IDs, weights, layer/token coverage and TP hashes independently verified; installed GPU selector replays every saved diagnostic on both A100s.',
        limitation='Original failure did not recur on three shard fixtures; original transient raw captures were not retained. This receipt does not claim their historical offending layer/token is known.')
    output = ROOT / 'results/measurement-study/recovery-storage/validation/nemotron-confirmation-v1/acceptance.json'
    atomic(output, result)
    print(json.dumps(dict(status=result['status'], output=str(output), captures=len(records), gpu_probes=len(probes), historical_reproduction_complete=result['historical_reproduction_complete'])), flush=True)


if __name__ == '__main__':
    main()
