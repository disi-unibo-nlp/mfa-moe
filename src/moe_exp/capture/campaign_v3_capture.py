"""Immutable v3 capture execution and scientific acceptance gates."""
from __future__ import annotations
import argparse
import gc
import io
import json
import os
from pathlib import Path
import resource
import socket
import time
from moe_exp.capture.production_plan import ROOT, atomic, digest, sha, read_location
from moe_exp.capture.bundle import worker_init, verify_loaded_imports
from moe_exp.capture.campaign_v3_plan import fit_attempt_runtime

FAST_V3 = Path('/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3')
WORK_V3 = ROOT / 'results/measurement-study/campaign-v3'


def approved_output(path):
    from moe_exp.capture.campaign_v3_plan import validate_storage_root
    return validate_storage_root(path)


def claim_capture_output(manifest, out):
    binding = {'manifest_binding': manifest['binding'], 'code_binding': manifest['code_binding']}
    marker = Path(out) / 'run-binding.json'
    if marker.exists():
        if json.loads(marker.read_text()) != binding:
            raise ValueError('Capture inputs/code changed; choose a fresh output root')
    elif any(Path(out).iterdir()):
        raise ValueError('Unbound existing capture output; choose a fresh output root')
    else:
        atomic(marker, binding)


def differential_receipt(path, binding, result=None):
    path = Path(path)
    if result is not None:
        if result.get('status') != 'passed':
            raise ValueError('Required native differential did not pass')
        value = {'binding': binding, 'result': result}
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError('Existing differential differs; use a fresh verification root')
        atomic(path, value)
        return result
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    if saved.get('binding') != binding or saved.get('result', {}).get('status') != 'passed':
        raise ValueError('Stale or failed differential receipt')
    return saved['result']


def load_manifest(path, bundle):
    manifest = json.loads(Path(path).read_text())
    if digest({k: v for k, v in manifest.items() if k != 'binding'}) != manifest.get('binding'):
        raise ValueError('Capture manifest binding mismatch')
    if manifest.get('code_binding') != sha(Path(bundle) / 'SHA256.json'):
        raise ValueError('Capture source bundle mismatch')
    if manifest.get('spec', {}).get('capture_version') != 2:
        raise ValueError('V3 native capture spec required')
    if 'outcome_binding' in manifest:
        raise ValueError('Capture manifests must not bind outcomes; replan with the cohort-only schema')
    from moe_exp.capture.campaign_v3_plan import validate_cohort_binding
    validate_cohort_binding(manifest.get('cohort_binding'), manifest.get('cohort_manifest_path'))
    from moe_exp.capture.annotation_provenance import verify_annotation_inputs
    verify_annotation_inputs(manifest.get('annotation_binding'))
    if len({r['id'] for r in manifest['attempts']}) != len(manifest['attempts']):
        raise ValueError('Duplicate attempt IDs in capture manifest')
    return manifest


def resume_records(manifest, output):
    completed, remaining, used = [], [], 0
    for row in manifest['attempts']:
        marker = Path(output) / (row['id'] + '.complete.json')
        if not marker.exists():
            remaining.append(row)
            continue
        record = json.loads(marker.read_text())
        if (record.get('manifest_binding') != manifest['binding'] or
                record.get('id') != row['id'] or record.get('tokens') != row['completion_tokens'] or
                record.get('trace_sha256') != row['trace_sha256']):
            raise ValueError('Incompatible capture resume: ' + row['id'])
        if len(record.get('files', [])) != 2:
            raise ValueError('Incomplete artifact marker')
        for artifact in record['files']:
            path = Path(artifact['path']).resolve()
            if not path.is_relative_to(Path(output).resolve()):
                raise ValueError('Artifact escapes immutable capture output')
            if path.stat().st_size != artifact['bytes'] or sha(path) != artifact['sha256']:
                raise ValueError('Corrupt accepted capture: ' + str(path))
            used += artifact['bytes']
        completed.append(record)
    return completed, remaining, used


def validate_verification(manifest, receipt):
    if receipt.get('status') != 'passed' or receipt.get('validation_binding') != manifest.get('validation_binding'):
        raise ValueError('No accepted verify-only receipt for this spec/cohort')
    if not manifest.get('validation_binding'):
        raise ValueError('Missing verification binding')
    if receipt.get('maximum_input_tokens') != manifest['maximum_input_tokens']:
        raise ValueError('Longest input was not validated')
    if receipt.get('all_routed_layers') != manifest['spec']['layers'] or not receipt.get('exact_saved_token_coverage'):
        raise ValueError('Incomplete native verification coverage')
    if not receipt.get('unchanged_resume_verified'):
        raise ValueError('Unchanged resume was not validated')
    required = manifest['source'] in ('gpt', 'glm')
    if required and receipt.get('differential', {}).get('status') != 'passed':
        raise ValueError('Native differential verification required')


def capture_measurements(records, startup_seconds):
    """Conservative empirical sizing from accepted captures, including reduction.

    The two accepted traces give the fixed per-attempt cost and the per-token
    cost of the runtime model used for shard sizing and the deadline drain.
    """
    import math
    if not records or not math.isfinite(startup_seconds) or startup_seconds <= 0:
        raise ValueError('Positive measured model startup required')
    attempts=[]
    for record in records:
        tokens, seconds = record['tokens'], record['seconds']
        if type(tokens) is not int or tokens <= 0 or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Invalid measured capture runtime')
        size=sum(item['bytes'] for item in record['files'])
        if size <= 0:
            raise ValueError('Invalid measured compressed artifact size')
        attempts.append(dict(id=record['id'],tokens=tokens,seconds=seconds,artifact_bytes=size))
    rates=sorted(row['seconds']/row['tokens'] for row in attempts)
    fit=fit_attempt_runtime(attempts,startup_seconds)
    return dict(status='measured',attempts=attempts,startup_seconds=startup_seconds,
                p90_seconds_per_token=rates[math.ceil(.9*len(rates))-1],
                p90_method='empirical_nearest_rank; short and longest validation traces',
                seconds_per_token=fit['seconds_per_token'],
                per_attempt_seconds=fit['per_attempt_seconds'],
                runtime_model=fit,
                bytes_per_token=max(row['artifact_bytes']/row['tokens'] for row in attempts),
                footprint_method='maximum observed compressed bytes per completion token',
                startup_includes_differential=True)


def _trace(row):
    from moe_exp.schemas import TraceRecord
    return TraceRecord(**read_location(row.get('source_location') or row['cohorts'][0]['locations'][0]))


def _to_cpu(value):
    """Every tensor of a capture result, detached onto the CPU (recursively)."""
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(item) for item in value)
    return value


def _hf_reference_child(spec, out, row, path, bundle):
    """Spawned child: HF reference capture of the short trace, saved to `path`, then exit.

    Its own process guarantees the HF weights (which accelerate hooks keep alive after `del`)
    leave the GPUs before the vLLM engine claims its memory."""
    worker_init(Path(bundle))
    import torch
    from moe_exp.capture.production_capture import HFBackend
    reference = HFBackend(spec, Path(out)).capture(_trace(row), 'hf-reference')
    torch.save(_to_cpu(reference), path)


def _deadline_reached(manifest, row, observed_rate):
    deadline = os.environ.get('MEASUREMENT_DEADLINE_EPOCH')
    if not deadline:
        return False
    sizing = manifest.get('sizing') or {}
    if sizing.get('status') == 'unmeasured_verification':
        margin = float(sizing.get('drain_margin_seconds') or 1800)
        return time.time() + margin >= float(deadline)
    rate = max(float(sizing.get('seconds_per_token') or manifest.get('seconds_per_token') or 0), float(observed_rate or 0))
    if rate <= 0:
        raise ValueError('Deadline guard requires a measured capture rate')
    per_attempt = float(sizing.get('per_attempt_seconds') or manifest.get('per_attempt_seconds') or 0)
    expected = 1.25 * (per_attempt + rate * row['completion_tokens'])
    margin = max(float(sizing.get('drain_margin_seconds') or 900), expected) + 300
    return time.time() + margin >= float(deadline)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--storage-root', type=Path)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--shard', type=int)
    a = p.parse_args(argv)
    if not os.environ.get('SLURM_JOB_ID') or not socket.gethostname().startswith('lrdn'):
        raise RuntimeError('GPU Slurm allocation required')
    binding = worker_init(a.bundle)
    if not (a.bundle / 'campaign-v3.json').exists():
        raise ValueError('New campaign-v3 frozen bundle required')
    m = load_manifest(a.manifest, a.bundle)
    if m['source'] != a.source:
        raise ValueError('Wrong manifest source')
    import torch
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Exactly two allocated GPUs required')
    from moe_exp.capture.production_capture import HFBackend, VLLMBackend, reduce_capture, pack_bytes
    from moe_exp.capture.sentence_schema import install
    install()
    storage = approved_output(a.storage_root or m.get('storage_root') or FAST_V3)
    if a.storage_root and Path(m.get('storage_root', storage)).resolve() != storage:
        raise ValueError('Storage override differs from frozen manifest')
    out = approved_output(m.get('capture_dir') or storage / 'capture' / a.source / m['cohort'] / str(m['shard']))
    if not out.is_relative_to(storage):
        raise ValueError('Capture output escapes requested storage root')
    out.mkdir(parents=True, exist_ok=True)
    claim_capture_output(m, out)
    rows = m['attempts']
    if a.verify_only:
        if len(rows) != 2 or max(r['total_input_tokens'] for r in rows) != m['maximum_input_tokens']:
            raise ValueError('Verify manifest must contain corpus-longest and short traces')
        short = min(rows, key=lambda r: r['total_input_tokens'])
        if short['total_input_tokens'] > 32768 or short['id'] == max(rows, key=lambda r: r['total_input_tokens'])['id']:
            raise ValueError('Distinct short differential fixture <=32768 required')
    else:
        receipt = json.loads(Path(m['verification_path']).read_text())
        validate_verification(m, receipt)
    completed, remaining, used = resume_records(m, out)
    differential_binding = dict(validation_binding=m.get('validation_binding'), code_binding=m['code_binding'], short_trace_sha256=short['trace_sha256'], short_id=short['id']) if a.verify_only else None
    differential = differential_receipt(out/'differential.json',differential_binding) if a.verify_only and a.source in ('gpt','glm') else None
    if a.verify_only and completed and a.source in ('glm', 'gpt') and not differential:
        raise ValueError('Verification resume lacks differential evidence')
    backend = None
    if remaining:
        start = time.monotonic()
        reference = None
        if a.verify_only and a.source == 'glm' and differential is None and short['completion_tokens'] < 2048:
            raise ValueError('GLM differential trace too short for the routing-dynamics agreement check')
        if a.verify_only and a.source == 'glm' and differential is None:
            # Sequential model loads; never keep two model backends resident. The HF reference
            # runs in a spawned child: in-process `del` left ~30 GiB of HF weights per GPU and
            # the vLLM engine then refused to start.
            import multiprocessing
            ref_path = out / 'hf-reference.pt'
            child = multiprocessing.get_context('spawn').Process(
                target=_hf_reference_child, args=(m['spec'], str(out), short, str(ref_path), str(a.bundle)))
            child.start()
            child.join()
            if child.exitcode != 0 or not ref_path.exists():
                raise RuntimeError(f'HF reference capture failed (exit {child.exitcode})')
            reference = torch.load(ref_path, map_location='cpu', weights_only=False)
            ref_path.unlink()
        backend_type = m['spec'].get('backend', 'hf' if a.source in ('gpt', 'qwen330b') else 'vllm')
        if backend_type not in ('hf', 'vllm'):
            raise ValueError('Unapproved capture backend')
        backend = HFBackend(m['spec'], out) if backend_type == 'hf' else VLLMBackend(m['spec'], out, transient_cap_bytes=m['transient_cap_bytes'])
        if a.verify_only and differential is None:
            if a.source == 'glm':
                from moe_exp.capture.capture_buffers import compare_backend_captures, DifferentialFailed
                candidate = backend.capture(_trace(short), 'vllm-differential')
                try:
                    differential = compare_backend_captures(reference, candidate, total_input_tokens=short['total_input_tokens'], layers=m['spec']['layers'])
                except DifferentialFailed as failed:
                    # keep the full report and the compact routing tensors for offline diagnosis
                    atomic(out / 'differential-failed.json', dict(binding=differential_binding, result=failed.report))
                    torch.save({side: {key: _to_cpu(data[key]) for key in ('router', 'selection', 'ids', 'weights')}
                                for side, data in (('hf', reference), ('vllm', candidate))},
                               out / 'differential-failure.pt')
                    raise
                del reference, candidate
            elif a.source == 'gpt':
                from moe_exp.capture.hf_chunked_capture import capture_chunked, compare_native_captures
                trace = _trace(short)
                backend.validate_attention(trace, out)
                reference = backend.capture_single(trace, 'single-pass-differential')
                candidate = capture_chunked(backend.model, backend.tokenizer, trace, m['spec']['layers'], m['spec'].get('prefill_chunk_size', 512))
                differential = compare_native_captures(reference, candidate)
                del reference, candidate
            else:
                differential = {'status': 'not_required', 'reason': 'native dispatch/selection validation at all tokens'}
            if a.source in ('gpt','glm'):
                differential_receipt(out/'differential.json',differential_binding,differential)
        atomic(out / 'load.json', dict(seconds=time.monotonic()-start, source=a.source,
              job=os.environ['SLURM_JOB_ID'], provenance=backend.provenance, imports=verify_loaded_imports(a.bundle)))
        observed_rate = 0.
        for row in sorted(remaining, key=lambda r: (-r['total_input_tokens'], r['id'])):
            if _deadline_reached(m, row, observed_rate):
                atomic(out / 'drained.json', dict(status='drained', manifest_binding=m['binding'], completed=len(completed)))
                return 75
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            data = backend.capture(_trace(row), 'capture-' + row['id'])
            packet, tensors = reduce_capture(row, data, backend, m)
            del data
            stream = io.BytesIO(); torch.save(tensors, stream); del tensors
            tensor_record = pack_bytes(out / (row['id'] + '.pt.gz'), stream.getvalue(), m['artifact_cap_bytes']-used)
            del stream
            packet['tensor_artifact'] = tensor_record
            packet_record = pack_bytes(out / (row['id'] + '.json.gz'), json.dumps(packet, allow_nan=False).encode(), m['artifact_cap_bytes']-used-tensor_record['bytes'])
            del packet
            elapsed = time.monotonic() - start
            observed_rate = max(observed_rate, elapsed / max(1, row['completion_tokens']))
            record = dict(id=row['id'], manifest_binding=m['binding'], trace_sha256=row['trace_sha256'],
                          files=[tensor_record, packet_record], tokens=row['completion_tokens'], seconds=elapsed,
                          job=os.environ['SLURM_JOB_ID'],
                          parent_peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                          parent_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          imports=verify_loaded_imports(a.bundle))
            atomic(out / (row['id'] + '.complete.json'), record)
            completed.append(record); used += sum(f['bytes'] for f in record['files'])
            atomic(out / 'progress.json', dict(completed=len(completed), expected=m['expected_attempts'], tokens=sum(r['tokens'] for r in completed), artifact_bytes=used))
            print(json.dumps({'completed': len(completed), 'expected': m['expected_attempts'], 'last_seconds': elapsed}), flush=True)
    reread, unfinished, checked_bytes = resume_records(m, out)
    if unfinished or len(reread) != m['expected_attempts'] or sum(r['tokens'] for r in reread) != m['expected_tokens']:
        raise ValueError('Incomplete exact capture coverage')
    if a.verify_only:
        if a.source in ('gpt','glm') and (not differential or differential.get('status')!='passed'):
            raise ValueError('Required differential acceptance missing')
        atomic(out / 'verification.json', dict(status='passed', manifest_binding=m['binding'],
               validation_binding=m.get('validation_binding'), bundle_sha256=binding['bundle_sha256'],
               verified_ids=[r['id'] for r in rows], maximum_input_tokens=m['maximum_input_tokens'],
               all_routed_layers=m['spec']['layers'], exact_saved_token_coverage=True,
               unchanged_resume_verified=True, differential=differential,
               measurements=capture_measurements(reread,float(json.loads((out/'load.json').read_text())['seconds'])),
               parent_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               imports=verify_loaded_imports(a.bundle)))
    else:
        atomic(out / 'result.json', dict(status='complete_native_collection', source=a.source, cohort=m['cohort'], shard=m['shard'],
               manifest_binding=m['binding'], attempts=len(reread), tokens=sum(r['tokens'] for r in reread),
               artifact_bytes=checked_bytes, records=reread))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
