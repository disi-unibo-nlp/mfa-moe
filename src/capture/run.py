"""Capture the routers of one shard of one model: a GPU job with two GPUs.

Replays the saved prompt and completion token ids of every trace of a frozen manifest through the model and records,
for every routed layer and completion token, the router signal, the hidden state, the selected experts and their
weights. Each trace is reduced (features, routing windows, class views, expert counts) and written as two gzip files,
`<id>.pt.gz` (tensors) and `<id>.json.gz` (packet), followed by the marker `<id>.complete.json`. A rerun skips the
traces that have a marker and checks that their files are intact.

    python -m capture.run MODEL MANIFEST OUTPUT_DIR [--verify-only] [--deadline-epoch EPOCH]

`--verify-only` runs the 2-trace verification (longest and a short trace) that must pass before the shards are run:
gpt compares its query-chunked attention with the native one, glm compares the vLLM capture with a Hugging Face
reference; the others only check that every routed layer and every token is covered. Its receipt is
`OUTPUT_DIR/verification.json`. With `--deadline-epoch` (the end of the Slurm job) no new trace is started in the last
30 minutes: the run writes `drained.json` and exits with 75, and the same command resumes.

A repeat capture goes to a new tree: `ROOT/verify` for the verification (run first), then `ROOT/0`, `ROOT/1` ... for the
shards. A shard run there needs the receipt `ROOT/verify/verification.json`, written by the same code. The paper's
folders (`capture_dir` of the manifests) are only read, by the gate and the analyses: `run-binding.json` and
`differential.json` bind a folder to the manifest and the hash of the running code, so a folder is only resumed by
the code that started it, and a finished receipt is never overwritten.
"""
import argparse
import gzip
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import resource
import socket
import time

from capture.buffers import approved_capture_path
from capture.manifest import (
    DIFFERENTIAL_MODELS, atomic, capture_folders, load_manifest, read_location, resume_records, validate_verification,
    write_new,
)
from provenance import code_provenance
from store import file_sha


DRAIN_SECONDS = 30 * 60  # no new trace is started when less than this remains of the Slurm job


def must_stop(deadline_epoch):
    """True when less than DRAIN_SECONDS of the Slurm job remain (no deadline given: never)."""
    return bool(deadline_epoch) and time.time() > deadline_epoch - DRAIN_SECONDS


def claim_capture_output(manifest, out, code):
    """Mark an empty folder as the output of this manifest and this code; refuse any other folder."""
    binding = {'manifest_binding': manifest['binding'], 'source_sha256': code['source_sha256']}
    marker = Path(out) / 'run-binding.json'
    if marker.exists():
        if json.loads(marker.read_text()) != binding:
            raise ValueError('The folder was written for other inputs or by other code; choose a new folder')
    elif any(Path(out).iterdir()):
        raise ValueError('Unbound existing capture output; choose a new folder')
    else:
        atomic(marker, binding)


def prepare_output(manifest, output, code):
    """The folder a run writes to: an approved path, claimed by this manifest and this code, not yet finished.

    Returns (folder, repeat); `repeat` is True outside the manifest's own `capture_dir` (the paper's folder)."""
    out = approved_capture_path(output)
    repeat = out != Path(manifest['capture_dir']).resolve()
    if repeat and out.name != str(manifest['shard']):
        raise ValueError(f"A repeat capture goes to ROOT/{manifest['shard']} (shards: ROOT/<shard number>, check: ROOT/verify)")
    out.mkdir(parents=True, exist_ok=True)
    claim_capture_output(manifest, out, code)
    for name in ('verification.json', 'result.json'):
        if (out / name).exists():
            raise ValueError(f'{out} already holds {name}; a finished capture is not run again')
    return out, repeat


def check_verification(manifest, out, repeat, code):
    """A shard needs the passed 2-trace verification of its model; in a repeat tree one written by this code."""
    path = capture_folders(manifest, out.parent if repeat else None)[1]
    receipt = json.loads(path.read_text())
    validate_verification(manifest, receipt)
    if repeat and receipt.get('code', {}).get('source_sha256') != code['source_sha256']:
        raise ValueError(f'{path} was written by other code; verify again with this code')


def pack_bytes(path, raw, limit):
    """Atomic, lossless output; fail before publishing beyond the shard cap."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    try:
        with tmp.open('wb') as stream:
            with gzip.GzipFile(filename='', fileobj=stream, mode='wb', compresslevel=3, mtime=0) as zipped:
                zipped.write(raw)
        if tmp.stat().st_size > limit:
            raise RuntimeError('Shard artifact storage cap reached')
        check = hashlib.sha256()
        with gzip.open(tmp, 'rb') as stream:
            while block := stream.read(1024 * 1024):
                check.update(block)
        if check.hexdigest() != hashlib.sha256(raw).hexdigest():
            raise RuntimeError('Lossless read-back mismatch')
        tmp.replace(path)
        return dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha(path), raw_sha256=check.hexdigest())
    finally:
        tmp.unlink(missing_ok=True)


def saved_differential(path, binding):
    """The differential receipt of an earlier run of this verification, or None."""
    path = Path(path)
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    if saved.get('binding') != binding or saved.get('result', {}).get('status') != 'passed':
        raise ValueError('Stale or failed differential receipt')
    return saved['result']


def save_differential(path, binding, result):
    if result.get('status') != 'passed':
        raise ValueError('Required native differential did not pass')
    value = {'binding': binding, 'result': result}
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Existing differential differs; use a fresh verification root')
    atomic(path, value)


def _trace(row):
    from schemas import TraceRecord
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


def _hf_reference_child(spec, out, row, path):
    """Spawned child: HF reference capture of the short trace, saved to `path`, then exit.

    Its own process guarantees the HF weights (which accelerate hooks keep alive after `del`)
    leave the GPUs before the vLLM engine claims its memory."""
    import torch
    from capture.backends import HFBackend
    reference = HFBackend(spec, Path(out)).capture(_trace(row), 'hf-reference')
    torch.save(_to_cpu(reference), path)


def hf_reference(spec, out, short):
    import torch
    path = out / 'hf-reference.pt'
    child = multiprocessing.get_context('spawn').Process(
        target=_hf_reference_child, args=(spec, str(out), short, str(path)))
    child.start()
    child.join()
    if child.exitcode != 0 or not path.exists():
        raise RuntimeError(f'HF reference capture failed (exit {child.exitcode})')
    reference = torch.load(path, map_location='cpu', weights_only=False)
    path.unlink()
    return reference


def run_differential(source, manifest, out, short, backend, reference, binding):
    """The verification trace captured a second way must agree with the first (glm, gpt)."""
    import torch
    from capture.agreement import DifferentialFailed, compare_backend_captures, compare_native_captures
    from capture.hf_chunked_capture import capture_chunked
    if source == 'glm':
        candidate = backend.capture(_trace(short), 'vllm-differential')
        try:
            return compare_backend_captures(reference, candidate, total_input_tokens=short['total_input_tokens'],
                                            layers=manifest['spec']['layers'])
        except DifferentialFailed as failed:
            # keep the full report and the compact routing tensors for offline diagnosis
            atomic(out / 'differential-failed.json', dict(binding=binding, result=failed.report))
            torch.save({side: {key: _to_cpu(data[key]) for key in ('router', 'selection', 'ids', 'weights')}
                        for side, data in (('hf', reference), ('vllm', candidate))},
                       out / 'differential-failure.pt')
            raise
    if source == 'gpt':
        trace = _trace(short)
        backend.validate_attention(trace, out)
        single_pass = backend.capture_single(trace, 'single-pass-differential')
        chunked = capture_chunked(backend.model, backend.tokenizer, trace, manifest['spec']['layers'],
                                  manifest['spec'].get('prefill_chunk_size', 512))
        return compare_native_captures(single_pass, chunked)
    return {'status': 'not_required', 'reason': 'native dispatch/selection validation at all tokens'}


def capture_one(row, backend, manifest, out, used, code):
    """Replay, reduce and write one trace (`used` = bytes already written in this folder); returns its marker."""
    import torch
    from capture.reduce import reduce_capture
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    data = backend.capture(_trace(row), 'capture-' + row['id'])
    packet, tensors = reduce_capture(row, data, backend, manifest)
    del data
    stream = io.BytesIO()
    torch.save(tensors, stream)
    del tensors
    tensor_record = pack_bytes(out / (row['id'] + '.pt.gz'), stream.getvalue(), manifest['artifact_cap_bytes'] - used)
    del stream
    packet['tensor_artifact'] = tensor_record
    packet_record = pack_bytes(out / (row['id'] + '.json.gz'), json.dumps(packet, allow_nan=False).encode(),
                               manifest['artifact_cap_bytes'] - used - tensor_record['bytes'])
    del packet
    record = dict(id=row['id'], manifest_binding=manifest['binding'], trace_sha256=row['trace_sha256'],
                  files=[tensor_record, packet_record], tokens=row['completion_tokens'],
                  seconds=time.monotonic() - start, job=os.environ['SLURM_JOB_ID'],
                  parent_peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                  parent_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, code=code)
    atomic(out / (row['id'] + '.complete.json'), record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source')
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--deadline-epoch', type=float)
    args = parser.parse_args(argv)
    source = args.source
    if not os.environ.get('SLURM_JOB_ID') or not socket.gethostname().startswith('lrdn'):
        raise RuntimeError('GPU Slurm allocation required')
    manifest = load_manifest(args.manifest)
    if manifest['source'] != source:
        raise ValueError('Wrong manifest source')
    code = code_provenance()
    out, repeat = prepare_output(manifest, args.output, code)
    import torch
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Exactly two allocated GPUs required')
    from capture.backends import HFBackend, VLLMBackend
    from capture.sentence_schema import install
    install()
    rows = manifest['attempts']
    if args.verify_only:
        longest = max(rows, key=lambda r: r['total_input_tokens'])
        short = min(rows, key=lambda r: r['total_input_tokens'])
        if len(rows) != 2 or longest['total_input_tokens'] != manifest['maximum_input_tokens']:
            raise ValueError('Verify manifest must contain corpus-longest and short traces')
        if short['total_input_tokens'] > 32768 or short['id'] == longest['id']:
            raise ValueError('Distinct short differential fixture <=32768 required')
    else:
        check_verification(manifest, out, repeat, code)
    completed, remaining, used = resume_records(manifest, out)
    needs_differential = args.verify_only and source in DIFFERENTIAL_MODELS
    differential_binding = differential = None
    if args.verify_only:
        differential_binding = dict(validation_binding=manifest.get('validation_binding'),
                                    source_sha256=code['source_sha256'],
                                    short_trace_sha256=short['trace_sha256'], short_id=short['id'])
    if needs_differential:
        differential = saved_differential(out / 'differential.json', differential_binding)
        if completed and not differential:
            raise ValueError('Verification resume lacks differential evidence')
    if remaining:
        start = time.monotonic()
        reference = None
        if needs_differential and differential is None and source == 'glm':
            if short['completion_tokens'] < 2048:
                raise ValueError('GLM differential trace too short for the routing-dynamics agreement check')
            # Sequential model loads; never keep two model backends resident.
            reference = hf_reference(manifest['spec'], out, short)
        if manifest['spec']['backend'] == 'hf':
            backend = HFBackend(manifest['spec'], out)
        elif manifest['spec']['backend'] == 'vllm':
            backend = VLLMBackend(manifest['spec'], out, transient_cap_bytes=manifest['transient_cap_bytes'])
        else:
            raise ValueError('Unapproved capture backend')
        if args.verify_only and differential is None:
            differential = run_differential(source, manifest, out, short, backend, reference, differential_binding)
            reference = None
            if needs_differential:
                save_differential(out / 'differential.json', differential_binding, differential)
        atomic(out / 'load.json', dict(seconds=time.monotonic() - start, source=source,
                                       job=os.environ['SLURM_JOB_ID'], provenance=backend.provenance, code=code))
        # Longest input first: it shows the real memory need before the rest is spent.
        for row in sorted(remaining, key=lambda r: (-r['total_input_tokens'], r['id'])):
            if must_stop(args.deadline_epoch):
                atomic(out / 'drained.json', dict(status='drained', manifest_binding=manifest['binding'],
                                                  completed=len(completed)))
                return 75
            record = capture_one(row, backend, manifest, out, used, code)
            completed.append(record)
            used += sum(f['bytes'] for f in record['files'])
            atomic(out / 'progress.json', dict(completed=len(completed), expected=manifest['expected_attempts'],
                                               tokens=sum(r['tokens'] for r in completed), artifact_bytes=used))
            print(json.dumps({'completed': len(completed), 'expected': manifest['expected_attempts'],
                              'last_seconds': record['seconds']}), flush=True)
    reread, unfinished, checked_bytes = resume_records(manifest, out)
    if unfinished or len(reread) != manifest['expected_attempts'] or sum(r['tokens'] for r in reread) != manifest['expected_tokens']:
        raise ValueError('Incomplete exact capture coverage')
    if args.verify_only:
        if needs_differential and (not differential or differential.get('status') != 'passed'):
            raise ValueError('Required differential acceptance missing')
        write_new(out / 'verification.json', dict(
            status='passed', manifest_binding=manifest['binding'], validation_binding=manifest.get('validation_binding'),
            verified_ids=[r['id'] for r in rows], maximum_input_tokens=manifest['maximum_input_tokens'],
            all_routed_layers=manifest['spec']['layers'], exact_saved_token_coverage=True,
            unchanged_resume_verified=True, differential=differential,
            parent_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, code=code))
    else:
        write_new(out / 'result.json', dict(
            status='complete_native_collection', source=source, cohort=manifest['cohort'], shard=manifest['shard'],
            manifest_binding=manifest['binding'], attempts=len(reread), tokens=sum(r['tokens'] for r in reread),
            artifact_bytes=checked_bytes, records=reread))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
