"""Measurement/run addendum: bind an output directory before invoking the unchanged s2 runner.

The legacy UID omits caps. Never invoke the legacy runner on a reusable arbitrary directory.
This wrapper refuses unbound nonempty directories and manifests with differing bytes/digests.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path


def check_completion(fin, flight, *, policy, fsm_factory, runner, results):
    """Fail before accepting a malformed cap, routed array or forced pulse."""
    import numpy as np
    request = flight.request
    if fin.token_ids is None or fin.finish_reason in results.INFRA_FINISH:
        return  # The legacy retry accounting handles infrastructure failures.
    ids = list(fin.token_ids)
    cap = request['max_tokens']
    natural = bool(ids) and results.is_natural_stop(fin.finish_reason, ids[-1])
    if len(ids) > cap or (len(ids) != cap and not natural):
        raise runner.RunnerError(f"{request['uid']}: {len(ids)} tokens violates cap {cap}/natural-stop contract")
    routed = np.asarray(fin.routed_experts)
    if routed.ndim != 3 or routed.shape[0] != len(ids) or routed.shape[2] != 8:
        raise runner.RunnerError(f"{request['uid']}: routed rows/top-k disagree with emitted tokens")
    if policy.operator.kind != 'force':
        return
    pulses, _ = results.replay_pulses(fsm_factory, runner._policy_schedule(policy, request),
        request['uid'], policy.schedule.hazard_ref, flight.prefix_ids, ids)
    prefix = len(flight.prefix_ids)
    for start, end in pulses:
        lo, hi = max(0, start - prefix), min(len(ids), end - prefix)
        for layer, targets in policy.targets.experts:
            if layer >= routed.shape[1]:
                raise runner.RunnerError('routed return does not contain absolute layer indices')
            for expert in targets:
                present = (routed[lo:hi, layer, :] == expert).any(axis=1)
                if (not present.all() if policy.operator.sign > 0 else present.any()):
                    raise runner.RunnerError(f"{request['uid']}: force membership failed at layer {layer}")


def bind_output(manifest: dict, out: Path, *, wrapper_sha256: str) -> None:
    from moe_steer.manifests import validate_manifest
    from moe_steer.spec import verify

    verify(manifest)
    validate_manifest(manifest)
    expected = {'schema': 'resume-output-binding-v1', 'manifest_sha256': manifest['sha256'],
                'code_tree': manifest['code_tree'], 'wrapper_sha256': wrapper_sha256}
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.binding.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        path = out / 'RUN_BINDING.json'
        if path.exists():
            if json.loads(path.read_text()) != expected:
                raise ValueError('output directory is bound to a different manifest or wrapper')
        else:
            pending = out / '.RUN_BINDING.pending'
            if any(p.name not in ('.binding.lock', pending.name) for p in out.iterdir()):
                raise ValueError('refusing to adopt an unbound nonempty output directory')
            # A crash before publication can leave only this private incomplete binding.
            # No results are accepted until the complete binding is atomically published.
            pending.unlink(missing_ok=True)
            with pending.open('x') as handle:
                json.dump(expected, handle, sort_keys=True)
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(pending, path)
        if json.loads(path.read_text()) != expected:
            raise ValueError('output binding readback failed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    args, rest = parser.parse_known_args(argv)
    if any(a == '--out' or a.startswith('--out=') for a in rest):
        parser.error('the wrapper supplies the digest-bound --out')
    from moe_steer import manifests, runner, results
    manifest = manifests.load_manifest(args.manifest)
    wrapper_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out = args.output_root / manifest['sha256'] / wrapper_sha
    bind_output(manifest, out, wrapper_sha256=wrapper_sha)
    # This separate addendum does not change the frozen runtime files. Check each
    # finished request before the unchanged runner writes an accepted result.
    original_finish = runner._finish
    def checked_finish(fin, flight, **kwargs):
        policy = kwargs['table'].policies[flight.request['policy_index']]
        check_completion(fin, flight, policy=policy, fsm_factory=kwargs['fsm_factory'],
                         runner=runner, results=results)
        return original_finish(fin, flight, **kwargs)
    runner._finish = checked_finish
    return runner.main(['--manifest', str(args.manifest), '--out', str(out), *rest])


if __name__ == '__main__':
    raise SystemExit(main())
