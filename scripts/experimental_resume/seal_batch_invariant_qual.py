"""Seal a separate batch-invariant engine qualification from the frozen pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_boundary_micro_screen as base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    # Use the repository root explicitly; the sealer is run from any cwd.
    repo = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
    source = base.sealed(repo / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_QUAL4_MANIFEST_v3.json')
    wrapper = repo / 'scripts/experimental_resume/run_batch_invariant_qual.py'
    body = {key: value for key, value in source.items() if key != 'sha256'}
    body['schema'] = 'routing-batch-invariant-qual-v1'
    body['source_pilot_sha256'] = source['sha256']
    body['batch_invariant_flag'] = '1'
    body['bi_driver_sha256'] = base.file_sha(wrapper)
    body['code_files'] = {**body['code_files'], str(wrapper): base.file_sha(wrapper)}
    sealed = {**body, 'sha256': base.digest(body)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        if base.sealed(args.out) != sealed:
            raise ValueError('existing batch-invariant manifest differs')
    else:
        args.out.write_text(json.dumps(sealed, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'path': str(args.out), 'sha256': sealed['sha256'],
                      'requests': sealed['expected_requests'],
                      'prefill': sealed['expected_prefill_tokens'],
                      'maximum_decode': sealed['maximum_decode_tokens']}))


if __name__ == '__main__':
    main()
