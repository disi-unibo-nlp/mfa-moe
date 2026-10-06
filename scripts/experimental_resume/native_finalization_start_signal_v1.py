"""A single CPU dependency-release event; no model imports or scheduler loops."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]


def main(job, version):
    if not re.fullmatch(r'[0-9]+', job) or not re.fullmatch(r'correction-v[0-9]+', version):
        raise ValueError('invalid start-signal binding')
    doc = REPO / 'report/experimental-resume-v1/native-finalization-v1' / version
    manifest = json.loads((doc / 'MANIFEST.json').read_text())
    raw = subprocess.check_output(['sacct', '-X', '-nP', '-j', job,
        '--format=JobIDRaw,State,ExitCode,Start,AllocTRES'], universal_newlines=True)
    body = {'schema': 'native-finalization-start-signal-v1', 'signal': 'START_DEPENDENCY_RELEASED',
        'watched_job_id': job, 'producer_job_id': os.environ['SLURM_JOB_ID'],
        'manifest_sha256': manifest['sha256'], 'observed_utc': datetime.now(timezone.utc).isoformat(),
        'exact_job_readback': raw, 'role': 'One-shot orchestration signal; excluded from scientific outcomes.'}
    body['sha256'] = hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    root = doc / 'health'; root.mkdir(exist_ok=True)
    path = root / (job + '-start-' + os.environ['SLURM_JOB_ID'] + '.json')
    with path.open('x') as stream:
        json.dump(body, stream, indent=1); stream.write('\n')
    print(json.dumps({'health_receipt': str(path), **body}), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
