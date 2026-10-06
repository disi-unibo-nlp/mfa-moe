"""Seal the distinct serial/eager six-arm engineering qualification manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_boundary_micro_screen as base
import run_boundary_micro_serial_qual_v1 as serial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    pilot = base.sealed(serial.PILOT)
    body = {key: value for key, value in pilot.items() if key != 'sha256'}
    body['schema'] = 'routing-boundary-serial-engine-qual-v1'
    body['source_pilot_sha256'] = pilot['sha256']
    body['entry_driver_sha256'] = base.file_sha(serial.__file__)
    body['engine_overrides'] = serial.ENGINE_OVERRIDES
    body['code_files'] = {**body['code_files'], str(Path(serial.__file__)): body['entry_driver_sha256']}
    value = {**body, 'sha256': base.digest(body)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        if base.sealed(args.out) != value:
            raise ValueError('existing serial qualification manifest differs')
    else:
        args.out.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'path': str(args.out), 'sha256': value['sha256'],
                      'requests': value['expected_requests'],
                      'prefill': value['expected_prefill_tokens'],
                      'maximum_decode': value['maximum_decode_tokens']}))


if __name__ == '__main__':
    main()
