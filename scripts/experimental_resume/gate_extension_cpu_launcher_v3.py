"""Verify a partition-only CPU launcher amendment without mutating sealed v2."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import mechanism_extension_reader_contract_v2 as contract
import rate_mechanism_extension_start_readers_v2 as reader
from freeze_mechanism_extension_220_v1 import sealed

AMENDMENT = contract.DOC / 'MECHANISM_EXTENSION_CPU_LAUNCH_AMENDMENT_v3.json'


def verify(kind):
    value = sealed(AMENDMENT)
    _, _, _, _, manifest = reader.expected_manifest()
    contract.require(value['source_qualification_manifest_sha256'] == manifest['sha256'] and
                     value['gate_driver_sha256'] == contract.file_sha(__file__) and
                     value['partition'] == 'lrd_all_viz', 'CPU launcher amendment changed')
    entry = value['launchers'][kind]
    contract.require(contract.file_sha(Path(entry['path'])) == entry['sha256'],
                     'CPU launcher bytes differ from sealed amendment')
    print(json.dumps({'cpu_launcher_amendment_sha256': value['sha256'],
                      'kind': kind, 'qualification_manifest_sha256': manifest['sha256']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('preflight', 'seal'))
    verify(parser.parse_args().kind)
