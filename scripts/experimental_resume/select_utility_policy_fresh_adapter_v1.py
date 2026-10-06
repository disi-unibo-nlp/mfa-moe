"""Check fresh recovery evidence before running the unchanged v2 selector."""
import argparse
from pathlib import Path

import fresh_frame_recovery_v1 as recovery
import select_utility_policy_v2 as frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'analysis', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    m, analysis, amendment = map(recovery.base.sealed, (args.manifest, args.analysis, recovery.AMENDMENT))
    provenance = recovery.validate_analysis(m, analysis, amendment)
    result = frozen.select(m, analysis, recovery.base.sealed(frozen.FRESH_DESIGN),
                           recovery.base.sealed(frozen.MEASUREMENT_FREEZE))
    body = {k: v for k, v in result.items() if k != 'sha256'}
    body.update(recovery_provenance=provenance,
                selection_operational_entry_sha256=recovery.base.file_sha(__file__))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    saved = frozen.storage.save(args.out, body, existing_ok=True)
    print(saved['status'], saved['sha256'], flush=True)


if __name__ == '__main__':
    main()
