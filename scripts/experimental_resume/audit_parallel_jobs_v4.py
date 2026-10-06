"""Include restored dense and sensitivity chains in allocation accounting."""
from __future__ import annotations

import audit_parallel_jobs_v2 as original


def main():
    prior_paths = original.receipt_paths

    def receipt_paths():
        paths = set(prior_paths())
        for pattern in (
            'utility-production-v1-*/outcomes-v3-*/dispatch',
            'generated-dense-fresh-metadata-recovery-v1/dense-generated-*/dispatch-v1',
        ):
            for directory in original.shared.RUNS.glob(pattern):
                paths.update(directory.glob('*.json'))
        for pattern in ('fresh-veto-sensitivity-chain-*', 'utility-account-transport-probe-*'):
            for directory in original.shared.DOC.glob(pattern):
                paths.update(directory.glob('*.json'))
        return sorted(paths)

    original.receipt_paths = receipt_paths
    original.main()


if __name__ == '__main__':
    main()
