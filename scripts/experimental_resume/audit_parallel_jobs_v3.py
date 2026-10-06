"""Extend v2 allocation accounting to the nested utility grading receipts."""
from __future__ import annotations

import audit_parallel_jobs_v2 as previous


def main():
    original_paths = previous.receipt_paths

    def receipt_paths():
        paths = set(original_paths())
        for directory in previous.shared.RUNS.glob('utility-production-v1-*/outcomes-v3-*/dispatch'):
            paths.update(directory.glob('*.json'))
        return sorted(paths)

    previous.receipt_paths = receipt_paths
    previous.main()


if __name__ == '__main__':
    main()
