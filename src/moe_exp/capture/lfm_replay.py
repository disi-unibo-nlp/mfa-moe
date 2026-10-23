"""LFM router capture: teacher-forced replay of the sampled discovery traces through the hooked engine.

Split out of the former `study.lfm`; the function is unchanged.
"""
from pathlib import Path
import time
from moe_exp.io import require


def replay(binding, rows, deadline):
    from moe_exp.steering.runtime import Backend
    from moe_exp.io import read_source
    bulk = Path(binding['config']['bulk_root'])
    backend = Backend(binding, 'lfm25', 1, deadline, replay=True)
    try:
        receipts = []
        for row in rows:
            require(time.time()+60 < deadline, 'replay allocation reserve')
            receipts.append(backend.capture_trace(read_source(row['source_location']), bulk/'lfm25/replay'/(row['attempt_id']+'.npz')))
        return {'status': 'complete', 'receipts': receipts}
    finally:
        backend.close()
