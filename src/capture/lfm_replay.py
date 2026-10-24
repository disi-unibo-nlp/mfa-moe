"""LFM2.5 router capture: replay the sampled discovery traces through the hooked engine, one .npz per trace.

Started by `python -m steering.pulse lfm-replay` (a GPU job). `rows` are the sampled traces (their saved source line and
attempt id); each capture goes to `<bulk_root>/lfm25/replay/<attempt_id>.npz` and the receipts are returned.
"""
from pathlib import Path
import time
from store import require


def replay(binding, rows, deadline):
    from steering.runtime import Backend
    from store import read_source
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
