"""Offline strict/J1 export using the existing frozen scorer in a fresh process."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def main():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('offline grading requires allocated CPU step')
    root = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
    from diagnose_mechanism_validation_v3 import pin_qualified_worker
    pin_qualified_worker(root / 'addenda/ordered/9727c10299b71e7a/moe_exp_src')
    sys.path.insert(0, str(root / 'code/s1-9a61e32f48c04c24'))
    from moe_steer import score
    from rate_overnight_semantics_v2 import sealed, save
    source, output = Path(sys.argv[1]), Path(sys.argv[2])
    value = sealed(source)
    bundle = value['bundle']
    score.assert_blind(bundle, 'routing v2 offline grade bundle')
    strict = score.strict_rows(bundle, workers=2)
    items = score.build_j1_items(bundle, {r['uid']: r for r in strict}, corpus='routing-study-local-v2')
    score.assert_blind(items, 'routing v2 J1 items')
    files = {name: score.write_jsonl(output / name, rows) for name, rows in
        [('blind.jsonl.gz', bundle), ('strict.jsonl.gz', strict), ('j1-items.jsonl', items)]}
    save(output / 'GRADE_PREPARATION.json', {'schema': 'routing-v2-offline-grade-preparation',
        'blind_input_sha256': value['sha256'], 'files': files, 'strict_count': len(strict), 'J1_items': len(items),
        'status': 'PENDING_BOUND_AND_PRICED_J1' if items else 'STRICT_COMPLETE',
        'scorer_contract': score.SCORER_CONTRACT, 'scorer_file_sha256': score.sha256_file(score.__file__),
        'grader_visibility': 'Only arm-blind bundle; gold is offline only.'})
    print(json.dumps({'strict': len(strict), 'J1_pending': len(items)}), flush=True)


if __name__ == '__main__':
    main()
