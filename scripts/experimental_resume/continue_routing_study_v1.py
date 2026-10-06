"""One native-login utility step, then attach final CPU closeout when qualified.

Invoke again after a CPU readiness producer finishes. No daemon, background
polling, cached budget, GPU replay, or additional recovery wave is introduced.
"""
from pathlib import Path
import argparse
import json

import routing_study_closeout_v2 as C

PLAN = C.OUT / 'CONTINUATION.json'


def freeze():
    n = C.native(); n.login_guard()
    closeout = C.validate_plan()
    operations, _ = n.validate_operations(C.OPERATIONS)
    value = C.save(PLAN, {'schema': 'routing-study-finite-continuation-v1',
        'code_files': {str(Path(__file__).resolve()): C.sha(__file__)},
        'closeout_plan_sha256': closeout['sha256'], 'native_operations_sha256': operations['sha256'],
        'scope': 'One qualified native readiness dispatch per invocation, then at most one final two-CPU closeout checkpoint. Existing receipts are reused; no persistent process or expanded GPU workload.'})
    print(json.dumps({'status': 'FROZEN', 'sha256': value['sha256'], 'plan': str(PLAN)}))


def step(submit):
    n = C.native(); n.login_guard()
    plan, closeout = C.sealed(PLAN), C.validate_plan()
    C.verify_files(plan)
    operations, _ = n.validate_operations(C.OPERATIONS)
    C.require(plan['closeout_plan_sha256'] == closeout['sha256'] and
              plan['native_operations_sha256'] == operations['sha256'], 'continuation source bindings changed')
    C.advance(submit)
    root = Path(operations['native_root'])
    if not (root / 'J1_NATIVE_DISPATCH.json').exists():
        return
    ready, done = C.sealed(root / 'J1_READY.json'), C.sealed(root / 'J1_NATIVE_DISPATCH.json')
    chain = C.sealed(Path(ready['grade_root']) / 'dispatch/CHAIN.json')
    C.require(done['readiness_sha256'] == chain['readiness_sha256'] == ready['sha256'] and
              done['result_sha256'] == chain['sha256'] and
              chain['native_operations_sha256'] == operations['sha256'], 'final J1 dispatch changed')
    receipt = C.OUT / 'checkpoint-final.json'
    if receipt.exists():
        value = C.sealed(receipt)
        C.require(value['binding']['plan_sha256'] == closeout['sha256'], 'final checkpoint rebound')
        print(json.dumps({'status': 'FINAL_CHECKPOINT_ALREADY_ATTACHED', 'job_id': value['job_id']}))
        return
    fresh, dense = C.sealed(closeout['fresh_chain']), C.sealed(closeout['dense_chain'])
    jobs = [fresh['analysis_job'], dense['analysis_job'], '59380077', '59380073', '59380092', chain['finalize_job']]
    rows = C.accounting(jobs)['rows']
    active = []
    terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'}
    for job in jobs:
        C.require(job in rows, 'missing exact final predecessor accounting')
        if rows[job]['state'] not in terminal:
            active.append(job)
    if submit:
        C.submit_checkpoint('final', active)
    else:
        print(json.dumps({'status': 'FINAL_CHECKPOINT_PREPARED_NO_SUBMISSION', 'active_predecessors': active}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'step'))
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if args.mode == 'freeze': freeze()
    else: step(args.submit)


if __name__ == '__main__':
    main()
