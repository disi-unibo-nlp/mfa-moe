"""Pin the immutable worker namespace before importing any overnight validation code.

An operational recovery entry point. It never changes a frozen runner, manifest,
sampler, policy or output UID. Each invocation records its exact source hashes.
"""
from __future__ import annotations


def main():
    import hashlib
    import importlib
    import json
    import os
    from pathlib import Path
    import sys
    import traceback

    allowed={'overnight_routing_runner_v2'}
    if len(sys.argv)<3 or sys.argv[1] not in allowed:
        raise ValueError('entry requires an allowlisted runner module and its arguments')
    module=sys.argv.pop(1)
    overlay=Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/'
                 'steering-v1/addenda/ordered/9727c10299b71e7a/moe_exp_src').resolve()
    import diagnose_mechanism_validation_v3 as pin
    expected=pin.pin_qualified_worker(overlay)
    from moe_steer import qualify
    original=qualify.finish_child
    def finish(code,driver=None):
        if code:
            kind,error,trace=sys.exc_info()
            if error is not None:traceback.print_exception(kind,error,trace,file=sys.stderr)
            else:print('overnight entry: nonzero exit without active Python exception',file=sys.stderr)
            sys.stderr.flush()
        return original(code,driver)
    qualify.finish_child=finish
    runner=importlib.import_module(module)
    from overnight_routing_adjudicated_v3 import install
    install()
    if Path(sys.modules['moe_exp'].__file__).resolve().parent!=expected:
        raise RuntimeError('validation rebound the qualified moe_exp package')
    if '--startup-import-check' in sys.argv:
        # Exercise the actual preparation order through engine.apply_env and
        # worker resolution, without fingerprinting, torch import or model load.
        sys.argv.remove('--startup-import-check')
        from moe_steer import engine,manifests as M
        if module=='overnight_routing_qualify_v2':
            manifest=runner.validate_manifest(runner.base.sealed(runner.MANIFEST))
            table=runner.common.build_policy_table(manifest['actions'])
            count=manifest['maximum_requests'];M.load_world()
        else:
            manifest_path=Path(sys.argv[sys.argv.index('--manifest')+1])
            manifest=runner.base.sealed(manifest_path)
            rows,actions,arms=runner.validate(manifest,runner.base.__file__)
            builder=getattr(runner,'build_policy_table',None) or runner.common.build_policy_table
            table=builder(actions)
            count=len(runner.build_requests(manifest,rows,arms,M.load_world(),table))
        env=engine.engine_env('/unused/read-only-policy.json','/unused/read-only-telemetry')
        env['PYTHONPATH']=os.pathsep.join((str(overlay),env['PYTHONPATH']))
        runner.base.prepare_worker_import_path(engine,env,overlay)
        spec=importlib.util.find_spec('moe_exp.routing_control.ordered_vllm')
        print(json.dumps({'status':'PASS_EXACT_STARTUP_IMPORT_ORDER','requests':count,
              'moe_exp':sys.modules['moe_exp'].__file__,'worker':spec.origin,
              'model_load':False,'source_module':module}),flush=True)
        return
    files={str(Path(path).resolve()):hashlib.sha256(Path(path).read_bytes()).hexdigest()
           for path in (__file__,pin.__file__,runner.__file__)}
    receipt={'schema':'overnight-adjudicated-entry-v3','module':module,'files':files,
             'worker_overlay':str(overlay),'job_id':os.environ.get('SLURM_JOB_ID'),
             'array_task_id':os.environ.get('SLURM_ARRAY_TASK_ID'),
             'mode':'qualified worker pinned before repository-only validation imports'}
    print(json.dumps(receipt,sort_keys=True),flush=True)
    if os.environ.get('OVERNIGHT_ENTRY_RECEIPT_DIR'):
        directory=Path(os.environ['OVERNIGHT_ENTRY_RECEIPT_DIR']).resolve()
        if not str(directory).startswith('/leonardo_work/IscrC_MIOSR/lmolfett/'):
            raise ValueError('entry receipts must remain in private user work storage')
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/f'entry-{os.getpid()}.json').open('x') as stream:
            stream.write(json.dumps(receipt,indent=1)+'\n')
    runner.main()


if __name__=='__main__':main()
