"""Pin immutable scoring namespace before any J1 orchestration imports."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

INSTALLED=None
OVERLAY=Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/addenda/ordered/9727c10299b71e7a/moe_exp_src')


def install():
    global INSTALLED
    if INSTALLED is not None:return INSTALLED
    from diagnose_mechanism_validation_v3 import pin_qualified_worker
    cached=sys.modules.get('moe_exp')
    if cached is None:pin_qualified_worker(OVERLAY)
    elif Path(cached.__file__).resolve()!=(OVERLAY/'moe_exp/__init__.py').resolve():
        raise RuntimeError('foreign scoring package cached before the immutable J1 entry')
    import utility_j1_chain_v1 as original
    original.PLAN=original.DOC/'UTILITY_J1_CHAIN_PLAN_v2.json'
    original.ENVELOPE=original.DOC/'UTILITY_J1_MAX384_ENVELOPE_v2.json'
    original.WRAPPERS={mode:original.SCRIPTS/('utility_j1_'+mode+'_v2.sbatch') for mode in original.MODES}
    old_code=original.code_files;old_binding=original.binding
    def files():
        extra=[Path(__file__),Path(pin_qualified_worker.__code__.co_filename)]
        return {**old_code(),**{str(p.resolve()):original.sha(p) for p in extra}}
    def binding():
        prior=original.sealed(original.DOC/'UTILITY_J1_CHAIN_PLAN_v1.json')
        return {**old_binding(),'schema':'utility-j1-chain-plan-v2','prior_unsubmitted_plan_sha256':prior['sha256'],
            'import_amendment':'Pin the immutable moe_exp overlay before shared dispatch/dense helpers import it. The frozen v1 scientific/pricing adapter and exact legacy GPU wrapper are unchanged.',
            'qualified_overlay':str(OVERLAY)}
    original.code_files=files;original.binding=binding
    old_attach=original.attach
    def attach(chain_dir,out_parent):
        if (Path(chain_dir)/'INITIAL_CHAIN.json').exists():return old_attach(chain_dir,out_parent)
        return follow_preproduction(original,Path(chain_dir),out_parent)
    original.attach=attach
    INSTALLED=original
    return original


def follow_preproduction(original,chain_dir,out_parent):
    """Follow at most three durable CPU successors; no daemon or polling."""
    plan=original.validate_plan()
    phases=('after-pilot','follow-pilot-analysis','follow-pilot-generation')
    candidates=[chain_dir/(phase+'-CHAIN.json') for phase in phases if (chain_dir/(phase+'-CHAIN.json')).exists()]
    original.require(candidates,'production has neither initial nor finite pilot-follow receipt')
    source=candidates[0];receipt=original.sealed(source)
    original.require(receipt['schema']=='utility-production-follow-pilot-v1' and
        chain_dir.name=='utility-production-submissions-v1-'+receipt['config_sha256'][:16] and
        receipt['next_phase'] in ('follow-pilot-analysis','after-pilot','price'),
        'production successor receipt config differs')
    directory=chain_dir/('j1-attachment-'+plan['sha256'][:12]);directory.mkdir(exist_ok=True)
    prepared=original.DOC/('utility-production-prepared-v1-'+receipt['config_sha256'][:16])
    if receipt['next_phase']=='price' and ((prepared/'HOLD.json').exists() or (prepared/'PRICE.json').exists()):
        # The price parent may intentionally decline to submit generation.
        def stopped():
            return {str(p):original.sealed(p)['sha256'] for p in (prepared/'HOLD.json',prepared/'PRICE.json') if p.exists()}
        dep,proof=original.dependencies.dependency(str(receipt['next_job']),stopped)
        if not dep:
            result=original.shared.save(directory/'PREPRODUCTION_HOLD.json',{'schema':'utility-j1-preproduction-hold-v2',
                'status':'HOLD_NO_PRODUCTION_INITIAL_RECEIPT','plan_sha256':plan['sha256'],
                'production_successor_sha256':receipt['sha256'],'proof':proof})
            print(json.dumps(result),flush=True);return
    target=chain_dir/(receipt['next_phase']+'-CHAIN.json') if receipt['next_phase']!='price' else chain_dir/'INITIAL_CHAIN.json'
    def completed():
        allowed=[target] if receipt['next_phase']!='price' else [target,prepared/'HOLD.json',prepared/'PRICE.json']
        artifacts={str(p):original.sealed(p)['sha256'] for p in allowed if p.exists()}
        original.require(artifacts,'successful production successor lacks its next receipt')
        return artifacts
    name='utility-j1-follow-'+receipt['next_phase']
    bound={'schema':'utility-j1-preproduction-follow-v2','plan_sha256':plan['sha256'],
           'config_sha256':receipt['config_sha256'],'successor_receipt_sha256':receipt['sha256']}
    with (directory/'WRITER.lock').open('a+') as lock:
        original.fcntl.flock(lock,original.fcntl.LOCK_EX|original.fcntl.LOCK_NB)
        original.verify.live_account_snapshot(directory,'LIVE_ACCOUNT-'+original.os.environ['SLURM_JOB_ID']+'.json')
        job=original.cpu_once(directory,name,[str(original.WRAPPERS['attach']),'--chain-dir',str(chain_dir),
            *(['--out-parent',str(out_parent)] if out_parent else [])],bound,str(receipt['next_job']),completed)
        original.shared.save(directory/(name+'-CHAIN.json'),{**bound,'next_attachment_job':job})
    print(json.dumps({'next_attachment_job':job,'next_phase':receipt['next_phase']}),flush=True)


def freeze(original):
    value=original.shared.save(original.PLAN,{'schema':'utility-j1-chain-plan-envelope-v2',
        'frozen_utc':datetime.now(timezone.utc).isoformat(),'binding':original.binding()})
    price=original.price_rows([{'item_id':format(i,'024x'),'prompt_tokens':16384} for i in range(384)],value['binding'])
    envelope=original.shared.save(original.ENVELOPE,{'schema':'utility-j1-max384-envelope-v2',
        'maximum_items':384,'maximum_prompt_tokens_per_item':16384,'maximum_complete_shards':len(price['shards']),
        'outcome_measurement_plan_sha256':value['binding']['outcome_measurement_plan_sha256'],
        'chain_plan_sha256':value['sha256'],**{k:price[k] for k in ('requested_allocation_GPU_hour_ceiling',
            'median_full_cap_all_attempts_projected_GPU_h','historical_item_rate_forecast_GPU_h',
            'pessimistic_minimum_positive_rate_projected_GPU_h','client_timeout_envelope_GPU_h')},
        'complete_generic_price':price,'scope':'Resource envelope only:384 possible blind items with maximum supported prompts. Placeholder item IDs are not assignments. Exact later prompts/items are priced before GPU submission; unchanged v1 pricing via qualified import adapter v2.'})
    print(json.dumps({'plan_sha256':value['sha256'],'envelope_sha256':envelope['sha256']}))


def main():
    original=install()
    if sys.argv[1:]==['freeze-plan']:freeze(original)
    else:original.main()


if __name__=='__main__':main()
