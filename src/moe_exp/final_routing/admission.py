"""Complete measured study reservation and immutable primary-design selection."""
from collections import Counter,defaultdict
from pathlib import Path
import math
from .common import MODELS, digest, load, publish, require, verified


def pack(rows,cost,load_seconds,*,capacity=1):
    """Deterministic batches and six-hour allocations with <=five planned hours."""
    require(0<=load_seconds<18000,'model load does not fit production allocation')
    ordered=sorted(rows,key=lambda r:r.get('uid',digest(r)))
    groups=[];current=[];seconds=load_seconds
    for start in range(0,len(ordered),capacity):
        batch=ordered[start:start+capacity]
        price=sum(cost(r) for r in batch)
        require(price>0 and price+load_seconds<=18000,'one covered batch cannot fit five planned hours')
        if current and seconds+price>18000:
            groups.append({'records':current,'planned_seconds':seconds});current=[];seconds=load_seconds
        current+=batch;seconds+=price
    if current:groups.append({'records':current,'planned_seconds':seconds})
    return groups


def generation_cost(price):
    require(price['status']=='complete' and price['remaining_allowance_preserved'],'generation execution profile is not qualified')
    return (price['seconds_per_token_upper']*price['maximum_generated_tokens']+
            price['full_context_prefill_seconds_upper']/price['capacity'])


def density_inventory(binding,mode):
    """Shared discovery-family 99th-percentile forecast at frozen allocation counts."""
    from .budget import context_forecast
    root=Path(binding['root']);observations=[];multiplicity={}
    available={}
    for model in MODELS:
        path=root/model/'DISCOVERY_CONTEXT_COUNTS.json'
        if path.exists():available[model]=verified(path)['records']
    require(all(m in available for m in MODELS[:-1]),'original discovery context inventory missing')
    for model in MODELS:
        assignments=root/model/('ASSIGNMENTS-'+mode+'.json')
        if assignments.exists():
            wanted=Counter((r['benchmark'],r['fold']) for r in verified(assignments)['records']
                           if r['prefix_status']=='enrolled' and r['nomination_status']=='nominated')
        else:
            require(model=='lfm25','missing original model assignments')
            wanted=Counter()
            for r in verified(root/'ALLOCATION.json')['records']:
                wanted[r['benchmark'],r['fold']]+=85+84*bool(r[mode+'_outer'])
        records=available.get(model)
        if records is None:
            # Before LFM discovery, reserve whole native traces using the
            # maximum observed density for each shared original family. Its
            # measured native pricing contexts additionally provide an upper
            # density multiplier. Recheck after actual LFM collection.
            combined={}
            for source_rows in available.values():
                for r in source_rows:
                    key=(r['benchmark'],r['family'],r['fold'])
                    if key not in combined or r['contexts']>combined[key]['contexts']:combined[key]=r
            records=list(combined.values())
        counts=Counter((r['benchmark'],r['fold']) for r in records)
        nominations=root/model/'COUNT_NOMINATIONS.json'
        horizons={r['fold']:r['horizon'] for r in verified(nominations)['records'] if r.get('horizon')} if nominations.exists() else {}
        if not horizons and model=='lfm25':
            horizons={f:verified(root/model/'QUALIFIED.json')['discovery_forecast_horizon'] for f in range(5)}
        for r in records:
            b,fold=r['benchmark'],r['fold']
            key=model+':'+b+':'+str(fold)
            multiplicity[key]=wanted[b,fold]/counts[b,fold]
            ends,starts=r.get('ends',[]),r.get('starts',[])
            h=horizons.get(fold)
            density=r['contexts']
            if h and ends:
                boundaries=[ends[min(len(ends)-1,len(ends)*i//4)] for i in (1,2,3)]
                density=max(1,max(sum(p>=x and p<x+h for p in starts) for x in boundaries))
            observations.append({**r,'model':key,'contexts':density})
    result=context_forecast(observations,multiplicity,draws=10000,seed=binding['config']['seed'])
    result['lfm_discovery_observed']='lfm25' in available
    return result


def forecasts(binding):
    root=Path(binding['root']);config=binding['config']
    prices={m:verified(root/m/'QUALIFIED.json') for m in MODELS}
    reader=verified(root/'READER_PRICE.json');cpu=verified(root/'CPU_PRICE.json')
    require(cpu['peak_rss_kib']<24*1024**2,'CPU price exceeds allocated host memory')
    require(cpu['reserved_fit_seconds']<=18000,'measured readout does not fit its five-hour planned CPU allocation')
    rp=reader['profiles']['covered']
    results={}
    for mode in ('full','fallback'):
        allocations=Counter();retained=0.;generation_details={}
        for model in MODELS:
            price=prices[model]
            require(price['profile_sha256']==binding['models'][model]['profile_sha256'],'stale model execution price')
            p=root/model/('ASSIGNMENTS-'+mode+'.json')
            if p.exists():
                rows=[r for r in verified(p)['records'] if r['prefix_status']=='enrolled' and r['nomination_status']=='nominated']
            else:
                rows=[{'uid':digest(['lfm-reserved-slot',i])} for i in range(129*85+(129 if mode=='full' else 32)*84)]
            cost=generation_cost(price)
            groups=pack(rows,lambda r:cost,price['load_seconds'],capacity=price['capacity'])
            allocations['generation']+=len(groups)
            generation_details[model]={'assignments':len(rows),'allocations':len(groups),'seconds_per_assignment':cost,
                'profile_sha256':price['profile_sha256']}
            # Capture storage is charged per horizon, not per sampled output token.
            max_h=max((r.get('horizon') or 512 for r in rows),default=512)
            retained+=len(rows)*(price['output_bytes_per_token']*price['maximum_generated_tokens']+
                                 price['capture_bytes_per_position_upper']*max_h+5*4096)
        lfm=prices['lfm25'];native_rows=verified(root/'lfm25/NATIVE_TASKS.json')['records']
        allocations['lfm_native']=len(pack(native_rows,lambda r:generation_cost(lfm),lfm['load_seconds'],capacity=lfm['capacity']))
        replay=lfm['replay_price']
        replay_seconds=replay['seconds_per_input_token_upper']*binding['models']['lfm25']['native_context_limit']
        allocations['lfm_replay']=len(pack([{'uid':digest(i)} for i in range(1547)],lambda r:replay_seconds,replay['load_seconds']))
        retained+=4647*lfm['output_bytes_per_token']*lfm['maximum_generated_tokens']
        retained+=1547*replay['bytes_per_completion_token_upper']*lfm['maximum_generated_tokens']
        contexts=density_inventory(binding,mode)
        count=contexts['upper_contexts']+100000
        batch_seconds=rp['batch_seconds'];batch_capacity=max(1,int((18000-rp['load_seconds'])/batch_seconds))
        allocations['reader']=math.ceil(math.ceil(count/64)/batch_capacity)
        bytes_per_context=max(r['output_bytes']/r['batch_size'] for r in reader['batch_receipts'])
        retained+=count*(bytes_per_context+5*4096)
        original=Path(binding['source_campaign']);legacy=[];legacy_bytes=0
        inherited=verified(root/'INHERITED_TASKS.json')['records']
        for model in MODELS[:-1]:
            price=load(original/model/'ANALYSIS_PRICE.json')
            missing=[r for r in inherited if r['model']==model and r['stage']=='coordination' and not Path(r['receipt_path']).exists()]
            per=max(1.,price['pair_seconds'])*1.25
            jobs=math.ceil(len(missing)/max(1,int(18000/per)))
            legacy.append({'model':model,'missing_pairs':[r['shard_index'] for r in missing],'pair_seconds':per,'allocations':jobs})
            allocations['legacy_cpu']+=jobs
            legacy_bytes+=len(missing)*price['pair_output_bytes']
        # Explicit CPU allocation contracts for native derivation, two readout
        # modes, endpoint/export panels and receipt-only dispatcher phases.
        allocations['analysis_cpu']=math.ceil(cpu['reserved_fit_seconds']/18000)+6
        base={k:n*6*(4 if k.endswith('_cpu') else 16)*2 for k,n in allocations.items()}
        base['controllers']=16*5/60*4
        temporary=max([p['memory']['temporary_bytes'] for p in prices.values()]+
                      [r['temporary_bytes'] for r in reader['batch_receipts']]+[1])
        storage={'work':{'fixed_bytes':2*1024**3+legacy_bytes,'per_worker_peak_bytes':4096*1024},
                 'fast':{'fixed_bytes':math.ceil(retained),'per_worker_peak_bytes':max(temporary,64*1024**2)}}
        results[mode]={'profiles_complete':True,'base_hours':base,'allocations':dict(allocations),
            'generation':generation_details,'lfm_native_attempts':4647,'lfm_discovery_traces':1547,
            'contexts':contexts,'reader_contexts':count,'reader_batch_capacity':batch_capacity,
            'legacy':legacy,'storage':storage,'recovery_attempts':1,'all_eight_models_reserved':True,
            'cpu_price_sha256':cpu['sha256'],'reader_price_sha256':reader['sha256']}
    return results


def admit(binding):
    from .resources import snapshot,availability,retained_usage
    from .inherited import accounting
    from .budget import admit as decide,concurrency
    root=Path(binding['root']);target=root/'ADMISSION.json'
    if target.exists():return verified(target)
    actual=accounting(binding);snap=snapshot();available=availability(binding['config'],snap,campaign_actual=actual['actual_hours'])
    all_forecasts=forecasts(binding)
    for value in all_forecasts.values():
        limit=concurrency(binding['config'],value['storage'],snap['free_bytes'])
        value['gpu_concurrency']=limit;value['cpu_concurrency']=min(8,limit) if limit else 0
        if limit<1:
            value['profiles_complete']=False;value['missing']=['retained outputs, temporary peak and separate 10-GiB reserves do not fit']
    decision=decide(binding['config'],all_forecasts,actual_hours=available['expended_hours'],committed_hours=0,
                    available_hours=available['available_hours'])
    publish(root/'ADMISSION_ACCOUNTING.json',{'snapshot':snap,'ancestors':actual,'available':available,
        'retained_usage':retained_usage(binding)})
    if decision['status']=='admitted':
        selected=all_forecasts[decision['mode']]
        decision.update(production_allocation_hours=decision['reserved_hours']-binding['config']['budget']['closeout_hours'],
            lfm_full_obligation_reserved=True,gpu_concurrency=selected['gpu_concurrency'],cpu_concurrency=selected['cpu_concurrency'])
    return publish(target,{**decision,'forecasts':all_forecasts,'baseline_consumed':25950,'original_ceiling_hours':42050,
        'margin':1.25,'primary_design_frozen':False,'physical_compaction':False})


def freeze_primary(binding):
    """Recheck actual LFM discovery before the first primary continuation."""
    from .resources import snapshot,availability,retained_usage
    from .inherited import accounting
    from copy import deepcopy
    from .budget import concurrency
    root=Path(binding['root']);admission=verified(root/'ADMISSION.json')
    if (root/'DESIGN.json').exists():return verified(root/'DESIGN.json')
    require(admission['status']=='admitted','production is not admitted')
    candidates=forecasts(binding);snap=snapshot();decisions=[]
    actual=accounting(binding);available=availability(binding['config'],snap,campaign_actual=actual['actual_hours'])
    initial=verified(root/'ADMISSION_ACCOUNTING.json')['retained_usage'];current=retained_usage(binding)
    reservations=[verified(p) for p in (root/'reservations').glob('*.json')]
    issued=sum(r['hours'] for r in reservations if r['phase']=='production')
    closeout_remaining=max(0,binding['config']['budget']['closeout_hours']-
        sum(r['hours'] for r in reservations if r['phase']=='finalization'))
    for mode in ('full','fallback'):
        if admission['mode']=='fallback' and mode=='full':continue
        value=candidates[mode]
        hours=sum(value['base_hours'].values())*1.25
        footprint=deepcopy(value['storage'])
        for fs in footprint:
            footprint[fs]['fixed_bytes']=max(0,footprint[fs]['fixed_bytes']-max(0,current[fs]-initial[fs]))
        space=concurrency(binding['config'],footprint,snap['free_bytes'])
        future=max(0,hours-issued)+closeout_remaining
        decisions.append({'mode':mode,'hours':hours,'storage_concurrency':space,
            'future_margin_inclusive_hours':future,'live_available_hours':available['available_hours'],
            'shortfall_hours':max(0,hours-admission['production_allocation_hours'],future-available['available_hours'])})
        if hours<=admission['production_allocation_hours'] and future<=available['available_hours'] and space:
            return publish(root/'DESIGN.json',{'status':'complete','mode':mode,'decisions':decisions,
                'lfm_discovery_sha256':verified(root/'lfm25/DISCOVERY_INPUTS.json')['sha256'],
                'allocation_sha256':verified(root/'ALLOCATION.json')['sha256'],'forecast':value,
                'frozen_before_primary':True,'subsequent_redesign_permitted':False,
                'fresh_accounting':available,'retained_usage':current,'remaining_storage':footprint,
                'gpu_concurrency':min(space,admission['gpu_concurrency']),
                'cpu_concurrency':min(8,space,admission['cpu_concurrency'])})
    return publish(root/'PRIMARY_NOT_ADMITTED.json',{'status':'resource_limited','decisions':decisions,
        'primary_released':False,'no_further_design_reduction':True})
