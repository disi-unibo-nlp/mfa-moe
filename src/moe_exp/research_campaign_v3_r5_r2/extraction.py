"""One verified capture pass; lossless layer-addressable labelled-token cache."""
from pathlib import Path
import shutil
import time
import resource
import numpy as np
from .contracts import load, publish, file_sha, check_files, require, digest
from .artifacts import atomic_file, npz
from .inputs import counts_for, read_source
from .native import archive
from .populations import class_tokens, validate_mapping


def array(path, value):
    with atomic_file(path) as stream: np.save(stream, value, allow_pickle=False)
    return {"path":str(path),"sha256":file_sha(path),"bytes":Path(path).stat().st_size}


def extract(campaign, model):
    root=Path(campaign['root']); directory=root/model/'compact'
    require(not (root/model/'CACHE_RETIRED.json').exists(),'retired cache cannot be regenerated')
    publish(directory/'OWNER.json',{'campaign_sha256':campaign['sha256'],'model':model})
    if (root/model/'EXTRACTION.json').exists():
        result=load(root/model/'EXTRACTION.json')
        check_files({a['path']:a['sha256'] for r in result['records'] for a in r['artifacts']})
        return result
    validate_mapping(campaign,model)
    sources=load(root/model/'SOURCES.json')['records']; _,tokens=counts_for(campaign,model,sources)
    contract=campaign['models'][model]; layers=contract['layers']; E=contract['experts']
    require(E<=65536,'lossless compact ID dtype requires at most 65536 experts')
    weighted=np.zeros((len(sources),9,len(layers),E),np.float32); turnover=[]; records=[]
    started=time.monotonic(); captured_bytes=0
    for si,source in enumerate(sources):
        aid=source['attempt_id']; target=directory/aid; receipt=target/'EXTRACTED.json'
        if receipt.exists():
            record=load(receipt); check_files({a['path']:a['sha256'] for a in record['artifacts']})
            with np.load(target/'weighted.npz',allow_pickle=False) as z:weighted[si]=z['weighted']
            turnover.extend(record['turnover']);records.append(record);continue
        raw=archive(source); ids=np.asarray(raw['ids']); weights=np.asarray(raw['weights'].float())
        require(ids.shape==weights.shape and ids.shape[0]==len(layers) and ids.shape[2]==contract['native_k'],'extraction native geometry differs')
        require((ids>=0).all() and (ids<E).all(),'invalid capture expert ID')
        reasoning=np.asarray(raw['reasoning_tokens'],np.int64)
        classes=class_tokens(campaign,model,source,reasoning)
        positions=np.sort(np.concatenate(classes)); owner=np.full(len(positions),-1,np.int8)
        for c,t in enumerate(classes):owner[np.searchsorted(positions,t)]=c
        require((owner>=0).all() and len(np.unique(positions))==len(positions),'compact ownership differs')
        # Retain original sentence IDs and break segments at annotation gaps.
        from moe_exp.schemas import TraceRecord
        from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
        layout=saved_layout(TraceRecord(**read_source(source['source_location'])))
        sentence=np.full(len(positions),-1,np.int32)
        for index,t in enumerate(layout['unit_tokens']):
            t=np.intersect1d(positions,t,assume_unique=True)
            sentence[np.searchsorted(positions,t)]=index
        require((sentence>=0).all(),'compact token missing original sentence ownership')
        breaks=np.r_[True,(np.diff(sentence)>1)|(np.diff(positions)>1)] if len(positions) else np.array([],bool)
        segment=np.cumsum(breaks,dtype=np.int32)-1
        artifacts=[array(target/'positions.npy',positions),array(target/'class.npy',owner),
                   array(target/'sentence.npy',sentence),array(target/'segment.npy',segment)]
        for li,layer in enumerate(layers):
            artifacts.append(array(target/f'layer-{layer}.npy',ids[li,positions].astype(np.uint16)))
        sets=classes+[reasoning,positions]
        local_turnover=[]
        for c,t in enumerate(sets):
            require(len(t)==tokens[si,c],'extraction nine token counts differ')
            for li in range(len(layers)):
                weighted[si,c,li]=np.bincount(ids[li,t].ravel(),weights=weights[li,t].ravel(),minlength=E)
        adjacent=reasoning[1:][np.diff(reasoning)==1]
        if len(adjacent):
            same=(ids[:,adjacent,:,None]==ids[:,adjacent-1,None,:]).any(-1).sum(-1)
            local_turnover=[{'benchmark':source['benchmark'],'family':source['family'],'layer':l,
                'routing_turnover':float((1-same[li]/contract['native_k']).mean())} for li,l in enumerate(layers)]
        artifacts.append(npz(target/'weighted.npz',weighted=weighted[si]))
        record=publish(receipt,{'attempt_id':aid,'source_sha256':digest(source),'artifacts':artifacts,
            'token_counts':list(map(int,tokens[si])),'turnover':local_turnover,'lossless_ids':True})
        turnover.extend(local_turnover);records.append(record);captured_bytes+=Path(source['tensor_path']).stat().st_size
        del raw,ids,weights
    durable=npz(root/model/'NATIVE_WEIGHTED.npz',weighted=weighted)
    result=publish(root/model/'EXTRACTION.json',{'status':'complete','records':records,'weighted':durable,'turnover':turnover,
        'seconds':time.monotonic()-started,'compressed_input_bytes':captured_bytes,
        'output_bytes':sum(a['bytes'] for r in records for a in r['artifacts']),
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'sources_sha256':load(root/model/'SOURCES.json')['sha256'],'ownership_sha256':load(root/model/'TOKEN_OWNERSHIP.json')['sha256'],
        'capture_passes_per_attempt':1,'temporary_cache_owner':campaign['sha256']})
    return result


def selected(campaign,model,source,layers):
    target=Path(campaign['root'])/model/'compact'/source['attempt_id']
    record=load(target/'EXTRACTED.json')
    require(record['source_sha256']==digest(source),'compact source identity differs')
    needed={str((target/f'layer-{l}.npy').resolve()) for l in layers}|{str((target/'class.npy').resolve())}
    declared={str(Path(a['path']).resolve()):a['sha256'] for a in record['artifacts']}
    require(needed <= set(declared), 'compact receipt omits requested arrays')
    check_files({path:declared[path] for path in needed})
    owner=np.load(target/'class.npy',mmap_mode='r',allow_pickle=False)
    ids={layer:np.load(target/f'layer-{layer}.npy',mmap_mode='r',allow_pickle=False) for layer in layers}
    return ids,[np.flatnonzero(owner==c) for c in range(7)]


def retire(campaign,model):
    from .observations import pair_tasks
    root=Path(campaign['root']); artifacts=[]
    for task in pair_tasks(campaign,model):
        path=root/model/'coordination'/f"pair-{task['shard_index']:03d}"/'RESULT.json'
        require(path.is_file(),'cache has unfinished dependent pair')
        check_files({a['path']:a['sha256'] for a in load(path).get('artifacts',[])})
        for p in path.parent.iterdir():
            if p.is_file() and p.suffix in ('.json','.npz','.parquet'):
                artifacts.append({'path':str(p),'sha256':file_sha(p)})
    require((root/model/'SUPPLEMENTARY.json').is_file(),'cache has unfinished supplementary output')
    check_files({a['path']:a['sha256'] for a in artifacts})
    extraction=load(root/model/'EXTRACTION.json');require(extraction['temporary_cache_owner']==campaign['sha256'],'cache not v3 owned')
    check_files({extraction['weighted']['path']:extraction['weighted']['sha256']})
    proof=publish(root/model/'CACHE_RETIREMENT_READY.json',{'dependencies':artifacts,'extraction_sha256':extraction['sha256']})
    directory=root/model/'compact'
    if directory.exists():shutil.rmtree(directory)
    return publish(root/model/'CACHE_RETIRED.json',{'status':'complete','proof_sha256':proof['sha256'],'original_captures_preserved':True})


def retire_failed(campaign,model,failure):
    """Replace unfinished dependent measurements with durable unavailable cells."""
    from .observations import pair_tasks
    root=Path(campaign['root']);directory=root/model/'compact'
    if not directory.exists():return
    require(failure.get('failure_scope')=='model','shared failure cannot retire dependent evidence')
    require(load(directory/'OWNER.json')['campaign_sha256']==campaign['sha256'],'failed cache not v3 owned')
    dependencies=[]
    for task in pair_tasks(campaign,model):
        path=root/model/'coordination'/f"pair-{task['shard_index']:03d}"/'RESULT.json'
        if not path.exists():publish(path,{'status':'upstream_unavailable','model':model,
            'pair':task,'failure':failure,'temporary_measurement_component_superseded':True})
        dependencies.append({'path':str(path),'sha256':file_sha(path)})
    check_files({a['path']:a['sha256'] for a in dependencies})
    proof=publish(root/model/'FAILED_CACHE_RETIREMENT.json',{'failure':failure,'dependencies':dependencies,
        'original_captures_preserved':True,'v3_owner_verified':True})
    shutil.rmtree(directory)
    publish(root/model/'CACHE_RETIRED.json',{'status':'upstream_unavailable','proof_sha256':proof['sha256']})
