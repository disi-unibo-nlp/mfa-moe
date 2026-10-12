"""Saved native associations, transfer and full-top-k coordination panels."""
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

from .artifacts import frame,npz
from .contracts import BENCHMARKS,CLASSES,load,publish,require
from .design import enriched_set,inclusion_statistics,family_weights
from .inputs import counts_for
from .native import archive
from .populations import class_tokens, validate_mapping
from .statistics import bootstrap_panel,log_odds_ratio


def pair_tasks(campaign,model):
    layers=campaign["models"][model]["layers"]
    return [{"layer_a":layers[i],"layer_b":layers[i+d],"depth_gap":d,"shard_index":n+1}
        for n,(i,d) in enumerate((i,d) for d in (1,2,4,8) for i in range(len(layers)-d))]


def supplementary(campaign,model):
    root=Path(campaign["root"])
    if campaign.get('resume_from'):
        from .resume import ancestry,reference_file
        for parent in ancestry(campaign):
            old=Path(parent['root']);receipt=old/model/'SUPPLEMENTARY.json'
            if not receipt.exists() or load(receipt).get('status')!='complete':continue
            if load(old/model/'SOURCES.json')['sha256']!=load(root/model/'SOURCES.json')['sha256']:continue
            require(parent['models'][model]['profile_sha256']==campaign['models'][model]['profile_sha256'],'supplementary native profile differs')
            require(parent['saved_inputs'][model]==campaign['saved_inputs'][model],'supplementary population differs')
            references=[]
            for name in ('transfer.parquet','source_conditioned_next.parquet','native_boundaries.parquet',
                         'native_weight_occupancy.parquet','native_turnover.parquet','ASSOCIATION_CONFIRMATION.json'):
                references.append(reference_file(root/model/name,old/model/name))
            original=load(receipt)
            return publish(root/model/'SUPPLEMENTARY.json',{**{k:v for k,v in original.items() if k!='sha256'},
                'source_receipt':str(receipt),'source_receipt_sha256':__import__('hashlib').sha256(receipt.read_bytes()).hexdigest(),
                'compatible_components_referenced':references,'population_binding':load(root/model/'SOURCES.json')['sha256'],
                'compatibility_basis':'unchanged original labels, nine-count validated ownership, weighting and estimand; compact extraction parity regression',
                'analysis_binding':campaign.get('analysis_binding')})
    sources=load(root/model/"SOURCES.json")["records"]
    source_index={r["attempt_id"]:i for i,r in enumerate(sources)}
    qf=load(root/"FOLDS.json")["question_families"]
    folds=load(root/"FOLDS.json")["folds"]
    contract=campaign["models"][model]
    sentences=pd.read_parquet(campaign["saved_inputs"][model]["sentences"])
    validate_mapping(campaign,model)
    counts,tokens=counts_for(campaign,model,sources)
    transfer=[]
    for benchmark in BENCHMARKS:
        # Exclude whole families with any occurrence in the held benchmark.
        excluded={s["family"] for s in sources if s["benchmark"]==benchmark}
        train=np.array([s["family"] not in excluded for s in sources])
        if not train.any():
            continue
        rate,support=inclusion_statistics(counts,tokens,sources,train)
        other=counts[:,:7].sum(1)[:,None]-counts[:,:7]
        nt=tokens[:,:7].sum(1)[:,None]-tokens[:,:7]
        baseline,_=inclusion_statistics(other,nt,sources,train)
        x=rate[:7]-baseline
        for c,name in enumerate(CLASSES):
            for li,l in enumerate(contract["layers"]):
                ids=enriched_set(x[c,li],.5)
                native=load(root/model/"NOMINATIONS.json")["records"]
                nominated=[r for r in native if r["class"]==name and r["layer"]==l and r["status"]=="nominated"]
                stability=[len(set(ids)&set(r["expert_sets"][1]))/len(set(ids)|set(r["expert_sets"][1]))
                    for r in nominated if set(ids)|set(r["expert_sets"][1])]
                held=[i for i,s in enumerate(sources) if s["benchmark"]==benchmark and tokens[i,c]>0]
                coverage=[float(counts[i,c,li,ids].sum()/tokens[i,c]/contract["native_k"]) for i in held]
                weights=family_weights([sources[i] for i in held]) if held else []
                transfer.append({"benchmark_excluded":benchmark,"class":name,"layer":l,"experts":ids,
                    "nomination_stability":float(np.mean(stability)) if stability else None,
                    "native_selected_slot_coverage":float(np.dot(weights,coverage)) if held else None,
                    "families":len({sources[i]["family"] for i in held}),"no_new_generation":True,
                    "status":"complete" if ids and held else "insufficient_support"})
    frame(root/model/"transfer.parquet",transfer)
    # Source-conditioned next-class rates use the original sentence histograms.
    # Only consecutive indices within one original annotation segment qualify.
    with np.load(campaign["saved_inputs"][model]["histogram"],allow_pickle=False) as z:
        hist=z["h"].astype(np.float32)*contract["native_k"]
        first,last=z["first16"].astype(np.float32),z["last16"].astype(np.float32)
        fi={int(v):i for i,v in enumerate(z["first16_rows"])}
        la={int(v):i for i,v in enumerate(z["last16_rows"])}
    require(len(hist)==len(sentences),"sentence histograms no longer align")
    next_rows,boundaries=[],[]
    previous={}
    grouped=defaultdict(list)
    for i,r in enumerate(sentences.itertuples()):
        prev=previous.get(r.attempt_id)
        if prev is not None:
            j,p=prev
            if r.sentence_index==p.sentence_index+1 and r.segment==p.segment:
                grouped[int(p.cls),int(r.cls)].append(j)
                if i in fi and j in la:
                    a,b=first[fi[i]],last[la[j]]
                    m=(a+b)/2
                    def kl(v):
                        return np.sum(np.where(v>0,v*np.log2(np.divide(v,m,out=np.ones_like(v),where=m>0)),0),axis=-1)
                    boundaries.append({"benchmark":r.dataset,"family":qf[r.question],"source":CLASSES[int(p.cls)],
                        "destination":CLASSES[int(r.cls)],"boundary_jsd":float(((kl(a)+kl(b))/2).mean())})
        previous[r.attempt_id]=(i,r)
    def transition_rate(indices):
        occurrence=defaultdict(list)
        for i in indices:
            r=sentences.iloc[i]
            occurrence[r.dataset,qf[r.question]].append(i)
        cells=[{"benchmark":b,"family":f} for b,f in occurrence]
        weights=family_weights(cells)
        value=sum(w*np.average(hist[occurrence[cell["benchmark"],cell["family"]]],axis=0,
            weights=sentences.iloc[occurrence[cell["benchmark"],cell["family"]]].n_tokens)
            for w,cell in zip(weights,cells)) if cells else None
        return value,occurrence
    for (source,target),indices in sorted(grouped.items()):
        value,occurrence=transition_rate(indices)
        others=[i for (s,t),rows in grouped.items() if s==source and t!=target for i in rows]
        baseline,_=transition_rate(others)
        for li,l in enumerate(contract["layers"]):
            for e in range(contract["experts"]):
                next_rows.append({"source_class":CLASSES[source],"next_class":CLASSES[target],"layer":l,"expert":e,
                    "source_inclusion_rate":float(value[li,e]),"family_support":len({f for b,f in occurrence}),
                    "other_next_class_inclusion":float(baseline[li,e]) if baseline is not None else None,
                    "source_conditioned_next_difference":float(value[li,e]-baseline[li,e]) if baseline is not None else None,
                    "benchmark_support":len({b for b,f in occurrence})})
    frame(root/model/"source_conditioned_next.parquet",next_rows)
    frame(root/model/"native_boundaries.parquet",boundaries)
    boundary_panel=bootstrap_panel(boundaries,["boundary_jsd"],panel="native-boundaries:"+model)
    # Original top-k weights are dispatched native weights, including native
    # scaling and shared paths. They are not assumed to sum to one.
    weighted=np.zeros(counts.shape,dtype=np.float32)
    by_attempt={aid:g for aid,g in sentences.groupby("attempt_id")}
    turnover=[]
    extraction=load(root/model/'EXTRACTION.json')
    with np.load(extraction['weighted']['path'],allow_pickle=False) as z:weighted=z['weighted']
    turnover=extraction['turnover']
    rate,support=inclusion_statistics(weighted,tokens,sources,np.ones(len(sources),bool))
    weighted_rows=[{"class":CLASSES[c],"layer":l,"expert":e,"native_weight_occupancy":float(rate[c,li,e]) if np.isfinite(rate[c,li,e]) else None,
        "family_support":int(support[c,li,e])} for c in range(7) for li,l in enumerate(contract["layers"]) for e in range(contract["experts"])]
    frame(root/model/"native_weight_occupancy.parquet",weighted_rows)
    frame(root/model/"native_turnover.parquet",turnover)
    confirmation=[]
    nominations=load(root/model/"NOMINATIONS.json")["records"]
    for fold in range(5):
        rows=[]
        for si,s in enumerate(sources):
            if folds[s["family"]]!=fold:continue
            row={"benchmark":s["benchmark"],"family":s["family"]}
            for n in nominations:
                if n["fold"]!=fold or n["status"]!="nominated":continue
                c=CLASSES.index(n["class"]);li=contract["layers"].index(n["layer"]);experts=n["expert_sets"][1]
                name=n["class"]+":"+str(n["depth_third"])
                nc=tokens[si,c];no=tokens[si,:7].sum()-nc
                row[name+":inclusion_difference"]=float(counts[si,c,li,experts].sum()/nc-
                    (counts[si,:7,li][:,experts].sum()-counts[si,c,li,experts].sum())/no) if nc and no else None
                row[name+":weight_difference"]=float(weighted[si,c,li,experts].sum()/nc-
                    (weighted[si,:7,li][:,experts].sum()-weighted[si,c,li,experts].sum())/no) if nc and no else None
            rows.append(row)
        columns=sorted({k for r in rows for k in r if k not in ("benchmark","family")})
        confirmation.append(bootstrap_panel(rows,columns,panel=f"association-confirmation:{model}:{fold}"))
    publish(root/model/"ASSOCIATION_CONFIRMATION.json",{"panels":confirmation,"conditional_on_nominations":True})
    return publish(root/model/"SUPPLEMENTARY.json",{"status":"complete","transfer_cells":len(transfer),
        "source_conditioned_cells":len(next_rows),"boundaries":boundary_panel,"annotation_gaps_bridged":False,
        "hidden_state_summary_source":campaign["saved_inputs"][model]["features"],
        "hidden_state_summaries_are_expert_output_norms":False,
        "native_weights_reused":True,"native_turnover_records":len(turnover),"held_out_confirmation":True})


def _joint_counts(a,b,E):
    n,k=a.shape
    x=sparse.csr_matrix((np.ones(n*k,dtype=np.int32),(np.repeat(np.arange(n),k),a.reshape(-1))),shape=(n,E))
    y=sparse.csr_matrix((np.ones(n*k,dtype=np.int32),(np.repeat(np.arange(n),k),b.reshape(-1))),shape=(n,E))
    return (x.T@y).toarray().astype(float)


def coordination(campaign,model,shard,metrics=None):
    import time
    coordination_start=time.monotonic()
    root=Path(campaign['root'])
    retained=root/model/'coordination'/f'pair-{shard:03d}'/'RESULT.json'
    if retained.exists():
        if metrics is not None:
            price=load(retained.parent/'COMPUTATION_PRICE.json')
            metrics.update({k:price[k] for k in ('bootstrap_seconds','coordination_seconds')})
        return load(retained)
    task=pair_tasks(campaign,model)[shard-1]
    contract=campaign["models"][model]
    E,k=contract["experts"],contract["native_k"]
    ia,ib=[contract["layers"].index(task[n]) for n in ("layer_a","layer_b")]
    sources=load(root/model/"SOURCES.json")["records"]
    folds=load(root/"FOLDS.json")["folds"]
    sentences=pd.read_parquet(campaign["saved_inputs"][model]["sentences"])
    validate_mapping(campaign,model)
    by_attempt={a:g for a,g in sentences.groupby("attempt_id")}
    _, original_tokens=counts_for(campaign,model,sources)
    lengths=defaultdict(lambda:np.zeros(14))
    for source,nine in zip(sources,original_tokens):
        n=nine[:7]
        lengths[source["benchmark"],source["family"]]+=np.concatenate((n,n.sum()-n))
    supports=np.zeros((5,6,14))
    for (b,f),n in lengths.items():
        supports[folds[f],BENCHMARKS.index(b)]+=n>0
    total=np.zeros((5,6,14,E,E))
    def selected(s):
        from .extraction import selected as compact_selected
        compact,classes=compact_selected(campaign,model,s,[task['layer_a'],task['layer_b']])
        return {ia:compact[task['layer_a']],ib:compact[task['layer_b']]},classes
    for s in sources:
        ids,classes=selected(s)
        matrices=[_joint_counts(ids[ia][t],ids[ib][t],E) if len(t) else np.zeros((E,E)) for t in classes]
        all_counts=sum(matrices)
        denom=lengths[s["benchmark"],s["family"]]
        slot=(folds[s["family"]],BENCHMARKS.index(s["benchmark"]))
        for c,m in enumerate(matrices+[all_counts-m for m in matrices]):
            if denom[c]>0:
                total[slot+(c,)]+=m/denom[c]
    def estimate(included):
        sup=supports[included].sum(0)
        val=total[included].sum(0)
        perbench=np.divide(val,sup[:,:,None,None],out=np.full_like(val,np.nan),where=sup[:,:,None,None]>0)
        good=(sup>0).sum(0)
        return np.divide(np.nansum(perbench,axis=0),good[:,None,None],out=np.full((14,E,E),np.nan),where=good[:,None,None]>0)
    def contrasts(value,c):
        j,j0=value[c],value[c+7]
        p,q=j.sum(1)/k,j.sum(0)/k
        p0,q0=j0.sum(1)/k,j0.sum(0)/k
        return log_odds_ratio(j,p,q)-log_odds_ratio(j0,p0,q0)
    nominations=[]
    for fold in range(5):
        discovery=[f for f in range(5) if f!=fold]
        value=estimate(discovery)
        for c,name in enumerate(CLASSES):
            contrast=contrasts(value,c)
            stability=np.zeros_like(contrast)
            valid=np.isfinite(contrast)
            for omit in discovery:
                alternative=contrasts(estimate([f for f in discovery if f!=omit]),c)
                stability+=np.isfinite(alternative)&(np.sign(alternative)==np.sign(contrast))
            candidates=[(int(a),int(b)) for a,b in zip(*np.where(valid))]
            candidates.sort(key=lambda ab:(-stability[ab],-abs(contrast[ab]),*ab))
            pairs=candidates[:16]
            nominations.append({"fold":fold,"class":name,"pairs":[{"expert_a":a,"expert_b":b,
                "discovery_log_odds_contrast":float(contrast[a,b]),"stability":float(stability[a,b]/4)} for a,b in pairs],
                "status":"nominated" if pairs else "insufficient_support"})
    directory=root/model/"coordination"/f"pair-{shard:03d}"
    if campaign.get("resume_from"):
        from .resume import source_file
        original=source_file(campaign,Path(model)/"coordination"/f"pair-{shard:03d}"/"NOMINATIONS.json")
        if original is not None:
            saved=load(original)
            require(saved["layer_a"]==task["layer_a"] and saved["layer_b"]==task["layer_b"],"inherited pair geometry differs")
            nominations=saved["records"]

    publish(directory/"NOMINATIONS.json",{"records":nominations,**task,"outcomes_used":False})
    descriptive=npz(directory/"descriptive_joints.npz",joint=estimate(list(range(5))),
        discovery_fold_joint_sums=total.astype(np.float32),family_support=supports)
    # Only nominated pairs are materialized for held-out inference. Every
    # nominated outcome remains represented even if its odds ratio is infinite.
    held=defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    for s in sources:
        fold=folds[s["family"]]
        ids,classes=selected(s)
        for n in nominations:
            if n["fold"]!=fold or not n["pairs"]:
                continue
            c=CLASSES.index(n["class"])
            family=(s["benchmark"],s["family"])
            for i,pair in enumerate(n["pairs"]):
                a,b=pair["expert_a"],pair["expert_b"]
                for part,toks in ((0,classes[c]),(1,np.concatenate([t for j,t in enumerate(classes) if j!=c]))):
                    denominator=lengths[family][c+part*7]
                    if denominator<=0:
                        continue
                    x=(ids[ia][toks]==a).any(1)
                    y=(ids[ib][toks]==b).any(1)
                    for field,v in (("j",np.sum(x&y)),("p",x.sum()),("q",y.sum())):
                        held[fold,c][family][f"{i}:{part}:{field}"]+=float(v/denominator)
    bootstrap_start=time.monotonic()
    panels=[]
    for n in nominations:
        fold,c=n["fold"],CLASSES.index(n["class"])
        if not n["pairs"]:
            panels.append({"fold":fold,"class":n["class"],"status":"insufficient_support"})
            continue
        rows=[{"benchmark":b,"family":f,**held[fold,c].get((b,f),{})} for b,f in lengths if folds[f]==fold]
        columns=[f"{i}:{p}:{key}" for i in range(len(n["pairs"])) for p in (0,1) for key in ("j","p","q")]
        def operation(values):
            v=values.reshape(len(values),len(n["pairs"]),2,3)
            j,p,q=v[:,:,:,0],v[:,:,:,1],v[:,:,:,2]
            cells=np.stack((j,p-j,q-j,1-p-q+j),axis=-1)
            good=(cells>0).all(-1)
            with np.errstate(invalid="ignore",divide="ignore"):
                odds=np.where(good,np.log(cells[:,:,:,0])+np.log(cells[:,:,:,3])-
                    np.log(cells[:,:,:,1])-np.log(cells[:,:,:,2]),np.nan)
            return np.stack((j[:,:,0]-j[:,:,1],p[:,:,0]-p[:,:,1],q[:,:,0]-q[:,:,1],odds[:,:,0]-odds[:,:,1]),axis=-1)
        key=f"coordination:{model}:{shard}:{fold}:{c}"
        linear_names=[f"pair{i}:{field}" for i in range(len(n["pairs"]))
            for field in ("joint_difference","marginal_a_difference","marginal_b_difference")]
        odds_names=[f"pair{i}:log_odds_contrast" for i in range(len(n["pairs"]))]
        linear_dependencies=[[i*6+j,i*6+3+j] for i in range(len(n["pairs"])) for j in range(3)]
        odds_dependencies=[list(range(i*6,(i+1)*6)) for i in range(len(n["pairs"]))]
        linear=bootstrap_panel(rows,columns,panel=key+":linear",alpha=.025,resample_key=key,
            transform=(linear_names,lambda v:operation(v)[:,:,:3].reshape(len(v),-1),linear_dependencies))
        odds=bootstrap_panel(rows,columns,panel=key+":log-odds",alpha=.025,resample_key=key,
            transform=(odds_names,lambda v:operation(v)[:,:,3],odds_dependencies))
        sufficient=publish(directory/f"sufficient-{fold}-{c}.json",{"records":rows,"columns":columns,
            "pairs":n["pairs"],"linear_dependencies":linear_dependencies,"log_odds_dependencies":odds_dependencies,
            "resample_key":key,"resamples":10000,"analysis_binding":campaign.get("analysis_binding"),"benchmark_population":list(BENCHMARKS)})
        panels.append({"fold":fold,"class":n["class"],"pairs":n["pairs"],"sufficient_statistics":sufficient["sha256"],
            "inference":{"results":{**linear["results"],**odds["results"]},"subpanels":{"linear":linear,"log_odds":odds},
                "simultaneous_coverage":.95,"alpha_split":[.025,.025],"paired_family_multiplicities":True,
                "undefined_draws_discarded":0,"resamples":10000}})
    pricepath=directory/'COMPUTATION_PRICE.json'
    price=load(pricepath) if pricepath.exists() else publish(pricepath,{
        'bootstrap_seconds':time.monotonic()-bootstrap_start,'coordination_seconds':time.monotonic()-coordination_start,
        'layers':[task['layer_a'],task['layer_b']],'resamples':10000})
    if metrics is not None:metrics.update({k:price[k] for k in ('bootstrap_seconds','coordination_seconds')})
    return publish(directory/"RESULT.json",{"status":"complete",**task,"panels":panels,
        "artifacts":[descriptive],
        "population":"full-A original labelled tokens; frozen nominations retained for inherited pairs",
        "sources_sha256":load(root/model/'SOURCES.json')['sha256'],"ownership_sha256":load(root/model/'TOKEN_OWNERSHIP.json')['sha256'],
        "full_top_k":True,"selection_stability_separate":True,"unestimable_nominations_retained":True})
