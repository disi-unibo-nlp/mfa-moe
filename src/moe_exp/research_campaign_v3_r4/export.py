"""Finite result package, paired uncertainty, missing cells and terminal reports."""
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from functools import lru_cache
import json
import os
import numpy as np

from .artifacts import atomic_file,frame
from .contracts import BENCHMARKS,CLASSES,CONFIGURATIONS,METHODS,MODELS,REMOVALS,Units,check_files,digest,load,publish
from .statistics import bootstrap_panel


def _resident(result,root,model,standardized=True,campaign=None):
    if not result:
        return None
    reports=result.get("memory_after")
    if standardized:
        from .masks import resolve
        mask=resolve(result.get('assignment',{}).get('mask'))
        layers=sorted(map(int,mask)) if mask else (campaign['models'][model]['layers'] if campaign else [])
        counts=[len(mask[str(l)]) for l in layers] if mask else None
        key=digest(counts) if mask else 'native'
        p=root/model/'qualification'/(key+'.json');reports=None
        if p.exists():
            q=load(p)
            if q['status']=='complete':
                valid=q.get('physical_layers')==layers and (not mask or q.get('physical_counts')==counts)
                if campaign:
                    native=campaign['models'][model]
                    valid &= q.get('profile_sha256')==native['profile_sha256'] and q.get('runtime_binding')==digest(campaign['runtime_files'])
                    valid &= q.get('full_context')==native['card']['max_model_len'] and q.get('fixed_kv_bytes')==native['kv_cache_memory_bytes']
                fixture=q.get('fixture',{})
                valid &= bool(q.get('fixture_cache_sha256')) and q.get('fixture_cache_sha256')==[v['cache_sha256'] for v in fixture.get('memory_after',[])]
                if valid:reports=fixture.get('memory_after')
    if not reports:
        return None
    values=[sum(v["bytes"] for v in r["memory"]["nvml_process_residence"]) for r in reports]
    return float(sum(values)) if len(values)==2 else None


def _planned(campaign,model):
    root=Path(campaign["root"])
    if (root/model/"TASKS.json").exists():
        return load(root/model/"TASKS.json")
    if not (root/"COHORTS.json").exists():
        return {"records":[],"cells":[{"model":model,"study":"all","status":"upstream_unavailable","uid":None}]}
    cohorts=load(root/"COHORTS.json")
    cells=[]
    for row in cohorts["pruning"]:
        for method,allocation,removal in [("native","shared",0.)]+[(m,a,r) for m in METHODS for a in ("equal","unequal") for r in REMOVALS]:
            cells.append({"study":"memory","model":model,**row,"method":method,"allocation":allocation,"removal":removal,"status":"upstream_unavailable","uid":None})
    prefix_path=root/model/"PREFIXES.json"
    from .assignments import roster
    prefixes=roster(campaign, model) if prefix_path.exists() else [{"model":model,**r,"prefix":None,"configuration_indices":[0,1]} for r in cohorts["causal"]]
    for row in prefixes:
        cells.append({"study":"causal",**row,"target":"native","arm":"native","threshold_index":0,"status":"upstream_unavailable","uid":None})
        for target in CLASSES:
            for depth in range(3):
                for sign in (-1,1):
                    for threshold in row["configuration_indices"]:
                        for arm in ("target","random"):
                            cells.append({"study":"causal",**row,"target":target,"depth_third":depth,"sign":sign,
                                "threshold_index":threshold,"arm":arm,"status":"upstream_unavailable","uid":None})
    if (root/model/"NATIVE_TASKS.json").exists():
        native=load(root/model/"NATIVE_TASKS.json")
        native_cells={(c["study"],c["benchmark"],c["family"],c.get("question")):c for c in native["cells"] if c["study"]=="memory"}
        cells=[native_cells.get((c["study"],c["benchmark"],c["family"],c.get("question")),c) if c.get("method")=="native" else c for c in cells]
        native_causal={(c["benchmark"],c["family"],c.get('question')):c for c in native["cells"] if c["study"]=="causal"}
        cells=[native_causal.get((c["benchmark"],c["family"],c.get('question')),c) if c.get("arm")=="native" else c for c in cells]
        return {"records":native["records"],"cells":cells}
    return {"records":[],"cells":cells}


def _response(row,result,context,annotations,cell, *, analysis_pulse_tokens=None, analysis_windows=None):
    from .responses import semantic_response
    values, sentences = semantic_response(row,result,context,annotations,cell,
        analysis_pulse_tokens=analysis_pulse_tokens, analysis_windows=analysis_windows)
    capture=((result or {}).get("capture") or {}).get("capture",{})
    layer=str((row or {}).get("action",{}).get("layer")) if (row or {}).get("action") else None
    if layer in capture:
        r=capture[layer]
        ids=np.asarray(r["executed_ids"])
        native=np.asarray(r["native_ids"])
        active=np.asarray(r["active"],bool)
        target_set=row["action"]["experts"]
        if len(ids):
            displacement=np.array([1-len(set(a)&set(b))/len(a) for a,b in zip(ids,native)])
            slots=np.isin(ids,target_set).sum(1)/ids.shape[1]
            baseline=np.isin(native,target_set).sum(1)/ids.shape[1]
            weights=np.asarray(r["executed_weights"])
            values.update(routing_displacement=float(displacement.mean()),
                pulse_displacement=float(displacement[active].mean()) if active.any() else None,
                recovery_displacement=float(displacement[~active].mean()) if (~active).any() else None,
                target_slot_excess=float((slots-baseline)[active].mean()) if active.any() else None,
                target_routing_weight=float((weights*np.isin(ids,target_set)).sum(1).mean()),
                captured_positions=len(ids),native_routing_k=ids.shape[1])
    return values,sentences


def contrast_bounds(treatment_lower,treatment_upper,control_lower,control_upper):
    if any(v is None for v in (treatment_lower,treatment_upper,control_lower,control_upper)):return None,None
    return treatment_lower-control_upper,treatment_upper-control_lower


def _panels(rows):
    universes = {study: sorted({r["family"] for r in rows if r["study"] == study and "family" in r}) for study in ("memory", "causal")}
    grouped=defaultdict(list)
    for r in rows:
        if r["study"]=="memory":
            panel=("memory",r["model"])
            name=f"{r['method']}:{r['allocation']}:{float(r['removal']):g}"
        elif r["study"]=="causal" and r.get("target")!="native" and "threshold_index" in r:
            panel=("causal",r["model"],r.get("depth_third"),r.get("sign"),r["threshold_index"])
            name=f"{r['target']}:{r['arm']}"
        else:
            continue
        grouped[panel].append((name,r))
    panels=[]
    for key,items in grouped.items():
        cells=defaultdict(dict);columns=set()
        for name,r in items:
            familykey=(r["benchmark"],r["family"],r.get("question",""))
            if key[0]=="memory":
                measurements={m:r.get(m) for m in ("strict_correct","unknown_lower","unknown_upper","resident_bytes","cumulative_tokens","new_tokens")}
            else:
                measurements={m:v for m,v in r.items() if m.startswith(("coverage:","first:","first_lower:","first_upper:","occupancy:","occupancy_lower:","occupancy_upper:","recovery:","recovery_lower:","recovery_upper:","native_difference:","native_difference_lower:","native_difference_upper:")) or m in (
                    "strict_correct","unknown_lower","unknown_upper","new_tokens","routing_displacement","pulse_displacement",
                    "recovery_displacement","target_slot_excess","class_switches","reentry_count","longest_observed_dwell_tokens",
                    "target_at_window_end","first_labelled_postpulse_matches_source","observed_target_exits","first_termination","first_unknown")}
            for metric,value in measurements.items():
                col=name+"|"+metric
                cells[familykey][col]=value
                columns.add(col)
        for values in cells.values():
            if key[0]=='memory':
                arms=sorted({c.split('|')[0] for c in values})
                pairs=[(a,'native:shared:0',a+'-minus-native') for a in arms if a!='native:shared:0']
                pairs += [(a,b,a+'-minus-'+b) for i,a in enumerate(arms) for b in arms[i+1:]
                          if a.split(':')[1:]==b.split(':')[1:] and not a.startswith('native:')]
            else:
                arms=sorted({c.split('|')[0] for c in values})
                pairs=[(a,a.replace(':target',':random'),a.replace(':target',':target-minus-random')) for a in arms if a.endswith(':target')]
            for a,b,label in pairs:
                for metric in [c.split('|',1)[1] for c in list(values) if c.startswith(a+'|')]:
                    if metric.startswith(('unknown_lower','unknown_upper','first_lower:','first_upper:','occupancy_lower:','occupancy_upper:','recovery_lower:','recovery_upper:')):continue
                    x,y=values.get(a+'|'+metric),values.get(b+'|'+metric)
                    values[label+'|'+metric]=x-y if x is not None and y is not None else None
                    columns.add(label+'|'+metric)
                for lower,upper in [('unknown_lower','unknown_upper')]+[(m,m.replace('_lower:','_upper:')) for m in [c.split('|',1)[1] for c in list(values) if c.startswith(a+'|') and '_lower:' in c]]:
                    lo,hi=contrast_bounds(values.get(a+'|'+lower),values.get(a+'|'+upper),values.get(b+'|'+lower),values.get(b+'|'+upper))
                    values[label+'|'+lower]=lo;values[label+'|'+upper]=hi
                    columns.update((label+'|'+lower,label+'|'+upper))
        panel_rows=[{"benchmark":b,"family":f,"question":q,**values} for (b,f,q),values in cells.items()]
        if columns:
            shared_key='memory:paired-seven-model' if key[0]=='memory' else 'causal:paired-seven-model:'+':'.join(map(str,key[2:]))
            panels.append({"identity":list(key),**bootstrap_panel(panel_rows,sorted(columns),
                panel="|".join(map(str,key)),resample_key=shared_key,
                benchmarks=BENCHMARKS, family_universe=universes[key[0]])})
    return panels


def _cross_model_panels(rows,campaign):
    root=Path(campaign["root"])
    native=defaultdict(dict)
    for model in MODELS:
        path=root/model/"NATIVE_ESTIMANDS.json"
        if path.exists():
            for r in load(path)["records"]:
                for metric in ("strict","unknown_lower","unknown_upper","tokens"):
                    native[r["benchmark"],r["family"]].setdefault(model+"|"+metric,[]).append(r[metric])
    native_rows=[{"benchmark":b,"family":f,**{k:float(np.mean([v for v in vs if v is not None])) if any(v is not None for v in vs) else None for k,vs in values.items()}}
        for (b,f),values in native.items()]
    panels=[]
    if native_rows:
        columns=[m+"|"+metric for m in MODELS for metric in ("strict","unknown_lower","unknown_upper","tokens")]
        panels.append({"identity":["native","paired-seven-model"],**bootstrap_panel(native_rows,columns,panel="native:paired-seven-model", benchmarks=BENCHMARKS,
            family_universe=sorted({r['family'] for r in native_rows}))})
    # Primary central treatment-minus-control comparisons share whole-family
    # multiplicities across all seven models and both arms. Sensitivity panels
    # retain the complete per-model configured outcomes separately.
    grouped=defaultdict(lambda:defaultdict(dict))
    for r in rows:
        if r["study"]!="causal" or r.get("threshold_index")!=0 or r.get("arm") not in ("target","random"):continue
        key=r["target"],r["depth_third"],r["sign"]
        cell=grouped[key][r["benchmark"],r["family"],r.get('question')]
        for metric in ("first:"+r["target"],"strict_correct","new_tokens"):
            cell[r["model"]+"|"+r["arm"]+"|"+metric]=r.get(metric)
    for key,families in grouped.items():
        panel_rows=[];columns=[]
        for (b,f,q),values in families.items():
            row={"benchmark":b,"family":f,"question":q}
            for m in MODELS:
                for metric in ("first:"+key[0],"strict_correct","new_tokens"):
                    a,c=values.get(m+"|target|"+metric),values.get(m+"|random|"+metric)
                    name=m+"|target-minus-random|"+metric
                    row[name]=a-c if a is not None and c is not None else None
                    if name not in columns:columns.append(name)
            panel_rows.append(row)
        panels.append({"identity":["causal-diagnostic","paired-seven-model","MMM",*key],**bootstrap_panel(panel_rows,columns,panel="causal:paired-seven-model:"+str(key),
            benchmarks=BENCHMARKS, family_universe=sorted({r['family'] for r in rows if r['study']=='causal' and 'family' in r}))})
    return panels


def _figures(directory,rows,campaign):
    from .figures import render
    return render(directory,rows,campaign)


def _observational_figures(directory,campaign):
    import pandas as pd
    import matplotlib.pyplot as plt
    from .design import family_weights
    root=Path(campaign["root"]);figures=[]
    for model in MODELS:
        modelroot=root/model
        panels=[]
        path=modelroot/"association_atlas.parquet"
        if path.exists():
            data=pd.read_parquet(path)
            means=data.groupby(["class","layer"]).inclusion_difference.apply(lambda s:float(s.clip(lower=0).sum()/5))
            panels.append(("associations","Positive inclusion excess; discovery folds",means))
        path=modelroot/"transfer.parquet"
        if path.exists():
            data=pd.read_parquet(path)
            means=data.groupby(["class","layer"]).native_selected_slot_coverage.mean()
            panels.append(("transfer","Benchmark-excluded native routing coverage",means))
        path=modelroot/"native_weight_occupancy.parquet"
        if path.exists():
            data=pd.read_parquet(path)
            means=data.groupby(["class","layer"]).native_weight_occupancy.sum(min_count=1)
            panels.append(("native-weights","Native routed weight mass; all routed layers",means))
        for name,title,means in panels:
            layers=campaign["models"][model]["layers"]
            array=np.array([[means.get((c,l),np.nan) for l in layers] for c in CLASSES])
            fig,axis=plt.subplots(figsize=(8,3.4));cmap=plt.get_cmap("viridis").copy();cmap.set_bad("lightgrey")
            im=axis.imshow(array,aspect="auto",cmap=cmap)
            axis.set_yticks(range(7),CLASSES);axis.set_xticks(range(len(layers))[::4],layers[::4])
            axis.set(title=model+" / "+title,xlabel="Native routed layer ID")
            fig.colorbar(im,ax=axis);fig.tight_layout()
            p=directory/"figures"/(model+"-"+name+".svg")
            with atomic_file(p) as handle:fig.savefig(handle,format="svg",metadata={"Date":None})
            plt.close(fig);figures.append(str(p))
        for name in ("coordination","activation","native-dynamics"):
            fig,axis=plt.subplots(figsize=(7,3.3));measured=False
            if name=="coordination":
                points=[]
                for resultpath in (modelroot/"coordination").glob("pair-*/RESULT.json"):
                    result=load(resultpath)
                    for panel in result.get("panels",[]):
                        for field,value in panel.get("inference",{}).get("results",{}).items():
                            if field.endswith("log_odds_contrast") and value["estimate"] is not None:
                                points.append((result["depth_gap"],value["estimate"]))
                if points:
                    axis.scatter(*zip(*points),s=3,alpha=.3);measured=True
                axis.set(xlabel="Routed-layer gap",ylabel="Held-out class log-odds-ratio contrast")
            elif name=="activation" and (modelroot/"SOURCES.json").exists():
                source=load(modelroot/"SOURCES.json")["records"];lookup={s["attempt_id"]:s for s in source}
                data=pd.read_parquet(campaign["saved_inputs"][model]["features"])
                columns=[c for c in data.columns if c.startswith("saved_hidden|") and c.split("|")[-1].startswith("L")]
                for metric in sorted({c.split("|")[2] for c in columns}):
                    selected=sorted([c for c in columns if c.split("|")[2]==metric])
                    x=[];y=[]
                    for c in selected:
                        valid=data[c].dropna();valid=valid[[a in lookup for a in valid.index]]
                        if len(valid):
                            x.append(int(c.split("|")[-1][1:]));y.append(float(np.dot(family_weights([lookup[a] for a in valid.index]),valid)))
                    if x:axis.plot(x,y,label=metric);measured=True
                axis.set(xlabel="Saved physical layer ID",ylabel="Hidden-state summary (not expert-output norm)")
                if measured:axis.legend(fontsize=6)
            elif name=="native-dynamics" and (modelroot/"DYNAMICS.json").exists():
                data=load(modelroot/"DYNAMICS.json");source={s["attempt_id"]:s for s in load(modelroot/"SOURCES.json")["records"]}
                transitions=[{**source[t["attempt_id"]],**t} for t in data["transitions"]]
                matrix=np.full((7,7),np.nan)
                for i,c in enumerate(CLASSES):
                    selected=[t for t in transitions if t["source"]==c]
                    if selected:
                        w=family_weights(selected)
                        for j,d in enumerate(CLASSES):matrix[i,j]=np.dot(w,[t["destination"]==d for t in selected])
                im=axis.imshow(matrix,vmin=0,vmax=1,cmap="viridis");fig.colorbar(im,ax=axis)
                axis.set_xticks(range(7),CLASSES,rotation=45,ha="right");axis.set_yticks(range(7),CLASSES);measured=True
            if not measured:axis.text(.5,.5,"Unmeasured / insufficient support",transform=axis.transAxes,ha="center")
            axis.set_title(model+" / "+name);fig.tight_layout()
            p=directory/"figures"/(model+"-"+name+".svg")
            with atomic_file(p) as handle:fig.savefig(handle,format="svg",metadata={"Date":None})
            plt.close(fig);figures.append(str(p))
    return figures


def _assemble(campaign,phase=None):
    root=Path(campaign["root"])
    rows,sentence_rows,ledgers=[],[],[]
    scientific_complete=True
    for model in MODELS:
        table=_planned(campaign,model)
        executions=Units(root/model/"executions");scores=Units(root/model/"scores");annotations=Units(root/model/"annotations")
        # Export is read-only. Bound caches avoid repeated Lustre reads for
        # native aliases and overlapping windows without retaining all captures.
        executions.state=lru_cache(maxsize=128)(executions.state)
        scores.state=lru_cache(maxsize=1024)(scores.state)
        annotations.state=lru_cache(maxsize=8192)(annotations.state)
        task_lookup={r["uid"]:r for r in table["records"]}
        contexts=[]
        for path in sorted((root/model).glob('CONTEXTS*.json')):
            contexts.extend(load(path)["records"])
        context_lookup={r["uid"]:r for r in contexts}
        native_lookup={(r["benchmark"],r["family"],r.get('question')):r["uid"] for r in table["cells"] if r.get("arm")=="native" and r.get("uid")}
        for cell in table["cells"]:
            cell_phase='memory' if cell['study']=='memory' else 'central' if cell.get('threshold_index',0)==0 else 'sensitivity'
            if phase and cell_phase != phase:continue
            uid=cell.get("uid")
            result=executions.state(uid) if uid else None
            status_=cell["status"]
            if uid:
                status_=(result or {}).get("status","upstream_unavailable")
                if status_=="started":
                    status_="execution_failed"
            if status_=="pending":
                status_="upstream_unavailable"
            row={**cell,"status":status_,"strict_correct":None,"unknown_lower":0.,"unknown_upper":1.,
                "resident_bytes":None,"memory_unavailable_reason":"missing_matching_physical_fixture"}
            if cell['study']=='causal' and not uid:
                unknown,_=_response(None,None,None,annotations,cell)
                row.update(unknown)
            if result and result.get("status")!="complete":row["retained_partial_tokens"]=result.get("generated_tokens")
            if result and result.get("status")=="complete":
                row["status"]="reused" if result.get("reused") else "complete"
                row.update(resident_bytes=_resident(result,root,model,campaign=campaign),execution_resident_bytes=_resident(result,root,model,False),
                    memory_measurement_conditions="qualified fixed-KV full-context fixture",new_tokens=result.get("generated_tokens"))
            if uid:
                scored=scores.state(uid)
                row["execution_status"]=row["status"]
                row["scoring_status"]=(scored or {}).get("status","upstream_unavailable")
                if scored:
                    row.update({k:scored.get(k) for k in ("strict_correct","unknown_lower","unknown_upper","cumulative_tokens","new_tokens","termination")})
                else:
                    row.update(strict_correct=None,unknown_lower=0.,unknown_upper=1.)
                if row["status"] in ("complete","reused") and row["scoring_status"]!="complete":
                    row["status"]="upstream_unavailable"
                if cell["study"]=="causal":
                    values,sentences=_response(task_lookup.get(uid),result,context_lookup.get(uid),annotations,cell)
                    row.update(values)
                    context=context_lookup.get(uid)
                    missing_labels=any((annotations.state(s["annotation_uid"]) or {}).get("status") not in ("complete","reused","invalid","truncated")
                        for s in (context or {}).get("sentences",[]))
                    row["annotation_status"]="upstream_unavailable" if missing_labels or (context is None and (result or {}).get("generated_tokens")) else "complete"
                    if row["status"] in ("complete","reused") and row["annotation_status"]!="complete":row["status"]="upstream_unavailable"
                    if cell.get("arm") in ("target","random"):
                        native_uid=native_lookup.get((cell["benchmark"],cell["family"],cell.get('question')))
                        if native_uid:
                            native_values,_=_response(task_lookup.get(native_uid),executions.state(native_uid),context_lookup.get(native_uid),annotations,cell,
                                analysis_pulse_tokens=values["analysis_pulse_tokens"], analysis_windows=values["analysis_windows"])
                            for metric,value in values.items():
                                if metric.startswith(("first:","occupancy:","recovery:")) or metric in ("restricted_target_dwell_tokens","target_transition_count","target_reentry_count","first_postpulse_matches_source","observed_target_exits"):
                                    reference=native_values.get(metric)
                                    row["native_difference:"+metric]=value-reference if value is not None and reference is not None else None
                                    if metric.startswith(('first:','occupancy:','recovery:')):
                                        prefix,suffix=metric.split(':',1)
                                        lo,hi=contrast_bounds(values.get(prefix+'_lower:'+suffix),values.get(prefix+'_upper:'+suffix),
                                            native_values.get(prefix+'_lower:'+suffix),native_values.get(prefix+'_upper:'+suffix))
                                        row['native_difference_lower:'+metric]=lo;row['native_difference_upper:'+metric]=hi
                            native_score=scores.state(native_uid) or {'unknown_lower':0.,'unknown_upper':1.}
                            for metric in ('strict_correct','cumulative_tokens','new_tokens'):
                                x,y=row.get(metric),native_score.get(metric)
                                row['native_difference:'+metric]=x-y if x is not None and y is not None else None
                            lo,hi=contrast_bounds(row.get('unknown_lower'),row.get('unknown_upper'),native_score.get('unknown_lower'),native_score.get('unknown_upper'))
                            row['native_difference_lower:accuracy']=lo;row['native_difference_upper:accuracy']=hi
                    sentence_rows.extend({"uid":uid,"model":model,"benchmark":cell["benchmark"],"family":cell["family"],**s} for s in sentences)
            scientific_complete &= row["status"] in ("complete","reused")
            if uid and row["status"] in ("complete","reused"):
                scientific_complete &= (scores.state(uid) or {}).get("status")=="complete"
                if cell["study"]=="causal":
                    scientific_complete &= uid in context_lookup or not result.get("generated_tokens")
                    scientific_complete &= all((annotations.state(s["annotation_uid"]) or {}).get("status") in
                        ("complete","reused","invalid","truncated") for s in context_lookup.get(uid,{}).get("sentences",[]))
            rows.append(row)
        if phase:continue
        from .observations import pair_tasks
        for pair in pair_tasks(campaign,model):
            p=root/model/"coordination"/f"pair-{pair['shard_index']:03d}"/"RESULT.json"
            ok=p.exists() and load(p)["status"]=="complete"
            scientific_complete &= ok
            ledgers.append({"model":model,"study":"coordination",**pair,"status":"complete" if ok else "upstream_unavailable","source":str(p)})
        for name in ("DERIVED","SUPPLEMENTARY","QUALIFICATION","PRICES"):
            p=root/model/(name+".json")
            component_status=load(p).get("status","complete") if p.exists() else "upstream_unavailable"
            scientific_complete &= component_status in ("complete","reused")
            ledgers.append({"model":model,"study":name.lower(),"status":component_status,"source":str(p)})
    return rows,sentence_rows,ledgers,bool(scientific_complete)


def deliver(campaign,directory,report,*,final=False,phase=None):
    root=Path(campaign['root'])
    manifest=directory/'EXPORT_FILES.json'
    if manifest.exists():check_files(load(manifest)['files'])
    # Durable results precede paper delivery; a publication failure cannot
    # erase successful reporting or scientific completion.
    if final:
        publish(root/'SCIENCE_COMPLETE.json',{'complete':report['SCIENCE_COMPLETE'],'report_sha256':report['sha256']})
        publish(root/'REPORT_COMPLETE.json',{'complete':True,'report_sha256':report['sha256']})
    from .decisions import paper_exhibits
    binding=paper_exhibits(campaign,directory,report)
    publication=None
    if binding and campaign.get('publish_results'):
        from .publication import deliver as publish_paper
        publication=publish_paper(campaign,report,binding)
    receipt=publish(directory/'PAPER_DELIVERY.json',{'complete':True,'report_sha256':report['sha256'],
        'binding_sha256':binding['sha256'] if binding else None,
        'publication_sha256':publication['sha256'] if publication else None,
        'phase':phase or 'final' if final else phase or 'snapshot'})
    if final:publish(root/'PAPER_DELIVERY.json',{'complete':True,'delivery_sha256':receipt['sha256']})
    if phase:publish(root/('PHASE_COMPLETE-'+phase+'.json'),{'phase':phase,'report_sha256':report['sha256'],
        'scientific_cells_complete':report['SCIENCE_COMPLETE'],'paper_delivery_sha256':receipt['sha256']})
    return {'status':'complete','REPORT_COMPLETE':final,'PAPER_DELIVERED':True,'SCIENCE_COMPLETE':report['SCIENCE_COMPLETE']}


def closeout(campaign,final=True,phase=None):
    root=Path(campaign["root"])
    directory=root/("deliveries/"+phase if phase else "output" if final else "snapshots/2026-10-15")
    if (directory/"REPORT.json").exists():
        report=load(directory/"REPORT.json")
        return deliver(campaign,directory,report,final=final,phase=phase)
    directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    if not final and not phase and (root/"REPORT_COMPLETE.json").exists():
        # If science closed before the milestone, reuse its committed package
        # through same-filesystem hardlinks. No repeat analysis or data copy.
        previous=load(root/"output/REPORT.json")
        for source in (root/"output").rglob("*"):
            if source.is_file() and source.name not in ("REPORT.json","PAPER_DELIVERY.json"):
                target=directory/source.relative_to(root/"output")
                target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
                if not target.exists():os.link(source,target)
        from .controller import status
        report=publish(directory/"REPORT.json",{**previous,"snapshot":True,
            "reused_final_report_sha256":previous["sha256"],"resource_usage":status(campaign),
            "observed_utc":datetime.now(timezone.utc).isoformat()})
        return deliver(campaign,directory,report)
    statepath=directory/"EXPORT_STATE.json"
    if statepath.exists():
        state=load(statepath)
    else:
        rows,sentence_rows,ledgers,scientific_complete=_assemble(campaign,phase=phase)
        from .controller import status
        state=publish(statepath,{"rows":rows,"sentence_rows":sentence_rows,"ledgers":ledgers,
            "scientific_complete":scientific_complete,"resource_observation":status(campaign),"observed_utc":datetime.now(timezone.utc).isoformat()})
        # Canonical JSON sorts keys. Use its readback for the first export too,
        # so dataframe column order is identical after an interrupted commit.
        state=load(statepath)
    rows,sentence_rows,ledgers,scientific_complete=state["rows"],state["sentence_rows"],state["ledgers"],state["scientific_complete"]
    # Dataframe cells use JSON for nested immutable identity records.
    export_rows=[{k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()} for r in rows]
    frame(directory/"cells.parquet",export_rows);frame(directory/"cells.csv",export_rows)
    frame(directory/"missingness.csv",[r for r in export_rows if r["status"] not in ("complete","reused")])
    frame(directory/"measurement_ledger.csv",ledgers)
    frame(directory/"sentence_dynamics.parquet",sentence_rows)
    from .primary import central_panels
    panels=_panels(rows)+_cross_model_panels(rows,campaign)
    if phase != 'memory':
        from .primary import supporting_panels
        panels+=central_panels(rows,campaign)+supporting_panels(rows,campaign)
    publish(directory/"UNCERTAINTY.json",{"panels":panels,"resamples":10000,"run_to_run_sd_reported":False})
    figures=_figures(directory,rows,campaign)
    if not phase:figures+=_observational_figures(directory,campaign)
    table_rows=[]
    for p in panels:
        for cell,value in p["results"].items():
            table_rows.append({"panel":" / ".join(map(str,p["identity"])),"cell":cell,**value})
    frame(directory/"paper_tables.csv",[{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in table_rows])
    with atomic_file(directory/"paper_tables.tex") as handle:
        text="\\begin{longtable}{lll}\nPanel & Estimate & Simultaneous interval \\\\\n\\hline\n"
        for r in table_rows:
            label=(r["panel"]+" / "+r["cell"]).replace("_","\\_").replace("|"," / ")
            estimate="unestimable" if r["estimate"] is None else f"{r['estimate']:.4g}"
            interval="unestimable" if not r["interval"] else "["+", ".join(f"{v:.4g}" for v in r["interval"])+"]"
            text+=f"{label} & {estimate} & {interval} \\\\\n"
        handle.write((text+"\\end{longtable}\n").encode())
    from .controller import status
    observed=state["resource_observation"]
    check_files(campaign["preserved_scientific_files"])
    check_files(campaign["preserved_operational_files"])
    decision_path=root/'PRE_PRODUCTION_DECISION.json'
    if not decision_path.exists():
        admission_receipt=load(root/'ADMISSION.json') if (root/'ADMISSION.json').exists() else None
        from .decisions import preproduction
        preproduction(campaign,admission_receipt)
    report_value={"campaign_id":campaign["campaign_id"],"REPORT_COMPLETE":True,"phase":phase or 'final' if final else phase or 'snapshot',
        "observed_utc":state["observed_utc"],
        "SCIENCE_COMPLETE":bool(scientific_complete),"snapshot":not final,"cells":len(rows),
        "status_counts":{s:sum(r["status"]==s for r in rows) for s in sorted({r["status"] for r in rows})},
        "figures":figures,"native_profiles":campaign["models"],"annotator":campaign["annotator"],
        "resource_usage":observed,"terminal":observed["terminal"],"uncertainty_panels":len(panels),
        "historical_sources_preserved":True,"strict_and_adjudicated_accuracy_mixed":False,
        "resume_from":campaign.get("resume_from"),"analysis_binding":campaign.get("analysis_binding"),
        "instrumentation_binding":campaign.get("instrumentation_binding"),
        "repair_scope":"native router instrumentation, full-A ownership, exact-attempt metadata and declared uncertainty",
        "coordination_alpha_split":[.025,.025],"undefined_resamples_discarded":0,
        "original_failed_measurements_preserved":True,"ownership_transfer":str(root/"OWNERSHIP_TRANSFER.json"),
        "reuse_catalogue":str(root/"REUSED_CATALOGUE.json"),"reuse_ledger":str(root/"REUSE.json"),
        "selection_conditioning":"uncertainty conditions on immutable masks and nominations; stability separate"}
    with atomic_file(directory/"report.md") as handle:
        text=f"# Research campaign {campaign['campaign_id']}\n\nReport complete: yes. All scientific cells complete: {'yes' if scientific_complete else 'no'}.\n\n"
        text+=f"This {phase+' phase report' if phase else 'October 15 snapshot' if not final else 'final report'} retains {len(rows):,} configured cells across all seven models. Missing or failed cells remain explicit and do not represent null effects.\n\n"
        text+="Accuracy uses the original strict benchmark contract. Unresolved answers retain lower and upper bounds. The reported intervals use 10,000 paired whole-family bootstrap resamples with equal benchmark and family weights. They condition on frozen selections; selection stability is separate. One replicate was run, so no run-to-run standard deviation is reported.\n\n"
        text+="Coordination declares separate linear and log-odds subpanels with error allowances 0.025 each and the same 10,000 whole-family resamples. Undefined draws are retained; an unestimable log-odds outcome does not suppress a supported linear interval. Full-A token ownership is recovered from original labels/offsets without changing frozen causal enrollment. Native accuracy joins each exact original attempt.\n\n"
        text+="Immediate response and pulse/recovery dynamics are exported together. Endpoint class, the first post-pulse sentence (unknown labels retained), clipped dwell, re-entry and observed target exits are descriptive summaries; no fitted latent-state model is used.\n\n"
        text+="Memory values are unavailable without matching qualified fixtures; empty panels establish no measured savings. Memory curves use actual GPU process residence measured in each retained-count shape's qualified full-context fixture, with the same KV capacity and execution conditions as native. Ordinary generation residence is also retained separately; serialized bytes and padding are not resident-memory measurements.\n\n"
        text+=f"Observed local accounting usage: {observed['actual_local_hours']:.3f} hours. The authorized ceiling is 42,050 hours, reduced by observed project expenditure and outstanding reservations.\n\n"
        text+="Terminal status: "+json.dumps(observed["terminal"],sort_keys=True)+"\n"
        handle.write(text.encode())
    publish(directory/'FINAL_SCIENTIFIC_DECISION.json',{'REPORT_COMPLETE':True,'SCIENCE_COMPLETE':bool(scientific_complete),
        'paired_accuracy_memory':str(directory/'UNCERTAINTY.json'),'causal_dynamics_and_unknowns':str(directory/'cells.parquet'),
        'missingness':str(directory/'missingness.csv'),'figures_binding':str(directory/'FIGURES.json'),
        'no_zero_crossing_equivalence_claim':True,'all_seven_models':list(MODELS)})
    from .contracts import file_sha
    publish(directory/'EXPORT_FILES.json',{'files':{str(p):file_sha(p) for p in sorted(directory.rglob('*'))
        if p.is_file() and p.name not in ('REPORT.json','EXPORT_FILES.json','PAPER_DELIVERY.json')}})
    report=publish(directory/"REPORT.json",report_value)
    return deliver(campaign,directory,report,final=final,phase=phase)
