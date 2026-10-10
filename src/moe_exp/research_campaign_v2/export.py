"""Finite result package, paired uncertainty, missing cells and terminal reports."""
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from functools import lru_cache
import json
import os
import numpy as np

from .artifacts import atomic_file,frame
from .contracts import CLASSES,CONFIGURATIONS,METHODS,MODELS,REMOVALS,Units,check_files,digest,load,publish
from .statistics import bootstrap_panel


def _resident(result,root,model,standardized=True):
    if not result:
        return None
    reports=result.get("memory_after")
    if standardized:
        mask=result.get("assignment",{}).get("mask")
        key=digest([len(mask[str(l)]) for l in sorted(map(int,mask))]) if mask else "native"
        p=root/model/"qualification"/(key+".json")
        reports=None
        if p.exists():
            q=load(p)
            if q["status"]=="complete":reports=q.get("fixture",{}).get("memory_after")
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
    prefixes=load(prefix_path)["records"] if prefix_path.exists() else [{"model":model,**r,"prefix":None,"configuration_indices":[0,1]} for r in cohorts["causal"]]
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
        native_causal={(c["benchmark"],c["family"]):c for c in native["cells"] if c["study"]=="causal"}
        cells=[native_causal.get((c["benchmark"],c["family"]),c) if c.get("arm")=="native" else c for c in cells]
        return {"records":native["records"],"cells":cells}
    return {"records":[],"cells":cells}


def _response(row,result,context,annotations,cell):
    values={}
    sentences=[]
    for s in (context or {}).get("sentences",[]):
        label=(annotations.state(s["annotation_uid"]) or {}).get("label")
        sentences.append({**s,"label":label})
    first=next((s for s in sentences if s["first_complete"]),None)
    label=first["label"] if first else None
    values["first_class"]=label
    values["first_sentence_status"]="labelled" if label else "unknown" if first else "termination_or_no_complete_sentence"
    values["first_sentence_tokens"]=first["n_tokens"] if first else None
    values["termination"]=(result or {}).get("finish_reason")
    terminated=first is None and (result or {}).get("status")=="complete" and result.get("finish_reason")=="stop"
    values["first_termination"]=float(terminated) if result and result.get("status")=="complete" else None
    values["first_unknown"]=float(label is None and not terminated) if result and result.get("status")=="complete" else None
    target=cell.get("target")
    for c in CLASSES:
        values["first:"+c]=float(label==c) if label else 0. if terminated else None
        values["first_lower:"+c]=float(label==c) if label else 0.
        values["first_upper:"+c]=float(label==c) if label else 0. if terminated else 1.
    windows=cell.get("windows") or ([row["window"]] if row and row.get("window") else [])
    pulse=(row or {}).get("action",{}).get("pulse_tokens",0) if (row or {}).get("action") else 0
    for index,window in enumerate(windows):
        name="w"+str(index)
        observed=[s for s in sentences if s["token_start"]<window]
        labelled=sum(max(0,min(s["token_end"],window)-s["token_start"]) for s in observed if s["label"])
        available=min(window,(result or {}).get("generated_tokens",0))
        values[f"coverage:{name}"]=labelled/available if available else None
        for c in CLASSES:
            n=sum(max(0,min(s["token_end"],window)-s["token_start"]) for s in observed if s["label"]==c)
            values[f"occupancy:{name}:{c}"]=n/labelled if labelled else None
            values[f"occupancy_lower:{name}:{c}"]=n/available if available else None
            values[f"occupancy_upper:{name}:{c}"]=(n+available-labelled)/available if available else None
        recovery=[s for s in observed if s["token_end"]>pulse and s["label"]]
        recovery_length=lambda s:max(0,min(s["token_end"],window)-max(s["token_start"],pulse))
        for c in CLASSES:
            denominator=sum(recovery_length(s) for s in recovery)
            values[f"recovery:{name}:{c}"]=sum(recovery_length(s) for s in recovery if s["label"]==c)/denominator if denominator else None
    adjacent=[(a,b) for a,b in zip(sentences,sentences[1:]) if a["label"] and b["label"] and b["sentence_index"]==a["sentence_index"]+1 and a["segment"]==b["segment"]]
    values["class_switches"]=sum(a["label"]!=b["label"] for a,b in adjacent) if adjacent else None
    values["observed_boundaries"]=len(adjacent)
    seen=set();reentry=0;run=0;longest=0
    previous=None
    for s in sentences:
        if not s["label"] or previous is None or s["sentence_index"]!=previous["sentence_index"]+1 or s["segment"]!=previous["segment"]:
            run=0;seen=set()
        if s["label"]:
            if previous and previous["label"]!=s["label"] and s["label"] in seen:
                reentry+=1
            seen.add(s["label"])
            run=run+s["n_tokens"] if previous and previous["label"]==s["label"] else s["n_tokens"]
            longest=max(longest,run)
        previous=s
    values.update(reentry_count=reentry if adjacent else None,longest_observed_dwell_tokens=longest if any(s["label"] for s in sentences) else None,
        persistence_to_window_end=float(sentences[-1]["label"]==target) if sentences and sentences[-1]["label"] else None,
        recovery_to_source=float(next((s["label"] for s in sentences if s["token_start"]>=pulse and s["label"]),None)==cell.get("prefix",{}).get("source_class")) if any(s["token_start"]>=pulse and s["label"] for s in sentences) else None,
        compensatory_transitions=sum(a["label"]==target and b["label"]!=target for a,b in adjacent) if adjacent else None)
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


def _panels(rows):
    grouped=defaultdict(list)
    for r in rows:
        if r["study"]=="memory":
            panel=("memory",r["model"])
            name=f"{r['method']}:{r['allocation']}:{r['removal']}"
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
                measurements={m:v for m,v in r.items() if m.startswith(("first:","first_lower:","first_upper:","occupancy:","recovery:","native_difference:")) or m in (
                    "strict_correct","unknown_lower","unknown_upper","new_tokens","routing_displacement","pulse_displacement",
                    "recovery_displacement","target_slot_excess","class_switches","reentry_count","longest_observed_dwell_tokens",
                    "persistence_to_window_end","recovery_to_source","compensatory_transitions","first_termination","first_unknown")}
            for metric,value in measurements.items():
                col=name+"|"+metric
                cells[familykey][col]=value
                columns.add(col)
        # Paired target/random differences are in the same declared simultaneous
        # panel, using the same family multiplicity and no independent-arm draws.
        if key[0]=="causal":
            for values in cells.values():
                for col in list(values):
                    if ":target|" not in col:
                        continue
                    random=col.replace(":target|",":random|")
                    diff=col.replace(":target|",":target-minus-random|")
                    values[diff]=values[col]-values[random] if values[col] is not None and values.get(random) is not None else None
                    columns.add(diff)
        panel_rows=[{"benchmark":b,"family":f,"question":q,**values} for (b,f,q),values in cells.items()]
        if columns:
            panels.append({"identity":list(key),**bootstrap_panel(panel_rows,sorted(columns),panel="|".join(map(str,key)))})
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
        panels.append({"identity":["native","paired-seven-model"],**bootstrap_panel(native_rows,columns,panel="native:paired-seven-model")})
    # Primary central treatment-minus-control comparisons share whole-family
    # multiplicities across all seven models and both arms. Sensitivity panels
    # retain the complete per-model configured outcomes separately.
    grouped=defaultdict(lambda:defaultdict(dict))
    for r in rows:
        if r["study"]!="causal" or r.get("threshold_index")!=0 or r.get("arm") not in ("target","random"):continue
        key=r["target"],r["depth_third"],r["sign"]
        cell=grouped[key][r["benchmark"],r["family"]]
        for metric in ("first:"+r["target"],"strict_correct","new_tokens"):
            cell[r["model"]+"|"+r["arm"]+"|"+metric]=r.get(metric)
    for key,families in grouped.items():
        panel_rows=[];columns=[]
        for (b,f),values in families.items():
            row={"benchmark":b,"family":f}
            for m in MODELS:
                for metric in ("first:"+key[0],"strict_correct","new_tokens"):
                    a,c=values.get(m+"|target|"+metric),values.get(m+"|random|"+metric)
                    name=m+"|target-minus-random|"+metric
                    row[name]=a-c if a is not None and c is not None else None
                    if name not in columns:columns.append(name)
            panel_rows.append(row)
        panels.append({"identity":["causal","paired-seven-model","MMM",*key],**bootstrap_panel(panel_rows,columns,panel="causal:paired-seven-model:"+str(key))})
    return panels


def _figures(directory,rows,campaign):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":8,"svg.fonttype":"none","svg.hashsalt":campaign["campaign_id"]})
    figures=[]
    for model in MODELS:
        memory=[r for r in rows if r["model"]==model and r["study"]=="memory"]
        causal=[r for r in rows if r["model"]==model and r["study"]=="causal"]
        fig,axes=plt.subplots(1,2,figsize=(10,3.3))
        for allocation,axis in zip(("equal","unequal"),axes):
            for method in METHODS:
                x=[];y=[]
                for removal in REMOVALS:
                    chosen=[r for r in memory if r.get("method")==method and r.get("allocation")==allocation and r.get("removal")==removal and r.get("resident_bytes") is not None and r.get("strict_correct") is not None]
                    if chosen:
                        from .design import family_weights
                        w=family_weights(chosen)
                        x.append(float(np.dot(w,[r["resident_bytes"] for r in chosen]))/1024**3)
                        y.append(float(np.dot(w,[r["strict_correct"] for r in chosen])))
                if x:
                    axis.plot(x,y,"o-",label=method)
            axis.set(title=model+" / "+allocation,xlabel="Actual process resident memory (GiB)",ylabel="Strict accuracy",ylim=(0,1))
            if axis.lines:
                axis.legend(fontsize=6)
            else:
                axis.text(.5,.5,"Unmeasured",transform=axis.transAxes,ha="center")
        fig.tight_layout()
        path=directory/"figures"/(model+"-memory.svg")
        path.parent.mkdir(parents=True,exist_ok=True)
        with atomic_file(path) as handle:
            fig.savefig(handle,format="svg",metadata={"Date":None})
        plt.close(fig);figures.append(str(path))
        for metric,title in (("first:","First complete sentence"),("recovery:","Pulse recovery dynamics")):
            destinations=list(CLASSES)+(["Termination","Unknown"] if metric=="first:" else [])
            data=np.full((7,len(destinations)),np.nan)
            for i,target in enumerate(CLASSES):
                for j,destination in enumerate(destinations):
                    candidates=[r for r in causal if r.get("target")==target and r.get("arm")=="target" and r.get("threshold_index")==0]
                    vals=[]
                    for r in candidates:
                        keys=["first_termination" if destination=="Termination" else "first_unknown"] if destination in ("Termination","Unknown") else [k for k in r if k.startswith(metric) and k.endswith(":"+destination)]
                        if keys and r.get(keys[-1]) is not None:
                            vals.append((r,r[keys[-1]]))
                    if vals:
                        from .design import family_weights
                        data[i,j]=np.dot(family_weights([r for r,v in vals]),[v for r,v in vals])
            fig,axis=plt.subplots(figsize=(4.5,3.8))
            cmap=plt.get_cmap("viridis").copy();cmap.set_bad("lightgrey")
            im=axis.imshow(data,vmin=0,vmax=1,cmap=cmap)
            axis.set_xticks(range(len(destinations)),destinations,rotation=45,ha="right")
            axis.set_yticks(range(7),CLASSES)
            axis.set(title=model+" / "+title,xlabel="Observed class",ylabel="Target class")
            fig.colorbar(im,ax=axis,label="Probability; grey = missing")
            fig.tight_layout()
            path=directory/"figures"/(model+("-response.svg" if metric=="first:" else "-dynamics.svg"))
            with atomic_file(path) as handle:
                fig.savefig(handle,format="svg",metadata={"Date":None})
            plt.close(fig);figures.append(str(path))
    return figures


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
                    for panel in result["panels"]:
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


def _assemble(campaign):
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
        for name in ("CONTEXTS-native.json","CONTEXTS-price.json","CONTEXTS.json"):
            if (root/model/name).exists():contexts.extend(load(root/model/name)["records"])
        context_lookup={r["uid"]:r for r in contexts}
        native_lookup={(r["benchmark"],r["family"]):r["uid"] for r in table["cells"] if r.get("arm")=="native" and r.get("uid")}
        for cell in table["cells"]:
            uid=cell.get("uid")
            result=executions.state(uid) if uid else None
            status_=cell["status"]
            if uid:
                status_=(result or {}).get("status","upstream_unavailable")
                if status_=="started":
                    status_="execution_failed"
            if status_=="pending":
                status_="upstream_unavailable"
            row={**cell,"status":status_}
            if result and result.get("status")!="complete":row["retained_partial_tokens"]=result.get("generated_tokens")
            if result and result.get("status")=="complete":
                row["status"]="reused" if result.get("reused") else "complete"
                row.update(resident_bytes=_resident(result,root,model),execution_resident_bytes=_resident(result,root,model,False),
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
                        native_uid=native_lookup.get((cell["benchmark"],cell["family"]))
                        if native_uid:
                            native_values,_=_response(task_lookup.get(uid),executions.state(native_uid),context_lookup.get(native_uid),annotations,cell)
                            for metric,value in values.items():
                                if metric.startswith(("first:","occupancy:","recovery:")):
                                    reference=native_values.get(metric)
                                    row["native_difference:"+metric]=value-reference if value is not None and reference is not None else None
                    sentence_rows.extend({"uid":uid,"model":model,"benchmark":cell["benchmark"],"family":cell["family"],**s} for s in sentences)
            scientific_complete &= row["status"] in ("complete","reused")
            if uid and row["status"] in ("complete","reused"):
                scientific_complete &= (scores.state(uid) or {}).get("status")=="complete"
                if cell["study"]=="causal":
                    scientific_complete &= uid in context_lookup or not result.get("generated_tokens")
                    scientific_complete &= all((annotations.state(s["annotation_uid"]) or {}).get("status") in
                        ("complete","reused","invalid","truncated") for s in context_lookup.get(uid,{}).get("sentences",[]))
            rows.append(row)
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


def closeout(campaign,final=True):
    root=Path(campaign["root"])
    directory=root/("output" if final else "snapshots/2026-10-15")
    if (directory/"REPORT.json").exists():
        report=load(directory/"REPORT.json")
        if final and not (root/"REPORT_COMPLETE.json").exists():
            publish(root/"SCIENCE_COMPLETE.json",{"complete":report["SCIENCE_COMPLETE"],"report_sha256":report["sha256"]})
            publish(root/"REPORT_COMPLETE.json",{"complete":True,"report_sha256":report["sha256"]})
        return {"status":"complete","REPORT_COMPLETE":final,"SCIENCE_COMPLETE":report["SCIENCE_COMPLETE"]}
    directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    if not final and (root/"REPORT_COMPLETE.json").exists():
        # If science closed before the milestone, reuse its committed package
        # through same-filesystem hardlinks. No repeat analysis or data copy.
        previous=load(root/"output/REPORT.json")
        for source in (root/"output").rglob("*"):
            if source.is_file() and source.name!="REPORT.json":
                target=directory/source.relative_to(root/"output")
                target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
                if not target.exists():os.link(source,target)
        from .controller import status
        report=publish(directory/"REPORT.json",{**previous,"snapshot":True,
            "reused_final_report_sha256":previous["sha256"],"resource_usage":status(campaign),
            "observed_utc":datetime.now(timezone.utc).isoformat()})
        return {"status":"complete","REPORT_COMPLETE":False,"SCIENCE_COMPLETE":report["SCIENCE_COMPLETE"]}
    statepath=directory/"EXPORT_STATE.json"
    if statepath.exists():
        state=load(statepath)
    else:
        rows,sentence_rows,ledgers,scientific_complete=_assemble(campaign)
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
    panels=_panels(rows)+_cross_model_panels(rows,campaign)
    publish(directory/"UNCERTAINTY.json",{"panels":panels,"resamples":10000,"run_to_run_sd_reported":False})
    figures=_figures(directory,rows,campaign)+_observational_figures(directory,campaign)
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
    report_value={"campaign_id":campaign["campaign_id"],"REPORT_COMPLETE":True,
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
        text+=f"This {'October 15 snapshot' if not final else 'final report'} retains {len(rows):,} configured cells across all seven models. Missing or failed cells remain explicit and do not represent null effects.\n\n"
        text+="Accuracy uses the original strict benchmark contract. Unresolved answers retain lower and upper bounds. The reported intervals use 10,000 paired whole-family bootstrap resamples with equal benchmark and family weights. They condition on frozen selections; selection stability is separate. One replicate was run, so no run-to-run standard deviation is reported.\n\n"
        text+="Coordination declares separate linear and log-odds subpanels with error allowances 0.025 each and the same 10,000 whole-family resamples. Undefined draws are retained; an unestimable log-odds outcome does not suppress a supported linear interval. Full-A token ownership is recovered from original labels/offsets without changing frozen causal enrollment. Native accuracy joins each exact original attempt.\n\n"
        text+="Immediate response and pulse/recovery dynamics are exported together. Persistence, recovery to the source class, re-entry and compensatory transitions are direct observed summaries; no fitted latent-state model is used.\n\n"
        text+="Memory curves use actual GPU process residence measured in each retained-count shape's qualified full-context fixture, with the same KV capacity and execution conditions as native. Ordinary generation residence is also retained separately; serialized bytes and padding are not resident-memory measurements.\n\n"
        text+=f"Observed local accounting usage: {observed['actual_local_hours']:.3f} hours. The authorized ceiling is 42,050 hours, reduced by observed project expenditure and outstanding reservations.\n\n"
        text+="Terminal status: "+json.dumps(observed["terminal"],sort_keys=True)+"\n"
        handle.write(text.encode())
    report=publish(directory/"REPORT.json",report_value)
    if final:
        publish(root/"SCIENCE_COMPLETE.json",{"complete":bool(scientific_complete),"report_sha256":report["sha256"]})
        publish(root/"REPORT_COMPLETE.json",{"complete":True,"report_sha256":report["sha256"]})
    return {"status":"complete","REPORT_COMPLETE":final,"SCIENCE_COMPLETE":bool(scientific_complete),"report_sha256":report["sha256"]}
