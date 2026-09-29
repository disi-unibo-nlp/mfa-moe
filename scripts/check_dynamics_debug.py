"""Bounded independent checks of saved debug router tensors; never loads a model."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from moe_exp.correlation_pipeline.dynamics.common import artifact_path, write_json
from moe_exp.correlation_pipeline.dynamics.routing import RoutingReducer, probabilities

p=argparse.ArgumentParser()
p.add_argument("--traces",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
args=p.parse_args()
checks=[]
with args.traces.open() as handle:
    records=[json.loads(line) for line in handle if line.strip()]
for record in records:
    router=torch.load(artifact_path(args.traces,record["model_logs"]["router_logits"]),
                      map_location="cpu",weights_only=True,mmap=True)
    experts=torch.load(artifact_path(args.traces,record["model_logs"]["selected_experts"]),
                       map_location="cpu",weights_only=True,mmap=True)
    for li in (0,len(router)-1):
        scores=router[li,:320].float().numpy()
        ids=experts[li,:320].numpy()
        probability=probabilities(scores)
        layer=record["model_logs"]["layer_indices"][li]
        reducer=RoutingReducer(probability.shape[1],ids.shape[1],model=record["model_id"],layer=layer)
        rows=[]
        for start in range(0,len(ids),37):
            rows.extend(reducer.push(ids[start:start+37],full_probabilities=probability[start:start+37]))
        worst=0.
        for row in rows:
            a,b=row["start_token"],row["end_token"]
            v=probability[a:b]
            local=np.mean([-sum(x*np.log(x)) for x in v])
            marginal=-sum(v.mean(0)*np.log(v.mean(0)))
            worst=max(worst,abs(local-row["full_local_entropy"]),abs(marginal-row["full_marginal_entropy"]))
            for lag in (1,4,16,64):
                values=[len(set(ids[t])&set(ids[t-lag]))/ids.shape[1] for t in range(max(a,lag),b)]
                if values:
                    worst=max(worst,abs(np.mean(values)-row[f"overlap_lag{lag}"]))
        assert worst<1e-10, worst
        assert reducer.retained_tokens<=320
        assert all(r["mixture_jsd"] is None and r["boundary_gaps"] is None for r in rows)
        checks.append(dict(problem_id=record["problem_id"],layer=layer,windows=len(rows),max_error=worst))
result=dict(status="passed",checks=checks,
            limitations="Saved debug bundle lacks executed weights and explicit native selection-score semantics.")
write_json(args.output,result)
print(json.dumps(result,indent=2))
