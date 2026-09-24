"""Recompute matched attempt- and problem-weighted guiding intervals from raw runs."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];rescore_path=R/'report/guiding_rescoring.json';snapshot=json.loads(rescore_path.read_text())
comparisons={k:v['comparison'] for k,v in snapshot['runs'].items() if v['status']=='complete'};out={'method':'paired_dataset_stratified_problem_percentile','replicates':5000,'seed':42,'source_sha256':{},'runs':{}}
ordered=[('Qwen fixed positive +1','results/moe_identity_guiding/qwen_global/sampling/full'),('Qwen fixed positive +2','results/moe_identity_guiding/qwen_strength_2/sampling/full'),('Qwen fixed positive -1','results/moe_identity_guiding/qwen_strength_-1/sampling/full'),('GPT-OSS fixed positive +1','results/moe_identity_guiding/oss_strength_1/sampling/full'),('GPT-OSS fixed positive +2','results/moe_identity_guiding/oss_strength_2/sampling/full'),('GPT-OSS fixed positive -1','results/moe_identity_guiding/oss_strength_-1/sampling/full'),('GPT-OSS fixed negative +1','results/moe_identity_guiding/oss_global/variants/negative_fixed/strength_1/sampling/full'),('GPT-OSS paper positive +1','results/moe_identity_guiding/oss_global/variants/positive_paper_eps_0.01/strength_1/sampling/full'),('GPT-OSS paper positive -1','results/moe_identity_guiding/oss_global/variants/positive_paper_eps_0.01/strength_-1/sampling/full'),('GPT-OSS paper negative +1','results/moe_identity_guiding/oss_global/variants/negative_paper_eps_0.01/strength_1/sampling/full'),('Qwen margin (c=16)','results/moe_margin_guiding/qwen_global/sampling/full'),('Qwen margin (c=25)','results/moe_margin_guiding/qwen_strength_1/sampling/full'),('GPT-OSS margin (c=25)','results/moe_margin_guiding/oss_strength_1/sampling/full')]
for label,folder in ordered:
 ref=comparisons[folder];groups=[]
 for cond in ('baseline','guided'):
  for file in ('manifest.json','generations.jsonl'):
   name=f'{folder}/{cond}/{file}';h=hashlib.sha256();records={}
   with (R/name).open('rb') as f:
    if file.endswith('.jsonl'):
     for line in f:
      h.update(line);r=json.loads(line);assert r['id'] not in records
      decision=snapshot['conditions'][f'{folder}/{cond}'][r['id']];assert decision['original_is_correct']==r['is_correct']
      records[r['id']]={k:r[k] for k in ('input','prompt_token_ids','sampling_args','is_correct')}
      records[r['id']]['is_correct']=decision['is_correct']
    else:
     raw=f.read();h.update(raw);m=json.loads(raw);assert m['status']=='complete'
   assert h.hexdigest()==snapshot['source_sha256'][name],name
   out['source_sha256'][name]=h.hexdigest()
   if records:groups.append(records)
 a,b=groups;assert len(a)==1395 and a.keys()==b.keys();grouped={}
 for key in sorted(a):
  for field in ('input','prompt_token_ids','sampling_args'):assert a[key][field]==b[key][field]
  data=a[key]['input'];ds=data['dataset'];problem=data.get('source_problem_id') or data['problem_id']
  assert type(a[key]['is_correct']) is bool and type(b[key]['is_correct']) is bool
  counts=grouped.setdefault(ds,{}).setdefault(problem,[0,0,0,0]);x=int(a[key]['is_correct']);y=int(b[key]['is_correct'])
  for i,v in enumerate([y-x,1,x,y]):counts[i]+=v
 names=sorted(grouped);strata=[np.array([grouped[d][p] for p in sorted(grouped[d])],dtype=np.int64) for d in names]
 P=sum(len(s) for s in strata);N=sum(s[:,1].sum() for s in strata);assert P==465 and N==1395
 draws=np.zeros((5000,2+len(names)))
 for rep in range(5000):
  rng=np.random.default_rng(np.random.SeedSequence([42,rep]));numerator=denominator=problem_sum=0
  for j,s in enumerate(strata):
   sampled=s[rng.integers(0,len(s),len(s))];summed=sampled.sum(axis=0)
   numerator+=summed[0];denominator+=summed[1];problem_sum+=(sampled[:,0]/sampled[:,1]).sum()
   draws[rep,j+2]=summed[0]/summed[1]
  draws[rep,0]=numerator/denominator;draws[rep,1]=problem_sum/P
 attempts={'baseline_accuracy':sum(s[:,2].sum() for s in strata)/N,'guided_accuracy':sum(s[:,3].sum() for s in strata)/N,'accuracy_delta':sum(s[:,0].sum() for s in strata)/N,'ci95':np.quantile(draws[:,0],[.025,.975]).tolist()}
 problems={'baseline_accuracy':sum((s[:,2]/s[:,1]).sum() for s in strata)/P,'guided_accuracy':sum((s[:,3]/s[:,1]).sum() for s in strata)/P,'accuracy_delta':sum((s[:,0]/s[:,1]).sum() for s in strata)/P,'ci95':np.quantile(draws[:,1],[.025,.975]).tolist()}
 assert np.allclose(attempts['ci95'],ref['bootstrap']['overall']['ci95'],atol=1e-12,rtol=0)
 assert abs(attempts['accuracy_delta']-ref['accuracy_delta'])<1e-12
 ds_results={}
 for j,(ds,s) in enumerate(zip(names,strata)):
  n=int(s[:,1].sum());correct=s[:,2:].sum(axis=0);interval=np.quantile(draws[:,j+2],[.025,.975]).tolist()
  assert np.allclose(interval,ref['bootstrap']['datasets'][ds]['ci95'],atol=1e-12,rtol=0)
  ds_results[ds]={'num_problems':len(s),'num_attempts':n,'baseline_correct':int(correct[0]),'guided_correct':int(correct[1]),'net_correct':int(s[:,0].sum()),'accuracy_delta':float(s[:,0].sum()/n),'ci95':interval,'improved_problems':int((s[:,0]>0).sum()),'worsened_problems':int((s[:,0]<0).sum()),'unchanged_problems':int((s[:,0]==0).sum())}
 out['runs'][folder]={'label':label,'num_problems':P,'num_attempts':int(N),'attempt_weighted':attempts,'problem_weighted':problems,'datasets':ds_results,'comparison':ref}
 print(label, 'attempt', [round(100*v,3) for v in [attempts['accuracy_delta'],*attempts['ci95']]], 'problem', [round(100*v,3) for v in [problems['accuracy_delta'],*problems['ci95']]],flush=True)
 if 'qwen_strength_1' in folder:print('QWEN MARGIN BY BENCHMARK',json.dumps(ds_results),flush=True)
# The equal-problem statistic equals the problem-count-weighted mean of dataset deltas here.
for r in out['runs'].values():assert abs(sum(x['num_problems']*x['accuracy_delta'] for x in r['datasets'].values())/465-r['problem_weighted']['accuracy_delta'])<1e-12
out['source_sha256']['report/guiding_rescoring.json']=hashlib.sha256(rescore_path.read_bytes()).hexdigest()
out['scoring_contract']=snapshot['scoring_contract']
(R/'report/guiding_sensitivity.json').write_text(json.dumps(out,indent=2)+'\n')
