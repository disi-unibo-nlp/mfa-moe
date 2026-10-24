"""Small post-hoc transition analysis of completed per-question summaries."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib
import numpy as np
from analyze_pilot_metrics import statistics,bh_adjust,CLASSES,write_json,write_rows

root=Path(__file__).resolve().parents[1]
base=root/'results/pilot-metrics/correlations-v1'
out=root/'results/pilot-metrics/transitions-v1'
rows=[json.loads(p.read_text()) for p in sorted(base.glob('trace-*.json'))]
assert len(rows)==64 and len({r['question_id'] for r in rows})==64
counts=[Counter((t['source'],t['destination']) for t in r['transitions']) for r in rows]
results=[]
for mode in ('next_sentence','conditional_class_change'):
 for source in CLASSES:
  destinations=list(CLASSES) if mode=='next_sentence' else [x for x in CLASSES if x!=source]
  den=np.array([sum(c[(source,d)] for d in destinations) for c in counts])
  available=np.flatnonzero(den>0)
  for dest in destinations:
   x=np.array([counts[i][(source,dest)]/den[i] for i in available])
   y=np.array([int(rows[i]['is_correct']) for i in available])
   controls=[[np.log(rows[i]['completion_tokens']),int(rows[i]['capped'])] for i in available]
   yes=x[y==1];no=x[y==0]
   support=int(np.sum(x>0));supported=min(len(yes),len(no),support)>=10
   rng=np.random.default_rng(42)
   diff=yes[rng.integers(0,len(yes),(1000,len(yes)))].mean(1)-no[rng.integers(0,len(no),(1000,len(no)))].mean(1)
   for adj,c in [('none',None),('log_length_and_cap',controls)]:
    stat=statistics(x,y,c)
    stat.update(mode=mode,source=source,destination=dest,adjustment=adj,
       mean_correct=float(yes.mean()),mean_incorrect=float(no.mean()),
       correct_questions=len(yes),incorrect_questions=len(no),edge_support=support,
       raw_difference=float(yes.mean()-no.mean()),raw_difference_ci=np.quantile(diff,[.025,.975]).tolist(),
       support_status='supported' if supported else 'insufficient_edge_or_outcome_support')
    if not supported:stat['p']=None
    results.append(stat)
families=defaultdict(list)
for r in results:families[(r['mode'],r['adjustment'])].append(r)
for rs in families.values():
 for r,q in zip(rs,bh_adjust([r.get('p') for r in rs])):r['q_bh']=q
write_rows(out/'comparisons.jsonl',results)
summary={'status':'complete','questions':64,'rows':len(results),'bootstrap':1000,
 'design':'Post-hoc exploratory scan; one attempt per question; equal weight per eligible question.',
 'denominators':'next_sentence includes self-transitions; conditional_class_change excludes self-transitions and natural termination.',
 'support':'At least 10 correct and 10 incorrect source-eligible questions and 10 questions exhibiting the edge for inference.',
 'inference':'Question-bootstrap intervals; raw correct-minus-incorrect difference uses outcome-stratified question bootstrap; r p-values use t reference; BH separately per transition-mode and adjustment.',
 'limitations':['Retrospective judge labels; not an online predictor.','Question difficulty, position, text and source-transition count may confound results.','Length/cap controls are sensitivity checks; source-transition count is not controlled.','A low transition entropy can reflect few observed transitions; not evidence of better control.'],
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
write_json(out/'summary.json',summary)
for mode in ('next_sentence','conditional_class_change'):
 for adj in ('none','log_length_and_cap'):
  rs=sorted([r for r in results if r['mode']==mode and r['adjustment']==adj and r.get('q_bh') is not None],key=lambda r:r['q_bh'])
  print(mode,adj,'significant',sum(r['q_bh']<.05 for r in rs),'tested',len(rs))
  for r in rs[:8]:print({k:r.get(k) for k in ['source','destination','n','edge_support','r','q_bh','mean_correct','mean_incorrect','raw_difference_ci']})
