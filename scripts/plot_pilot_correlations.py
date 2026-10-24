from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
p=root/'results/pilot-metrics'
rs=[json.loads(l) for l in (p/'original-v1/combined_correlations.jsonl').open()]
lookup={(r['metric'],r['adjustment']):r for r in rs}
cs=[json.loads(l) for l in (p/'correlations-v1/class_metric_contrasts.jsonl').open()]
fig,(a,b)=plt.subplots(1,2,figsize=(14,6))
names=[('Confidence','router_confidence'),('Top-1 margin','router_margin'),('Selected mass','router_selected_mass'),('Boundary margin','router_boundary_margin'),('Top-k gap','router_topk_non_topk_gap'),('Top-1 switching','router_switch_rate'),('Top-k overlap','router_topk_overlap')]
for i,(label,name) in enumerate(names):
 for adjustment,color,offset in [('none','tab:blue',-.12),('log_length_and_cap','tab:orange',.12)]:
  r=lookup[('full/reasoning/'+name+'_mean_layers',adjustment)];lo,hi=r['confidence_interval']
  a.plot([lo,hi],[i+offset,i+offset],color=color);a.scatter(r['r'],i+offset,color=color,label=('Raw' if adjustment=='none' else 'Length/cap adjusted') if i==0 else None)
a.set_yticks(range(7),[x[0] for x in names]);a.axvline(0,color='gray',ls='--');a.set_xlabel('Correlation with correctness; 95% question-bootstrap CI')
a.set_title('Original routing metrics\nNo global routing association passes BH 5%');a.legend(loc='lower left',fontsize=9);a.invert_yaxis()
classes=['Read','Analyze','Plan','Implement','Explore','Verify','Monitor']
for i,c in enumerate(classes):
 r=next(x for x in cs if x['metric']==c+'/W256/mixture_effective_experts');lo,hi=r['confidence_interval'];color='tab:green' if r['q_bh']<.05 else 'gray'
 b.plot([lo,hi],[i,i],color=color);b.scatter(r['mean_within_question_difference'],i,color=color)
 b.text(6.2,i,f"n={r['questions']}; q={r['q_bh']:.3f}",va='center',fontsize=9)
b.set_yticks(range(7),classes);b.axvline(0,color='gray',ls='--');b.set_xlim(-5.5,10.5);b.invert_yaxis()
b.set_title('Reasoning-class associations\n256-token windows; green: BH q < .05')
b.set_xlabel('Effective-expert-count difference: class endpoint vs others')
fig.suptitle('64 MATH-500 questions: NF4 replay of FP8 completions',fontsize=15)
fig.text(.5,.015,'Retrospective, noncausal associations. Native top-k remains eight; effective count describes window diversity.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.06,1,.95])
for suffix in ['png','pdf']:fig.savefig(p/('pilot-findings.'+suffix),dpi=180)
print(p/'pilot-findings.pdf')
