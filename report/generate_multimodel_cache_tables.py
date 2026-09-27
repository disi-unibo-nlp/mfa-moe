"""Render the verified multi-model cache findings for report and thesis."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NAMES={'oss':'GPT-OSS','qwen':'Qwen','gemma':'Gemma'}


def table(caption,label,cols,header,rows,wide=False):
    lines=[r'\begin{table}[ht]',r'\centering\small',r'\caption{'+caption+'}',r'\label{'+label+'}']
    if wide:lines+=[r'\resizebox{\linewidth}{!}{%']
    lines += [r'\begin{tabular}{'+cols+'}',r'\toprule',header+r' \\',r'\midrule']
    lines += [r+' \\\\' for r in rows]
    lines += [r'\bottomrule',r'\end{tabular}']
    if wide:lines+=['}']
    lines += [r'\end{table}']
    return '\n'.join(lines)+'\n'


def main():
    data={m:json.loads((ROOT/f'report/{m}_cache_100_findings.json').read_text()) for m in NAMES}
    quality=[];loads=[];effects=[]
    for m,d in data.items():
        c=d['comparison']; b=c['baseline'];g=c['guided'];ci=c['bootstrap']['overall']['ci95']
        quality.append(f"{NAMES[m]} & {100*b['accuracy']:.0f}/{100*g['accuracy']:.0f} & {100*c['accuracy_delta']:+.1f} [{100*ci[0]:+.1f}, {100*ci[1]:+.1f}] & {b['mean_generated_tokens']:,.1f}/{g['mean_generated_tokens']:,.1f} & {100*(g['mean_generated_tokens']/b['mean_generated_tokens']-1):+.2f} & {b['truncated']}/{g['truncated']}")
        for budget,e in d['effects'].items():
            s={r['condition']:r for r in d['summaries'] if r['capacity_per_layer']==int(budget)}
            loads.append(f"{NAMES[m]} & {budget} & "+' & '.join(f"{s[k]['loads_per_answer']:,.0f}" for k in ['baseline_lru','baseline_frequency_pins','guided_lru','guided_target_pins']))
            effects.append(f"{NAMES[m]} & {budget} & "+' & '.join(f"{e[k]['change_percent']:+.2f} [{e[k]['ci95'][0]:+.2f}, {e[k]['ci95'][1]:+.2f}]" for k in ['per_answer','per_token']))
    text=table('Matched 100-problem cache experiments. B/G denotes baseline/guided. Accuracy uses offline equivalence rescoring; changes and intervals are percentage points. Token changes are relative percentages.','tab:cache-multimodel-quality','llrrrr','Model & Correct B/G & Change [95\\% CI] & Mean tokens B/G & Token change (\\%) & Capped B/G',quality,True)
    text+=table('Simulated mean expert loads per complete answer, including cold-start pin loads, across all routed layers. Capacity is slots per layer; frequency and target pins use the same slot budget within each model.','tab:cache-multimodel-loads','llrrrr','Model & Slots & Baseline LRU & Baseline frequency & Guided LRU & Guided targets',loads,True)
    text+=table('Relative change (percent) for guided target pinning versus baseline LRU, with paired dataset-stratified problem-bootstrap 95\\% intervals. Negative values mean fewer simulated loads. Intervals use 5,000 resamples, seed 42, and no multiplicity correction.','tab:cache-multimodel-effects','llrr','Model & Slots & Loads/answer: change [CI] & Loads/token: change [CI]',effects,True)
    (ROOT/'report/multimodel_cache_tables.tex').write_text(text)


if __name__=='__main__':main()
