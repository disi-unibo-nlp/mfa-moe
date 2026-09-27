"""CPU-only fixed-cache proxy from saved identity-hook histograms.

Run from the repository root: python3 report/analyze_pinned_expert_cache.py
Before/after counters share GUIDED hidden states; before is NOT a baseline run.
No temporal cache, SSD traffic, or latency can be recovered from histograms.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = 'results/moe_identity_guiding/qwen_global/sampling/full'


def read(relative):
    return json.loads((ROOT / relative).read_text())


def analyze(manifest):
    policy = manifest['policy']
    workers = manifest['routing_diagnostics']
    if manifest['engine_args']['tensor_parallel_size'] != 1 or len(workers) != 1:
        raise ValueError('This analysis requires one nonreplicated diagnostic worker')
    rows = []
    for layer, d in workers[0]['layers'].items():
        p = policy['layers'][layer]
        targets = [i for i, score in enumerate(p['scores']) if score > 0]
        before, after = d['expert_counts_before'], d['expert_counts_after']
        total = d['token_evaluations'] * policy['top_k']
        assert len(before) == len(after) == policy['num_experts']
        assert sum(before) == sum(after) == total
        assert all(0 <= n <= d['token_evaluations'] for n in before + after)
        # Equal-budget cache chosen only from saved calibration frequencies.
        frequent = [e['expert'] for e in sorted(
            p['experts'], key=lambda e: (-e['frequency_mass'], e['expert']))[:len(targets)]]
        rows.append(dict(
            layer=int(layer), cache_slots=len(targets), targets=targets,
            calibration_frequency_cache=frequent, selections=total,
            target_hits_before=sum(before[i] for i in targets),
            target_hits_after=sum(after[i] for i in targets),
            frequency_hits_before=sum(before[i] for i in frequent),
            frequency_hits_after=sum(after[i] for i in frequent),
            oracle_hits_before=sum(sorted(before, reverse=True)[:len(targets)]),
            changed_topk_sets=d['changed_topk_sets'], token_evaluations=d['token_evaluations']))
    totals = {k: sum(r[k] for r in rows) for k in rows[0]
              if k not in ('layer', 'targets', 'calibration_frequency_cache')}
    for r in rows + [totals]:
        for name in ('target_hits_before', 'target_hits_after', 'frequency_hits_before',
                     'frequency_hits_after', 'oracle_hits_before'):
            r[name + '_fraction'] = r[name] / r['selections']
        r['miss_reduction_same_targets'] = (r['target_hits_after'] - r['target_hits_before']) / (r['selections'] - r['target_hits_before'])
        r['miss_reduction_vs_frequency_cache'] = (r['target_hits_after'] - r['frequency_hits_before']) / (r['selections'] - r['frequency_hits_before'])
        r['miss_reduction_vs_before_oracle'] = (r['target_hits_after'] - r['oracle_hits_before']) / (r['selections'] - r['oracle_hits_before'])
    return rows, totals


def main():
    manifest_path = RUN + '/guided/manifest.json'
    rows, total = analyze(read(manifest_path))
    scores_path = 'report/guiding_rescoring.json'
    comparison = read(scores_path)['runs'][RUN]['comparison']
    result = dict(
        scope='Fixed resident cache at six steered layers, on guided-run states only',
        assumptions=['Pinned experts resident before execution; cold-load cost omitted',
                     'Each nonresident selection counted as one potential load; no reuse outside pinned set',
                     'Equal expert size within each layer; no batching or prefetch model'],
        limitations=['Pre-bias histograms are not separately generated baseline routing',
                     'Counters include prefill, decode and engine padding',
                     'Hook top-k is not separately observed native dispatch; ties can differ',
                     'No temporal order, per-attempt routing, or unsteered-layer counters',
                     'No measured SSD traffic, wall-clock speed, or cache-effect confidence interval',
                     'Aggregate accuracy can conceal benchmark-specific losses'],
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                       for p in (manifest_path, scores_path)},
        layers=rows, aggregate=total, matched_accuracy_and_tokens=comparison)
    (ROOT / 'report/pinned_expert_cache.json').write_text(json.dumps(result, indent=2) + '\n')
    pct = lambda x: f'{100*x:.2f}%'
    lines = ['# Quick pinned-expert cache analysis', '',
             'Reproduce: `python3 report/analyze_pinned_expert_cache.py`', '',
             'Qwen fixed positive strength +1, full stochastic evaluation. This is a new analysis of saved counters, not a new generation or SSD timing experiment.', '',
             '## Fixed resident-cache proxy', '',
             'Pin eight experts per steered layer (48 layer–expert identities across six layers). Count every selection outside that set as a potential expert load. Both columns use the same hidden states from the guided run.', '',
             '| Layer | Target-cache hits before bias | Target-cache hits after bias | Relative reduction in nonresident selections |',
             '|---|---:|---:|---:|']
    for r in rows + [dict(total, layer='All six')]:
        lines.append(f"| {r['layer']} | {pct(r['target_hits_before_fraction'])} | {pct(r['target_hits_after_fraction'])} | {pct(r['miss_reduction_same_targets'])} |")
    lines += ['',
        f"An equal-budget cache chosen by calibration frequency covers {pct(total['frequency_hits_before_fraction'])} of pre-bias selections. Guided target pinning reduces nonresident selections by {pct(total['miss_reduction_vs_frequency_cache'])} against this control on the same guided states.", '',
        f"Even a hindsight-optimal static eight-expert cache covers only {pct(total['oracle_hits_before_fraction'])} of pre-bias selections. Against that static oracle, the reduction is {pct(total['miss_reduction_vs_before_oracle'])}. This oracle is an analytical control, not a deployable baseline or a bound on adaptive caching.", '',
        '## Separately generated, matched quality evaluation', '',
        f"Across {comparison['num_examples']} attempts on {comparison['bootstrap']['overall']['num_problems']} problems, rescored accuracy is {pct(comparison['baseline']['accuracy'])} baseline and {pct(comparison['guided']['accuracy'])} guided: {100*comparison['accuracy_delta']:+.2f} percentage points.",
        f"The saved paired problem-bootstrap 95% interval is [{100*comparison['bootstrap']['overall']['ci95'][0]:+.2f}, {100*comparison['bootstrap']['overall']['ci95'][1]:+.2f}] points; this is not proof of non-inferiority.",
        f"Mean generated tokens: {comparison['baseline']['mean_generated_tokens']:.1f} baseline versus {comparison['guided']['mean_generated_tokens']:.1f} guided. AIME24 alone changes by {100*comparison['bootstrap']['datasets']['aime24']['accuracy_delta']:+.2f} points; aggregate quality is not uniform across tasks.", '',
        '## Interpretation and limits', '',
        'The bias demonstrably concentrates hook-selected experts into a small fixed set, alongside a small aggregate accuracy difference in the matched generation experiment. This supports testing an SSD-streaming trade-off. It does not establish reduced actual I/O or faster inference.', '',
        *['- ' + s + '.' for s in result['limitations']], '',
        'The six-layer reduction must not be reported as a whole-model reduction. Prefill and batched tokens can share one expert load, so selection counts are not disk-read counts. Cache warm-up and the remaining layers also cost time.', '',
        'Next validation: collect ordered decode-only expert IDs for paired baseline/guided generations; compare equal-memory target pinning and LRU caches; then benchmark actual expert reads and end-to-end answer latency. Fix the acceptable accuracy loss before choosing the deployment policy.', '']
    (ROOT / 'report/pinned_expert_cache.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in total.items() if 'fraction' in k or 'reduction' in k}, indent=2))


if __name__ == '__main__':
    main()
