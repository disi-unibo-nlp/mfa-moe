"""Generate a versioned claim ledger and paper figures from sealed small summaries."""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
DOC = REPO / 'report/experimental-resume-v1'
R = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
S = R / 'steering-v1'
BASE = S / 'runs/routing-control-v1'
sys.path.insert(0, str(REPO / 'src'))
from moe_exp.routing_control.counterfactual import digest, sealed

INPUTS = {
    'b1': S / 'runs/resume-v1/r3e-b1-clean-v2/result.json',
    'first_stage': BASE / 'micro-screen-first-stage-qual4-ac4c9651e71fe067/FIRST_STAGE.json',
    'token_zero': DOC / 'CAUSAL_MICROSCREEN_TOKEN_ZERO_AUDIT_v1.json',
    'semantic': DOC / 'CAUSAL_MICROSCREEN_SEMANTIC_PILOT_v1.json',
    'native_replay': DOC / 'CAUSAL_NATIVE_ONLY_REPLAY_AUDIT_v3.json',
    'score_failure': BASE / 'counterfactual-qualification-b7c4c5273ab6fdc7/QUALIFICATION.json',
    'approach_proxy': BASE / 'approach-boundary-proxy-v1/RESULT.json',
    'veto_scout': DOC / 'NATIVE_PREFIX_VETO_12_SCOUT_COMPARISON_v0.json',
}
ARMS = ('native', 'native_duplicate', 'random_bias0.5', 'random_bias1', 'target_bias0.5', 'target_bias1')
LABELS = ('Native', 'Native repeat', 'Random +0.5', 'Random +1', 'Target +0.5', 'Target +1')


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('paper artifacts require an allocated CPU Slurm step')
    values = {key: sealed(path) for key, path in INPUTS.items()}
    driver_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    binding = {'schema': 'discovery-evidence-input-binding-v2', 'driver_sha256': driver_sha,
        'sources': {key: {'path': str(INPUTS[key]), 'sha256': value['sha256']}
            for key, value in values.items()}}
    bundle_sha = digest(binding)
    out = R / 'paper/discovery-evidence-v2' / bundle_sha[:16]
    out.mkdir(parents=True, exist_ok=True)
    frozen_path = out / 'FROZEN.json'
    if frozen_path.exists():
        frozen = sealed(frozen_path)
        if frozen['input_binding_sha256'] != bundle_sha:
            raise ValueError('paper output binding changed')
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected
            for path, expected in frozen['files'].items()):
            raise ValueError('completed paper artifact changed')
        print(json.dumps({'status': 'VERIFIED_EXISTING', 'output': str(out)}))
        return
    (out / 'INPUT_BINDING.json').write_text(json.dumps({**binding, 'sha256': bundle_sha}, indent=1) + '\n')
    b1, first, token, semantic, replay, failure, approach = (values[key] for key in (
        'b1', 'first_stage', 'token_zero', 'semantic', 'native_replay', 'score_failure', 'approach_proxy'))
    selected = {arm: sum(bool(row['target_selected']) for row in token['records'] if row['arm'] == arm)
        for arm in ARMS}
    summaries = first['arm_summaries']
    rows = [{'arm': arm, 'assigned': summaries[arm]['assigned'],
        'target_fraction_conditional_on_generated_reasoning': summaries[arm]['mean_target_fraction_on_valid'],
        'target_selected_per_256_assigned': summaries[arm]['mean_target_selected_per_256_assigned'],
        'first_token_target_selected': selected[arm], 'first_token_denominator': 8,
        'jointly_accepted_start_prefixes': semantic['start_both_positive_unique_prefixes'],
        'unique_start_prefixes': semantic['start_unique_prefixes']} for arm in ARMS]
    with (out / 'routing_pilot.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    common = {'uncertainty': None, 'multiplicity_family': 'exploratory engineering pilot; descriptive counts, no confirmatory p-value'}
    claims = [{**common, 'id': 'clean-gpt-b1-joint-routing-gain',
        'population': {**b1['population'], 'model': 'GPT source routing dataset, exact-ID-clean'},
        'intervention': 'native observational, family-grouped held-out class-conditioned routing estimator',
        'estimate': b1['family_clustered']['estimate_nats_per_token_pair'],
        'uncertainty': {'method': 'family-clustered bootstrap', 'ci95': b1['family_clustered']['family_cluster_ci95']},
        'multiplicity_family': 'single preregistered joint B1 endpoint; class/pair secondary quantities separate',
        'status': 'CLEAN_OBSERVATIONAL', 'source': str(INPUTS['b1']),
        'interpretation': 'class-associated routing information; does not establish latent semantic specialization or transfer to Qwen3.6'},
        {**common, 'id': 'qwen36-target-expert-routing-pilot',
        'population': 'four selected discovery families × two seeds, Qwen3.6, 256-token continuations',
        'intervention': 'layer28 experts9/189, target bias0.5/1 vs zero and exposure-matched random biases, k8/shared expert retained',
        'estimate': {arm: rows[i] for i, arm in enumerate(ARMS)},
        'status': 'ROUTING_FIRST_STAGE_PILOT', 'source': str(INPUTS['first_stage']),
        'interpretation': 'whole-continuation target fractions include post-intervention text changes and reasoning-closure differences; first-token comparison isolates a common starting prefix'},
        {**common, 'id': 'qwen36-four-prefix-semantic-enrollment-audit',
        'population': 'same four selected pilot prefixes; two arm-blind Qwen3.8 draws per start',
        'intervention': 'full-prefix start eligibility audit separate from continuation target ratings',
        'estimate': {'jointly_accepted': semantic['start_both_positive_unique_prefixes'],
            'unique_prefixes': semantic['start_unique_prefixes'], 'valid_stopped_ratings': semantic['valid_stopped_ratings'],
            'assigned_ratings': 104}, 'status': 'ENROLLMENT_FAILURE_IDENTIFIED', 'source': str(INPUTS['semantic']),
        'interpretation': 'zero jointly accepted unchecked-candidate starts; target-only continuation positives cannot identify transition-control effects; LLM audit, not human truth'},
        {**common, 'id': 'qwen36-native-only-repeatability',
        'population': 'same four pilot prefixes × two seeds, 16 zero-policy requests under exact populated pilot table',
        'intervention': 'native-only duplicate replay; no edited neighbor requests',
        'estimate': {'token_stream_equal_pairs': sum(row['tokens_equal'] for row in replay['native_only_pairs']),
            'first_token_unordered_topk_equal_pairs': sum(row['first_token_unordered_topk_equal'] for row in replay['native_only_pairs']),
            'first_token_target_mask_equal_pairs': sum(row['first_token_target_mask_equal'] for row in replay['native_only_pairs']), 'pairs': 8},
        'status': 'NUMERICAL_REPEATABILITY_LIMITATION', 'source': str(INPUTS['native_replay']),
        'interpretation': 'native-only variability reproduces route differences; cannot assert bitwise neighbor isolation or exact exchangeability'},
        {**common, 'id': 'counterfactual-qualification-v2',
        'population': 'four pilot prefixes × four native positions, 148 engineering assignments',
        'intervention': 'bias/force inclusion/exclusion, teacher-force score alignment, closure and reset controls',
        'estimate': {key: failure[key] for key in ('assigned', 'completed', 'alignment_mean_abs',
            'alignment_max_abs', 'native_repeat_mean_abs', 'native_repeat_max_abs', 'recovery_pass')},
        'status': 'FAILED_REGISTERED_SCORE_ALIGNMENT', 'source': str(INPUTS['score_failure']),
        'interpretation': 'routing/reset mechanics passed; score alignment failed and cannot rank semantic experts; new identical-prefix API calibration is a separate prospective measurement'},
        {**common, 'id': 'approach-class-proxy-last64-shortlist',
        'population': approach['population'], 'intervention': 'native observational last64 boundary contrast; source Explore, next contiguous Plan/Implement',
        'estimate': approach['support'], 'status': 'OBSERVATIONAL_SHORTLIST_NOT_QUALIFIED',
        'source': str(INPUTS['approach_proxy']),
        'interpretation': '20 matched positive families; one crossfit fold reverses direction and one expert lacks four matched randoms; no action is frozen; identified-approach semantic start not established'}]
    body = {'schema': 'discovery-claim-ledger-v2', 'input_binding_sha256': bundle_sha, 'claims': claims}
    (out / 'CLAIM_LEDGER.json').write_text(json.dumps({**body, 'sha256': digest(body)}, indent=1) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#586a78', '#7d8b95', '#b6b9bd', '#969da5', '#3b80ad', '#dc8a36']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    ysets = ([r['target_fraction_conditional_on_generated_reasoning'] for r in rows],
        [r['target_selected_per_256_assigned'] for r in rows], [r['first_token_target_selected']/8 for r in rows])
    titles = ('A. Over emitted reasoning', 'B. Fixed 256-token denominator', 'C. Identical first prediction prefix')
    for ax, heights, title in zip(axes, ysets, titles):
        bars = ax.bar(range(6), heights, color=colors)
        ax.set_xticks(range(6), LABELS, rotation=55, ha='right')
        ax.set_ylim(0, .85)
        ax.set_title(title, loc='left', fontsize=10)
        ax.set_ylabel('Fraction with ≥1 target expert selected')
        for i, bar in enumerate(bars):
            label = f'{selected[ARMS[i]]}/8' if ax is axes[2] else f'{heights[i]:.3f}'
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.025, label, ha='center', fontsize=9)
    fig.suptitle('Qwen3.6 routing pilot: four families, two seeds; layer 28 experts 9 and 189', fontsize=12)
    fig.text(.01, .01, 'Descriptive engineering data. Panels A/B include diverging text and closure. Jointly eligible semantic starts: 0/4.', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .93))
    fig.savefig(out / 'routing_pilot.pdf')
    fig.savefig(out / 'routing_pilot.png', dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    point = b1['family_clustered']['estimate_nats_per_token_pair']
    lo, hi = b1['family_clustered']['family_cluster_ci95']
    ax.errorbar([point], [0], xerr=np.array([[point-lo], [hi-point]]), fmt='o', capsize=5, color='#3b80ad')
    ax.axvline(0, color='#969da5', lw=.8)
    ax.set_xlim(-.001, .011)
    ax.set_yticks([0], ['Class-associated gain'])
    ax.set_xlabel('Held-out routing gain, nats per token pair')
    ax.set_title('GPT source dataset: 509 clean questions / 496 families', loc='left', fontsize=11)
    fig.text(.02, .02, '95% family-clustered interval; native observational result. Qwen3.6 transfer is untested.', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(out / 'clean_b1.pdf')
    fig.savefig(out / 'clean_b1.png', dpi=200)
    plt.close(fig)
    text = ('# Discovery evidence snapshot v2\n\n'
        'The routing pilot demonstrates expert-selection changes on Qwen3.6, but its four starts had zero jointly accepted full-prefix unchecked-candidate enrollments. It therefore does not estimate successful initiation of verification. The target-only rating counts are omitted from the efficacy figure.\n\n'
        'Whole-continuation fractions are conditional on emitted reasoning and can change with early closure; the fixed 256-token denominator retains that exposure difference. The first-token panel holds the input prefix fixed. All plotted pilot quantities are descriptive; four selected families do not support a population claim.\n\n'
        'The independent clean B1 result concerns the GPT source routing dataset. Its interval uses frozen duplicate families; it is observational and does not establish semantic causality or model transfer.\n\n'
        'Native-only replay shows numerical route variability even without edited requests. The failed counterfactual score alignment remains failed. Full-prefix eligibility, a prospective score-API calibration and a separate batch-invariant backend qualification address these concrete measurement/enrollment issues before further semantic action validation.\n')
    (out / 'README.md').write_text(text)
    files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    body = {'schema': 'discovery-evidence-frozen-v2', 'input_binding_sha256': bundle_sha,
        'job_id': os.environ['SLURM_JOB_ID'], 'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'files': files, 'status': 'COMPLETE_SAVED_RESULT_FIGURES_AND_LEDGER'}
    frozen_path.write_text(json.dumps({**body, 'sha256': digest(body)}, indent=1) + '\n')
    pointer = {'schema': 'discovery-evidence-pointer-v2', 'output': str(out), 'frozen': str(frozen_path),
        'input_binding_sha256': bundle_sha, 'job_id': os.environ['SLURM_JOB_ID']}
    (DOC / 'DISCOVERY_EVIDENCE_SNAPSHOT_v2.json').write_text(json.dumps(pointer, indent=1) + '\n')
    print(json.dumps(pointer))


if __name__ == '__main__':
    main()
