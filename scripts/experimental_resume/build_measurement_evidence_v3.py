"""Bound saved measurement results to standalone paper figures; no new fits."""
from __future__ import annotations

import csv
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
    'start_agreement': DOC / 'FULL_PREFIX_NATIVE_VETO_AGREEMENT_v1.json',
    'b4': S / 'runs/resume-v1/r3e-cv-clean-v1/full.json',
    'score_batched': BASE / 'score-api-calibration-b4aa1c878c25fe32/QUALIFICATION.json',
    'score_serial': BASE / 'score-api-serial-e7554bd7efceb410/QUALIFICATION.json',
}


def csv_write(path, rows):
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('paper rendering requires an allocated CPU Slurm step')
    values = {key: sealed(path) for key, path in INPUTS.items()}
    if values['b4']['status'] != 'PASS' or values['b4']['stage'] != 'full':
        raise ValueError('B4 input must be the completed full result')
    if any(values[key]['pass'] or values[key]['completed'] != 80
           for key in ('score_batched', 'score_serial')):
        raise ValueError('expected preserved complete score-qualification failures')
    binding = {'schema': 'measurement-evidence-binding-v3',
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'helper_sha256': hashlib.sha256((REPO / 'src/moe_exp/routing_control/counterfactual.py').read_bytes()).hexdigest(),
        'sources': {key: {'path': str(INPUTS[key]), 'sha256': value['sha256']}
                    for key, value in values.items()}}
    bundle = digest(binding)
    out = R / 'paper/measurement-evidence-v3' / bundle[:16]
    out.mkdir(parents=True, exist_ok=True)
    frozen_path = out / 'FROZEN.json'
    if frozen_path.exists():
        frozen = sealed(frozen_path)
        if frozen['input_binding_sha256'] != bundle or any(
                hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in frozen['files'].items()):
            raise ValueError('completed paper bundle changed')
        print(json.dumps({'status': 'VERIFIED_EXISTING', 'output': str(out)}))
        return
    (out / 'INPUT_BINDING.json').write_text(json.dumps({**binding, 'sha256': bundle}, indent=2) + '\n')
    counts = values['start_agreement']['counts']
    transitions = ('candidate_to_verify', 'approach_to_commit', 'failed_check_to_revise')
    start_rows = []
    for transition in transitions:
        n = counts[transition + '|fire|rows']
        for model in ('qwen', 'native'):
            row = {'transition': transition, 'model': model, 'detector_fire_windows': n}
            row.update({status: counts.get(transition + '|fire|' + model + '_' + status, 0)
                        for status in ('accepted', 'rejected', 'split', 'unresolved')})
            if sum(row[k] for k in ('accepted', 'rejected', 'split', 'unresolved')) != n:
                raise ValueError('rating statuses do not partition assignments')
            start_rows.append(row)
    csv_write(out / 'full_prefix_start_audit.csv', start_rows)
    b4 = values['b4']
    comparisons = b4['evaluation']['comparisons']
    comparison_order = ('B4_primary', 'secondary_PAlex_only', 'secondary_PA_perclass',
                        'sensitivity_PA_only', 'ablation_marginal_block_vs_controls')
    if set(comparisons) != set(comparison_order):
        raise ValueError('saved prediction comparison family changed')
    prediction_rows = [{'comparison': key, 'gain_nats_per_scored_question': row['gain'],
        'ci95_lower': row['ci95'][0], 'ci95_upper': row['ci95'][1],
        'simultaneous_ci95_lower': row['simultaneous_ci95'][0],
        'simultaneous_ci95_upper': row['simultaneous_ci95'][1]}
        for key in comparison_order for row in (comparisons[key],)]
    csv_write(out / 'clean_b4_prediction.csv', prediction_rows)
    score_rows = []
    for profile in ('score_batched', 'score_serial'):
        for metric, check in values[profile]['numerical_calibration']['checks'].items():
            score_rows.append({'profile': profile, 'metric': metric, **check})
    csv_write(out / 'score_qualification.csv', score_rows)
    claims = [
        {'id': 'full-prefix-discovery-start-measurement',
         'population': '372 stratified discovery windows in the frozen 48-family pool; fire strata shown',
         'intervention': 'none; two draws per LLM, model-specific rubrics', 'estimate': start_rows,
         'uncertainty': 'descriptive counts; correlated draws, differing rubrics and parse noncoverage',
         'multiplicity_family': 'measurement audit, no confirmatory tests', 'status': 'LLM_AUDIT',
         'interpretation': 'reader approval is not human truth or a semantic control effect'},
        {'id': 'clean-b4-registered-result', 'population': {**b4['population'],
            'scored_families': b4['evaluation']['family_count']},
         'intervention': 'observational nested-CV correctness prediction; training-fold-only features',
         'estimate': b4['primary']['gain_nats_per_attempt'],
         'uncertainty': {'family_ci95': b4['primary']['family_ci95'],
            'within_B4_five_comparison_ci95': b4['primary']['simultaneous_ci95']},
         'multiplicity_family': {'within_B4': 5, 'registered_cross_endpoint_Holm': b4['primary']['holm_family'],
            'cross_endpoint_adjustment_status': 'separate obligation; not supplied by within-B4 max-t'},
         'status': 'NO_SUPPORTED_PREDICTIVE_IMPROVEMENT',
         'interpretation': 'similar observed losses do not establish equivalence; remedies remain exploratory'},
        {'id': 'score-extraction-qualification', 'population': 'four engineering pilot prefixes, four native positions, 80 requests per profile',
         'intervention': 'designated-token vs full-vocabulary native scoring; batched and serial/eager profiles',
         'estimate': score_rows, 'uncertainty': 'repeat-baseline calibration; no semantic interval',
         'multiplicity_family': 'both mean and p99 numerical gates required', 'status': 'FAILED_BOTH_PROFILES',
         'interpretation': 'counterfactual loss ranking unqualified; does not establish inability to steer'},
    ]
    ledger = {'schema': 'measurement-claim-ledger-v3', 'input_binding_sha256': bundle, 'claims': claims}
    (out / 'CLAIM_LEDGER.json').write_text(json.dumps({**ledger, 'sha256': digest(ledger)}, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.1))
    statuses = ('accepted', 'rejected', 'split', 'unresolved')
    colors = ('#3b80ad', '#b9bec4', '#dc8a36', '#726487')
    for i, (ax, transition) in enumerate(zip(axes, transitions)):
        rows = [r for r in start_rows if r['transition'] == transition]
        bottom = np.zeros(2)
        for status, color in zip(statuses, colors):
            heights = np.array([r[status] for r in rows])
            ax.bar([0, 1], heights, bottom=bottom, label=status, color=color)
            bottom += heights
        ax.set_xticks([0, 1], ['Qwen3.8', 'Native Qwen3.6'])
        ax.set_title(('Candidate → verification', 'Approach → commitment', 'Failed check → revision')[i], fontsize=10)
        ax.set_ylabel('Detector-fire windows')
        ax.set_ylim(0, rows[0]['detector_fire_windows'] * 1.13)
        for x, row in enumerate(rows):
            ax.text(x, bottom[x] + .025 * bottom[x], str(row['accepted']) + ' accepted', ha='center', fontsize=9)
    axes[0].legend(fontsize=8)
    fig.suptitle('Full-prefix start audit: reader agreement and missing coverage', fontsize=12)
    fig.text(.01, .01, 'Stratified discovery LLM ratings; differing rubrics. Approval is not semantic truth or an intervention result.', fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .94))
    fig.savefig(out / 'full_prefix_start_audit.pdf')
    fig.savefig(out / 'full_prefix_start_audit.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = ('B4 vs C+M', 'PAlex vs C+M', 'Per-class PA vs C+M', 'PA vs C+M', 'Marginal block vs C')
    for i, row in enumerate(prediction_rows):
        point = row['gain_nats_per_scored_question']
        lo, hi = row['simultaneous_ci95_lower'], row['simultaneous_ci95_upper']
        ax.errorbar(point, i, xerr=np.array([[point-lo], [hi-point]]), fmt='o', color='#3b80ad', capsize=4)
    ax.axvline(0, color='#969da5', lw=.8)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel('Predictive log-loss gain, nats per scored question (positive is better)')
    ax.set_title('Clean correctness prediction: 508 scored questions / 495 scored families', loc='left', fontsize=11)
    fig.text(.01, .015, '95% family-clustered simultaneous intervals for five contrasts. Cross-endpoint Holm is separate. No causal claim.', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(out / 'clean_b4_prediction.pdf')
    fig.savefig(out / 'clean_b4_prediction.png', dpi=180)
    plt.close(fig)
    (out / 'README.md').write_text(
        '# Saved measurement evidence v3\n\n'
        'These figures summarize frozen discovery measurement and clean predictive results. '
        'They contain no semantic steering efficacy estimate.\n\n'
        'The CSVs retain source values. The claim ledger records populations, limits and '
        'multiplicity obligations. Old failures and protocols remain intact.\n')
    files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
    frozen = {'schema': 'measurement-evidence-frozen-v3', 'input_binding_sha256': bundle,
              'job_id': os.environ['SLURM_JOB_ID'], 'files': files}
    frozen_path.write_text(json.dumps({**frozen, 'sha256': digest(frozen)}, indent=2) + '\n')
    pointer = {'schema': 'measurement-evidence-pointer-v3', 'output': str(out),
               'frozen': str(frozen_path), 'input_binding_sha256': bundle, 'job_id': os.environ['SLURM_JOB_ID']}
    (DOC / 'MEASUREMENT_EVIDENCE_SNAPSHOT_v3.json').write_text(json.dumps(pointer, indent=2) + '\n')
    print(json.dumps({'status': 'COMPLETE', **pointer}))


if __name__ == '__main__':
    main()
