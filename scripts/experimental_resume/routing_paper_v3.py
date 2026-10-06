"""Publish saved routing estimates and attach CPU reporting to exact Slurm jobs.

No new model fits, selection, grading, or pooling across experimental stages.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

import dispatch_overnight_readers_v1 as shared

STAGES = ('LEGACY', 'A', 'B', 'C', 'FRESH')


def locations(stage):
    if stage == 'LEGACY':
        return shared.DOC / 'MECHANISM_1024_SEMANTIC_ITT_v1.json', None
    version = 'v1' if stage in ('A', 'B') else 'v2'
    prefix = 'OVERNIGHT_FRESH_COMPARISON' if stage == 'FRESH' else 'OVERNIGHT_DISCOVERY_' + stage
    manifest = shared.base.sealed(shared.DOC / f'{prefix}_MANIFEST_{version}.json')
    tag = manifest['sha256'][:16]
    receipts = shared.DOC / (('overnight-submissions-' if version == 'v1' else 'overnight-submissions-v2-') + tag)
    return shared.DOC / f'{prefix}_ANALYSIS_{version}' / 'ANALYSIS.json', receipts


def verified_job(job_id):
    raw = subprocess.run(['sacct', '-X', '-n', '-P', '-j', job_id, '-o',
                          'JobID,State,ExitCode,Elapsed,End'], check=True,
                         capture_output=True, text=True).stdout
    rows = [r.split('|') for r in raw.splitlines() if r.strip()]
    rows = [r for r in rows if r[0] == job_id]
    if len(rows) != 1 or rows[0][1:3] != ['COMPLETED', '0:0']:
        raise ValueError('analysis Slurm completion does not verify: ' + raw)
    return dict(zip(('job_id', 'state', 'exit_code', 'elapsed', 'end'), rows[0]))


def evidence(stage, source, result):
    primary = result['primary_itt'] if stage == 'LEGACY' else result['primary']
    population = (f"{result['assigned']} assigned requests; "
                  f"{result.get('accepted_families', result.get('families'))} families; stage {stage}")
    rows, claims = [], []
    contrasts = [(r, primary['multiplicity']) for r in primary['contrasts']]
    secondary = result.get('emitted_token_secondary')
    if secondary:
        contrasts.extend((r, secondary['multiplicity']) for r in secondary['contrasts'])
    for row, multiplicity in contrasts:
        endpoint = row.get('endpoint', 'both_positive')
        interval = row['simultaneous_ci95']
        rows.append({'endpoint': endpoint, 'scope': row['scope'], 'arm': row['arm'], 'reference': row['reference'],
                     'estimate': row['estimate'], 'lower': interval[0] if interval else None,
                     'upper': interval[1] if interval else None, 'families': row['families']})
        claims.append({'id': 'routing-v3:' + ':'.join((stage, endpoint, row['scope'], row['arm'], row['reference'])),
                       'population': population + '; contrast scope ' + row['scope'],
                       'intervention': row['arm'] + ' versus ' + row['reference'],
                       'estimate': row['estimate'], 'uncertainty': interval,
                       'multiplicity_family': f"stage {stage}, {multiplicity} frozen endpoint contrasts; "
                                              'family-clustered Bonferroni percentile bootstrap approximation',
                       'status': 'ESTIMATE_AVAILABLE' if interval is not None else 'INSUFFICIENT_PRECISION',
                       'source': str(source), 'source_sha256': result['sha256'],
                       'endpoint': endpoint,
                       'interpretation': ('Emitted tokens within the fixed continuation horizon. '
                                          if endpoint == 'emitted_tokens' else
                                         'Substantive transition after the trigger within the fixed continuation horizon; '
                                         'two arm-blind LLM readers. Unscored assignments remain in the endpoint. '
                                         ) + 'This does not measure original-prompt final-answer accuracy or 16k utility.'})
    return rows, claims


def figures(directory, rows, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    valid = [r for r in rows if r['endpoint'] == 'both_positive' and
             r['estimate'] is not None and r['lower'] is not None]
    if valid:
        fig, axis = plt.subplots(figsize=(10, max(4, .34 * len(valid) + 1)))
        for index, row in enumerate(valid):
            axis.plot([100 * row['lower'], 100 * row['upper']], [index, index], color='#456a8c')
            axis.plot(100 * row['estimate'], index, 'o', color='#132f4c', markersize=4)
        axis.set_yticks(range(len(valid)), [r['scope'] + ': ' + r['arm'] + ' − ' + r['reference'] for r in valid])
        axis.axvline(0, color='#999999', linewidth=.8)
        axis.set_xlabel('Semantic transition difference (percentage points), simultaneous 95% intervals')
        axis.invert_yaxis(); axis.grid(axis='x', alpha=.15)
        fig.tight_layout()
        for extension in ('pdf', 'svg'):
            fig.savefig(directory / ('semantic_effects.' + extension))
        plt.close(fig)
    summary = result.get('arm_summary', {})
    if summary:
        names = list(summary)
        fig, axis = plt.subplots(figsize=(max(7, .65 * len(names)), 4))
        rates = [summary[a]['reader_pair_valid'] / summary[a]['assigned'] for a in names]
        axis.bar(names, rates, color='#456a8c')
        axis.set_ylim(0, 1); axis.set_ylabel('Fraction with two valid stopped ratings')
        axis.tick_params(axis='x', labelrotation=30); fig.tight_layout()
        fig.savefig(directory / 'reader_coverage.pdf'); plt.close(fig)


def build(stage):
    source, receipts = locations(stage)
    result = shared.base.sealed(source)
    job = '59343015' if stage == 'LEGACY' else shared.base.sealed(receipts / 'analysis.json')['job_id']
    record = verified_job(job)
    rows, claims = evidence(stage, source, result)
    out = shared.DOC / ('routing-paper-v3-' + stage.lower() + '-' + result['sha256'][:16])
    out.mkdir(exist_ok=True)
    if (out / 'SUMMARY.json').exists():
        previous = shared.base.sealed(out / 'SUMMARY.json')
        if previous['driver_sha256'] != shared.base.file_sha(__file__):
            raise ValueError('report path is already bound to different code')
        for name, digest in previous['artifacts'].items():
            if shared.base.file_sha(out / name) != digest:
                raise ValueError('completed paper artifact changed')
        print(json.dumps({'status': 'ALREADY_COMPLETE', 'output': str(out)})); return
    fields = ('endpoint', 'scope', 'arm', 'reference', 'estimate', 'lower', 'upper', 'families')
    with (out / 'semantic_contrasts.tsv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
    inherited = shared.DOC / 'CLAIM_LEDGER_v2.json'
    previous_claims = json.loads(inherited.read_text())
    ids = [r['id'] for r in previous_claims + claims]
    if len(ids) != len(set(ids)):
        raise ValueError('claim ID collision')
    shared.save(out / 'CLAIM_LEDGER.json', {'schema': 'routing-paper-claim-ledger-v3',
                 'inherited_ledger_file_sha256': shared.base.file_sha(inherited),
                 'claims': previous_claims + claims})
    shared.save(out / 'MEASUREMENT.json', {'schema': 'routing-paper-measurement-v3',
                 'stage': stage, 'source_sha256': result['sha256'],
                 'arm_summary': result.get('arm_summary'),
                 'available_sensitivity_fields': [k for k in result if 'sensitivity' in k or 'bound' in k],
                 'accuracy_change': {'status': 'NOT_MEASURED_BY_CONTINUATION_STUDY'},
                 'utility_16k': {'status': 'SEPARATE_ORIGINAL_PROMPT_STUDY_PENDING'},
                 'gate_weight_trajectory': {'status': 'NOT_SAVED; ROUTE_ARRAYS_CONTAIN_EXPERT_IDS'},
                 'historical_confirm_exposure': 'Previously exposed; retain original disclosure',
                 'population_status': result.get('stage_label', stage),
                 'no_new_fits': True})
    figures(out, rows, result)
    artifacts = {p.name: shared.base.file_sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    summary = shared.save(out / 'SUMMARY.json', {'schema': 'routing-paper-snapshot-v3',
                 'stage': stage, 'source': str(source), 'source_sha256': result['sha256'],
                 'driver_sha256': shared.base.file_sha(__file__), 'analysis_slurm': record,
                 'report_slurm_job': os.environ['SLURM_JOB_ID'], 'artifacts': artifacts,
                 'new_claims': len(claims), 'status': 'COMPLETE_SAVED_RESULT_ADDENDUM'})
    print(json.dumps({'output': str(out), 'sha256': summary['sha256'], 'status': summary['status']}))


def chain(stage, phase):
    _, receipts = locations(stage)
    if receipts is None:
        raise ValueError('legacy report is directly submitted')
    if phase == 'attach':
        dependency = shared.base.sealed(receipts / 'GENERATION_CHAIN.json')['reader_dispatch_job']
        next_phase = 'dispatch'
    else:
        dependency = shared.base.sealed(receipts / 'analysis.json')['job_id']
        next_phase = 'build'
    directory = receipts / 'paper-v3'
    directory.mkdir(exist_ok=True)
    wrapper = Path(__file__).with_suffix('.sbatch')
    subprocess.run(['bash', '-n', str(wrapper)], check=True)
    binding = {'stage': stage, 'phase': next_phase, 'parent_job': dependency,
               'files': {str(p): shared.base.file_sha(p) for p in (Path(__file__), wrapper)}}
    shared.submit(directory, next_phase, ['--dependency=afterok:' + dependency,
                  '--job-name=st-paper-' + stage + '-' + next_phase, str(wrapper), stage, next_phase],
                  os.environ.copy(), binding)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('phase', choices=('build', 'attach', 'dispatch'))
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'lrd_all_viz':
        raise RuntimeError('paper work requires authorized CPU Slurm')
    build(args.stage) if args.phase == 'build' else chain(args.stage, args.phase)


if __name__ == '__main__':
    main()
