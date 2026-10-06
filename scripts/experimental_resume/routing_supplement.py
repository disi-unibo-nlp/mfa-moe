"""Saved-result observational routing profiles and expert-use paper supplement."""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
from pathlib import Path

R = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
DOC = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/experimental-resume-v1')
SOURCES = {
    'gpt': (R/'reasoning-kinematics/rk2/out/gpt/profiles_clean.json',
            R/'reasoning-kinematics/rk2/out/gpt/experts_nulls.json'),
    'qwen36': (R/'reasoning-kinematics/rk2/out/qwen36/profiles_saved.json',
               R/'reasoning-kinematics/rk2/out/qwen36/experts_nulls.json'),
}
CLASSES = ('Read', 'Analyze', 'Plan', 'Implement', 'Explore', 'Verify', 'Monitor')
EVENTS = ('->Explore', '->Verify')


def load(path):
    return json.loads(path.read_text())


def collect():
    profiles = {}
    expert_rows = []
    for model, (path, experts_path) in SOURCES.items():
        profile, experts = load(path), load(experts_path)
        if (profile['name'] != model or experts['name'] != model or
                experts['exposure']['confirm'] != 0 or
                model == 'gpt' and profile['population'] != 'clean'):
            raise ValueError('no-confirm native observational profile required: ' + model)
        grids = {metric: profile['definition']['lags_'+metric] for metric in ('V', 'a')}
        events = {}
        for event in EVENTS:
            cell = profile['sets'][event]
            events[event] = {'n_events': cell['n_events'], 'n_matched': cell['n_matched'],
                             'n_questions': cell['n_questions'],
                             'velocity_JSD2_bits': cell['V']['excess'],
                             'acceleration_JSD2_bits_per_32_tokens': cell['a']['excess'],
                             'acceleration_excess_0_64': cell['accel_excess_0_64']}
            for metric, key in (('V', 'velocity_JSD2_bits'), ('a', 'acceleration_JSD2_bits_per_32_tokens')):
                if len(events[event][key]['point']) != len(grids[metric]):
                    raise ValueError('event profile and token grid length mismatch')
        profiles[model] = {'population': profile['population'], 'tag': profile['tag'],
                           'confirm_exposure': experts['exposure']['confirm'],
                           'n_questions': profile['counts']['questions'],
                           'grids_tokens_relative_to_event': grids, 'events': events,
                           'profile_source': str(path), 'expert_source': str(experts_path)}
        rare = experts['rarefied']['class_minus_pooled']
        baseline = experts['rarefied']['pooled_effective_experts']['layer_mean']
        if experts['classes'] != list(CLASSES):
            raise ValueError('seven-class expert summary changed')
        for index, label in enumerate(CLASSES):
            expert_rows.append({'model': model, 'class': label, 'population': profile['tag'],
                'confirm_exposure': 0, 'n_attempts': experts['n_attempts'],
                'rarefied_pooled_effective_experts': baseline,
                'class_minus_pooled_effective_experts': rare['layer_mean'][index],
                'nominal_ci_lo': rare['ci_lo'][index], 'nominal_ci_hi': rare['ci_hi'][index],
                'contrast_as_fraction_of_pooled': rare['layer_mean'][index]/baseline,
                'fraction_ci_lo': rare['ci_lo'][index]/baseline,
                'fraction_ci_hi': rare['ci_hi'][index]/baseline})
    return profiles, expert_rows


def figures(out, profiles, expert_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True, sharex=True)
    color = {'->Explore': '#1a728a', '->Verify': '#c4652b'}
    for row, model in enumerate(SOURCES):
        for column, (metric, key) in enumerate((('V', 'velocity_JSD2_bits'),
                                                ('a', 'acceleration_JSD2_bits_per_32_tokens'))):
            ax = axes[row, column]
            grid = profiles[model]['grids_tokens_relative_to_event'][metric]
            for event in EVENTS:
                cell = profiles[model]['events'][event][key]
                ax.plot(grid, cell['point'], label=event, color=color[event])
                ax.fill_between(grid, cell['ci_lo'], cell['ci_hi'], color=color[event], alpha=.17)
            ax.axhline(0, color='gray', lw=.8)
            ax.axvline(0, color='gray', lw=.8, ls=':')
            ax.set_title(model + (' velocity' if metric == 'V' else ' acceleration'))
            ax.set_xlabel('Tokens relative to transition')
            ax.set_ylabel('Event minus matched control, JSD₂ bits' if metric == 'V'
                          else 'Change in JSD₂ bits per 32 tokens')
    axes[0, 0].legend()
    fig.suptitle('Native event-locked routing profiles; dev+tune only; observational nominal intervals')
    fig.savefig(out/'routing-velocity-acceleration.pdf')
    fig.savefig(out/'routing-velocity-acceleration.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True, sharey=True)
    for ax, model in zip(axes, SOURCES):
        rows = [row for row in expert_rows if row['model'] == model]
        y = list(range(len(rows)))
        point = [row['contrast_as_fraction_of_pooled'] for row in rows]
        ax.errorbar(point, y, xerr=[
            [v-row['fraction_ci_lo'] for v, row in zip(point, rows)],
            [row['fraction_ci_hi']-v for v, row in zip(point, rows)]], fmt='o', capsize=2)
        ax.axvline(0, color='gray', lw=.8)
        ax.set_yticks(y, CLASSES)
        ax.set_xlabel('Class minus pooled effective experts / pooled')
        ax.set_title(model + ' rarefied expert use')
    fig.suptitle('Native class-conditional expert use; no confirm questions; descriptive nominal intervals')
    fig.savefig(out/'expert-use-profiles.pdf')
    fig.savefig(out/'expert-use-profiles.png', dpi=160)
    plt.close(fig)


def run():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('paper figures require CPU Slurm')
    profiles, expert_rows = collect()
    now = datetime.datetime.now(datetime.timezone.utc)
    out = R/'paper/resume-v1'/('routing-supplement-'+now.strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True, exist_ok=False)
    sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
               for pair in SOURCES.values() for path in pair}
    frozen = {'schema': 'routing-profile-paper-supplement-v1',
              'sources': sources, 'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'job_id': os.environ['SLURM_JOB_ID'], 'no_new_fits': True,
              'scope': 'GPT clean dev+tune and Qwen3.6 dev+tune only; native observational profiles'}
    (out/'FROZEN.json').write_text(json.dumps(frozen, indent=1)+'\n')
    (out/'routing-profiles.json').write_text(json.dumps(profiles, indent=1)+'\n')
    with (out/'expert-use-profiles.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(expert_rows[0]))
        writer.writeheader(); writer.writerows(expert_rows)
    figures(out, profiles, expert_rows)
    (out/'README.md').write_text('Native event-locked routing velocity, acceleration and rarefied expert-use profiles. '
        'All estimates are descriptive and nominal on no-confirm dev+tune populations. '
        'Transitions associated with routing changes do not establish causal relaxation, semantic control or an optimal sequence.\n')
    (DOC/'ROUTING_PROFILE_SUPPLEMENT.json').write_text(json.dumps({'path': str(out),
        'job_id': os.environ['SLURM_JOB_ID']}, indent=1)+'\n')
    print(json.dumps({'path': str(out), 'models': list(profiles), 'expert_rows': len(expert_rows)}))


if __name__ == '__main__':
    run()
