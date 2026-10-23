"""Read-only source aggregation for the CL evidence review; no scientific refits.

    python -m routing.benchmark_inventory OUT_DIR

Writes the CSV tables, receipts and the TeX table into OUT_DIR only (the published copy is
report/model-analysis-review-20261008).
Saved question-bootstrap intervals are copied, never recomputed. No inference,
raw routing tensors, Slurm, or checkpoint weight payloads are read.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]) if __name__ == '__main__' else None
TEX = OUT
BASE = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
ANALYSIS = BASE / 'v3_analysis/results-r2'
OUTCOMES = BASE / 'campaign-v3/outcomes/outcomes_v3-99bf8d59506e2893.parquet'
SAVED = BASE / 'results-v3/outcomes-99bf8d59506e2893'
MODELS = ['gpt', 'nemotron', 'qwen35', 'qwen36', 'gemma', 'glm', 'qwen330b']
LABELS = {'gpt': 'GPT-OSS', 'nemotron': 'Nemotron', 'qwen35': 'Qwen3.5',
          'qwen36': 'Qwen3.6', 'gemma': 'Gemma', 'glm': 'GLM', 'qwen330b': 'Qwen3-30B'}
DATASETS = ['aime24', 'aime25', 'amc23', 'math500', 'minerva', 'olympiad']
SOURCES = {}


def source(path):
    path = Path(path)
    raw = path.read_bytes()
    SOURCES[str(path)] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return path


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def basic(frame):
    row = dict(attempts=len(frame), questions=frame['question'].nunique(),
               capped=int(frame['termination'].eq('length').sum()),
               completion_tokens_sum=int(frame['completion_tokens'].sum()),
               completion_tokens_median=float(frame['completion_tokens'].median()),
               completion_tokens_mean=float(frame['completion_tokens'].mean()))
    for column, label in [('strict_correct', 'strict'), ('adjudicated_correct', 'adjudicated')]:
        values = frame[column]
        n = int(values.notna().sum())
        yes = int(values.fillna(0).sum())
        row.update({label + '_scored': n, label + '_correct': yes,
                    label + '_unknown': len(frame) - n, label + '_accuracy': yes / n,
                    label + '_all_assignment_lower': yes / len(frame),
                    label + '_all_assignment_upper': (yes + len(frame) - n) / len(frame)})
    return row


def main():
    TEX.mkdir(parents=True, exist_ok=True)
    source(__file__)
    source(BASE / 'outcomes_analysis.py')
    source(BASE / 'campaign-v3/acceptance/j1.json')
    source(OUTCOMES.with_suffix('.manifest.json'))
    source(SAVED / 'provenance.json')
    source(ROOT / 'src/generation/model_profiles.py')
    table = pd.read_parquet(source(OUTCOMES))
    assert SOURCES[str(OUTCOMES)]['sha256'] == '99bf8d59506e2893218fd070d264d1fca4a1e4330052c11c347b727b3802e9ee'
    table['model'] = table['model'].replace({'qwen35gptq': 'qwen35'})
    table['question'] = table['dataset'] + '|' + table['source_problem_id'].fillna(table['problem_id'])
    for column in ['strict_correct', 'adjudicated_correct']:
        table[column] = table[column].astype(float)
    saved = pd.read_csv(source(SAVED / 'accuracy.csv')).set_index(['model', 'dataset', 'corpus'])
    cells = []
    models = []
    questions = {}
    for (model, dataset, corpus), frame in table[table.corpus != 'continuation'].groupby(['model', 'dataset', 'corpus']):
        row = dict(model=model, dataset=dataset, corpus=corpus, **basic(frame))
        old = saved.loc[(model, dataset, corpus)]
        assert row['attempts'] == old.attempts and row['questions'] == old.questions
        assert abs(row['adjudicated_accuracy'] - old.adjudicated) < 1e-12
        assert abs(row['strict_accuracy'] - old.strict) < 1e-12
        assert abs(row['capped'] / row['attempts'] - old.capped_rate) < 1e-12
        assert row['completion_tokens_median'] == old.median_tokens
        row['adjudicated_ci_low'] = old.adjudicated_ci_low
        row['adjudicated_ci_high'] = old.adjudicated_ci_high
        cells.append(row)
    for (model, corpus), frame in table.groupby(['model', 'corpus']):
        models.append(dict(model=model, corpus=corpus, **basic(frame)))
        if corpus != 'continuation':
            questions[(model, corpus)] = set(frame.question)
    assert len(cells) == 84
    assert all(len(q) == 1547 for q in questions.values())
    assert len(set.intersection(*questions.values())) == 1547
    cells = pd.DataFrame(cells)
    model_table = pd.DataFrame(models)
    cells.to_csv(OUT / 'benchmark_cells.csv', index=False)
    model_table.to_csv(OUT / 'benchmark_models.csv', index=False)
    for name in ['budget', 'continuation', 'errors', 'length_gap']:
        source(SAVED / (name + '.csv'))

    readout = json.loads(source(BASE / 'forum/tests/r3_context/readout-results.v3.json').read_text())
    readout_rows = []
    for model in MODELS:
        result = readout['models'][model]['judge_context']
        row = dict(model=model, status=result['status'], questions=result['n_questions'],
                   families=result['n_families'], sentences=result['n_sentences'])
        for key in ['routing', 'routing_excess_over_lexical_null']:
            row.update({key + '_delta_logloss': result[key]['P'],
                        key + '_simultaneous_ci7_low': result[key]['simultaneous_ci7'][0],
                        key + '_simultaneous_ci7_high': result[key]['simultaneous_ci7'][1]})
        readout_rows.append(row)
    pd.DataFrame(readout_rows).to_csv(OUT / 'routing_readout.csv', index=False)

    anticipation = json.loads(source(BASE / 'forum/tests/r3_context/anticipation-results.json').read_text())
    anticipation_rows = []
    for model, modes in anticipation['models'].items():
        for mode, result in modes.items():
            if not isinstance(result, dict) or 'routing' not in result:
                continue
            for feature, metric in result['routing'].items():
                anticipation_rows.append(dict(model=model, mode=mode, feature=feature,
                    status=result.get('status'), questions=result.get('questions'), pairs=result.get('pairs'),
                    delta_logloss=metric['P'], simultaneous_ci6_low=metric['simultaneous_ci6'][0],
                    simultaneous_ci6_high=metric['simultaneous_ci6'][1]))
    pd.DataFrame(anticipation_rows).to_csv(OUT / 'anticipation.csv', index=False)

    cohort_rows = []
    expert_rows = []
    expert_associations = []
    cohort_questions = {'A': [], 'B': []}
    for model in MODELS:
        for cohort in ['A', 'B']:
            p = ANALYSIS / model / cohort
            columns = ['attempt_id','question','correct','capped','completion_tokens','n_reasoning']
            if cohort == 'A':
                columns += ['labelled_units','labelled_tokens']
            frame = pd.read_parquet(source(p / 'attempts.parquet'), columns=columns)
            cohort_rows.append(dict(model=model, cohort=cohort, attempts=len(frame),
                questions=frame.question.nunique(), scored=int(frame.correct.notna().sum()),
                correct=int(frame.correct.fillna(0).sum()), capped=int(frame.capped.sum()),
                completion_tokens_median=float(frame.completion_tokens.median()),
                reasoning_tokens_sum=int(frame.n_reasoning.sum()),
                labelled_units_sum=int(frame.labelled_units.sum()) if cohort == 'A' else None,
                labelled_tokens_sum=int(frame.labelled_tokens.sum()) if cohort == 'A' else None))
            cohort_questions[cohort].append(set(frame.question))
        p = ANALYSIS / model / 'A'
        provenance = json.loads(source(p / 'provenance.json').read_text())
        selected = pd.read_csv(source(p / 'E_class_experts.csv'))
        spec = provenance['spec']
        expert_rows.append(dict(model=model, routed_layers=len(spec['layers']), native_experts_per_layer=spec['num_experts'],
            native_k=spec['top_k'], selected_class_layer_expert_rows=len(selected),
            distinct_selected_layer_experts=len(selected[['layer','expert']].drop_duplicates()),
            source=str(p / 'E_class_experts.csv'), class_counts_path=str(p / 'class_counts.npz')))
        use = pd.read_csv(source(p / 'E_expert_usage_associations.csv'))
        use = use[(use.scope == 'pooled') & (use.layer == 'mean') & (use.adjustment == 'lendiff_frac')].copy()
        use.insert(0, 'model', model)
        expert_associations.append(use)
    pd.DataFrame(cohort_rows).to_csv(OUT / 'cohort_populations.csv', index=False)
    pd.DataFrame(expert_rows).to_csv(OUT / 'expert_profile_inventory.csv', index=False)
    pd.concat(expert_associations).to_csv(OUT / 'expert_usage_correctness_saved.csv', index=False)
    source(ROOT / 'report/experimental-resume-v1/R3E_B4_PRECISION_READOUT_v1.md')
    source(ANALYSIS / 'cross-4/F_A_across_meta.csv')
    source(ANALYSIS / 'cross-4/F_summary.md')

    card = cells[cells.corpus == 'card-v3'].set_index(['model', 'dataset'])
    card_models = model_table[model_table.corpus == 'card-v3'].set_index('model')
    unknown_total = int(card_models.adjudicated_unknown.sum())
    lines = [r'% Generated by report/model-analysis-review-20261008/build_inventory.py; existing scores only.',
        r'\begin{table}[t]', r'\centering', r'\small',
        r'\caption{Saved seven-model card-profile mathematical accuracy (\%). Each model has 4,647 assigned attempts on the same 1,547 problems (32,529 attempts in total). AIME24/25 and AMC23 use 32 samples per problem; the other datasets use one. Entries are adjudicated accuracy among scored attempts. $U$ counts unresolved attempts per model, retained separately. These checkpoints use different native decoding profiles and output limits; this table is a benchmark description, not an equal-budget architecture comparison.}',
        r'\label{tab:seven-model-saved-math}', r'\resizebox{\linewidth}{!}{%',
        r'\begin{tabular}{lrrrrrrr}', r'\toprule',
        r'Model & AIME24 & AIME25 & AMC23 & MATH500 & Minerva & Olympiad & $U$ \\',
        r'Problems & 30 & 30 & 40 & 500 & 272 & 675 & \\', r'\midrule']
    for model in MODELS:
        values = [f'{100 * card.loc[(model,d), "adjudicated_accuracy"]:.1f}' for d in DATASETS]
        lines.append(' & '.join([LABELS[model]] + values + [str(int(card_models.loc[model, 'adjudicated_unknown']))]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}}',
        r'\par\smallskip\parbox{\linewidth}{\footnotesize The source inventory retains strict scores, all denominators, completion-token costs, caps, and the original question-bootstrap 95\% intervals. Unresolved outcomes total ' + str(unknown_total) + r'; repeated samples are not independent questions. The historical 32k campaign is a separate 32,529-attempt population.}',
        r'\end{table}', '']
    (TEX / 'benchmark_readout_tables.tex').write_text('\n'.join(lines))

    inventory = dict(schema='model-analysis-review-v1', generated_utc_date='2026-10-08',
        source_sidecar_sha256=SOURCES[str(OUTCOMES)]['sha256'], source_rows=len(table),
        corpus_rows={str(k):int(v) for k,v in table.corpus.value_counts().items()},
        shared_benchmark_questions_all_models_and_both_corpora=1547,
        cohorts={c:dict(union_questions=len(set.union(*sets)), common_questions=len(set.intersection(*sets)))
                 for c,sets in cohort_questions.items()},
        validation='84 saved accuracy means, strict means, denominators, caps and medians matched the outcome sidecar exactly; no model fitting or CI recomputation.',
        benchmark_ci='Original 1000-replicate question bootstrap, seed0; attempt-weighted mean among observed adjudicated outcomes. Intervals are per cell, not simultaneous.',
        caveats=[
            'Adjudicated and strict score means drop their own missing values; scored denominators differ. Explicit unknown counts and all-assignment bounds are exported.',
            'Historical accepted J1 audit (98/100) is a separate frozen scorer deployment from current minimal-verification shared J1 FAIL (97/100).',
            'Native decoding profiles differ in temperature, top-k, output caps, context and checkpoint precision; pooled attempt means are descriptive and over-weight repeated benchmarks.',
            'Cohort A selects one attempt per problem without correctness selection; cohort B is an outcome-conditioned case-control sample, not a benchmark accuracy estimate.',
            'Class experts are selected outcome-blind using the same cohort; saved significance is not held-out rank stability or evidence of dispensability.',
            'Present-class readout and next-class anticipation are observational, automatically labelled visible-episode tasks; neither proves memory savings or useful intervention.',
            'Existing GPT B4 correctness-prediction and calibrated precision study are preserved and must not be proposed as missing all-model evidence.',
        ], outputs=['benchmark_cells.csv','benchmark_models.csv','routing_readout.csv','anticipation.csv','cohort_populations.csv','expert_profile_inventory.csv','expert_usage_correctness_saved.csv'],
        latex=str(TEX / 'benchmark_readout_tables.tex'))
    dump('INVENTORY.json', inventory)
    dump('SOURCES.json', SOURCES)
    print(json.dumps({'card_models':card_models[['adjudicated_correct','adjudicated_scored','adjudicated_unknown','capped','completion_tokens_median']].reset_index().to_dict(orient='records'),
        'cohorts':inventory['cohorts'], 'outputs':inventory['outputs'], 'latex':inventory['latex']}, indent=2))


if __name__ == '__main__':
    main()
