#!/usr/bin/env python
"""Outcome-blind pre-registration of the routing "efficiency link".

For every model m and Schoenfeld class c this measures, on NATIVE judge-labelled reasoning
sentences only, how the class of the current sentence is associated with

  * the tokens that remain after the sentence (beta_tok), and
  * the correctness of the trace (beta_acc),

at the same absolute position of the same question.  Nothing from any intervention run is read.
The result is published as an immutable receipt (see ``moe_exp.final_routing.common.publish``)
together with a one-page human summary.

CPU only, no torch.  Usage (from the repository root)::

    PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \\
      MKL_NUM_THREADS=4 python scripts/study/efficiency_link.py

Estimator (one regression per model, outcome and class c, no intercept shared across questions)::

    y_s = a[question(s), position_bin(s)] + beta_c * 1[label(s) == c] + e_s

Fixed effects are absorbed exactly by within-group (Frisch-Waugh-Lovell) demeaning, so
``beta_c = sum_s w x~ y / sum_s w x~^2``.  The sums are additive over question families, which
makes the family-cluster bootstrap (resample families with replacement) exact and cheap:
``beta*_c = (k . A_c) / (k . B_c)`` with k the bootstrap multiplicities of the families.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd

try:  # transparent huge pages make numpy page faults ~100x slower on the login nodes
    ctypes.CDLL(None).prctl(41, 1, 0, 0, 0)
except Exception:  # pragma: no cover - best effort only
    pass

from moe_exp.final_routing.common import FAST, publish, file_sha  # noqa: E402

CLASSES = ('Read', 'Analyze', 'Plan', 'Implement', 'Explore', 'Verify', 'Monitor')
MODELS = ('gpt', 'nemotron', 'qwen35', 'qwen36', 'gemma', 'glm', 'qwen330b')
STUDY = FAST / 'final-routing-20261015-r1'
LEXICAL = STUDY / 'lexical'
OUT_JSON = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/final-routing-20261015-r1'
                '/driver/EFFICIENCY_LINK_PREREG.json')
OUT_MD = OUT_JSON.with_suffix('.md')
POSITION_EDGES = (0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, float('inf'))
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261017
OUTCOMES = ('tok_reason', 'tok_total', 'acc', 'acc_excl_trunc')
COLUMNS = ['label', 'family_id', 'question', 'dataset', 'trace_digest', 'sentence_index', 'context_hash',
           'reasoning_tok_start', 'n_tokens', 'n_reasoning_tokens', 'completion_tokens',
           'truncated', 'termination', 'strict_correct',
           'trace_inclusion_probability', 'arm_inclusion_probability_b']
VARIANTS = {
    'primary': dict(dedup=True, ipw=False, poscontrol=False, group='question'),
    'no_dedup': dict(dedup=False, ipw=False, poscontrol=False, group='question'),
    'ipw': dict(dedup=True, ipw=True, poscontrol=False, group='question'),
    'poscontrol': dict(dedup=True, ipw=False, poscontrol=True, group='question'),
    'dataset_fe': dict(dedup=True, ipw=False, poscontrol=False, group='dataset'),
}
SENSITIVITY = tuple(k for k in VARIANTS if k != 'primary')

ESTIMATOR_SPEC = {
    'population': 'NATIVE judge-labelled reasoning sentences (Qwen3.8-27B GEPA judge, campaign-v3 '
                  'labels), seven models; rows with a null label dropped. The labelled sentences '
                  'are a stratified SUBSET of each trace.',
    'dedup': 'Within a trace, rows sharing the same context_hash (identical judge input; '
             'repetition loops) are reduced to the first occurrence by sentence_index. Rows of '
             'distinct sentences are never merged otherwise; (trace_digest, sentence_index) is '
             'unique in the source tables.',
    'unit': 'one labelled reasoning sentence s of a trace',
    'regressor': 'x_s = 1[label(s) == c], one separate regression per class c (one-versus-rest)',
    'controls': 'fixed effect for (question, absolute-position bin of reasoning_tok_start); bin '
                'edges 0,256,512,1024,2048,4096,8192,16384,32768,inf, bin = [edge_i, edge_i+1). '
                'NO relative position and NO trace fixed effects (both leak the outcome).',
    'estimator': 'OLS with the question x bin fixed effects absorbed by within-group demeaning '
                 '(Frisch-Waugh-Lovell); beta_c = sum(w x~ y)/sum(w x~^2), x~ = x minus its '
                 'group mean. beta is the difference in the expected outcome when the current '
                 'sentence is class c versus not, at the same absolute position bin of the same '
                 'question.',
    'outcome_tok_reason': 'y = log1p(n_reasoning_tokens - (reasoning_tok_start + n_tokens)), '
                          'i.e. log1p of the REASONING tokens remaining after the sentence. The '
                          'completion-token end position of a sentence is not stored in the '
                          'native tables, so this is the primary token outcome (beta_tok). '
                          'Sentences of truncated traces (termination == length) are excluded.',
    'outcome_tok_total': 'y = log1p(completion_tokens - (reasoning_tok_start + n_tokens)) = log1p '
                         'of remaining reasoning tokens plus the answer tokens after the '
                         'reasoning. Same sample as tok_reason. Reported as a secondary token '
                         'outcome; it is not the pre-registered test statistic.',
    'outcome_acc': 'y = strict_correct (math_verify) of the trace; truncated traces count as '
                   'incorrect (y = 0 even where a value is stored); non-truncated traces without '
                   'a strict_correct value (unscored) are dropped. Pre-registered accuracy '
                   'outcome (beta_acc).',
    'outcome_acc_excl_trunc': 'as outcome_acc but truncated traces are excluded.',
    'weights': 'PRIMARY: unweighted (each labelled sentence counts once). 1/p design weights are '
               'NOT used because (i) trace_inclusion_probability is NULL for all cohort-B-only '
               'traces, whose own arm_inclusion_probability_b belongs to a different design, '
               '(ii) 1/p cuts the effective number of traces (Kish ESS 223-303 of 1,593-1,813 '
               'traces per model, p coalesced), moving the estimand toward the few questions '
               'with many attempts, and (iii) the test later averages over families with equal '
               'weight. The 1/p variant is reported as a sensitivity analysis '
               '(w = 1/trace_inclusion_probability, else 1/arm_inclusion_probability_b).',
    'bootstrap': {'cluster': 'family_id (FOLDS.json question family; the same 1486 families in '
                             'all seven models)',
                  'draws': BOOTSTRAP_DRAWS, 'seed': BOOTSTRAP_SEED,
                  'resampling': 'numpy.random.default_rng(seed).multinomial(n_families, '
                                'uniform, size=draws) over the sorted union of family ids; the '
                                'SAME draws are used for every model, outcome, class and for '
                                'the pooled estimate',
                  'interval': '95% percentile interval (numpy.nanpercentile 2.5, 97.5, linear '
                              'interpolation); draws in which the class has no within-group '
                              'variation are NaN and ignored',
                  'sign': "'+' if lower > 0, '-' if upper < 0, '0' otherwise"},
    'pooled': 'per class, the unweighted mean of the seven model estimates (models equally '
              'weighted), recomputed in every bootstrap draw with the shared family draws.',
    'sensitivity_variants': {
        'no_dedup': 'primary without the context_hash de-duplication',
        'ipw': 'primary with sentence weights 1/p (see weights)',
        'poscontrol': 'primary plus a within-group linear control for log1p(reasoning_tok_start) '
                      '(addresses the coarse position bins; the first bin spans 256 tokens)',
        'dataset_fe': 'primary with (dataset, position bin) fixed effects instead of '
                      '(question, position bin): a BETWEEN-question association that is confounded '
                      'by question difficulty but, unlike the primary, is not limited for the '
                      'trace-level accuracy outcome to the few questions with several attempts'},
    'opposite_sign_in': 'per cell and outcome, the sensitivity variants (no_dedup, ipw, '
                        'poscontrol) in which the OPPOSITE non-zero sign to a non-zero primary '
                        'sign is obtained; dataset_fe changes the estimand and is excluded from '
                        'this flag.',
    'sign_convention_tok': 'beta_tok > 0: a sentence of class c is followed by MORE remaining '
                           'tokens than other sentences at the same position (longer path); '
                           'beta_tok < 0: fewer remaining tokens (shorter path).',
    'sign_convention_acc': 'beta_acc > 0: class-c sentences are more frequent in correct traces.',
}

TEST_SPEC = {
    'name': 'efficiency_link_predicted_vs_measured',
    'status': 'pre-specified before any intervention outcome exists',
    'unit_of_analysis': 'cell = (model, class) with a nominated layer set and a pre-registered '
                        'native estimate (lfm25 has no native lexicon table and therefore no '
                        'prediction: at most 7 models x 7 classes = 49 cells)',
    'measured_token_effect': 'Delta_tok(m,c) = mean over question families f of '
                             '[ mean log(total tokens) over target_activate executions of cell '
                             '(m,c) in f at central strength  -  mean log(total tokens) over '
                             'random_activate executions of the matched random control at central '
                             'strength in f ]',
    'measured_accuracy_effect': 'Delta_acc(m,c) = mean over families f of [ accuracy of '
                                'target_activate - accuracy of random_activate ] at central '
                                'strength (same cells and pairing as Delta_tok)',
    'frozen_predictors': 'beta_tok(m,c) (outcome tok_reason) and beta_acc(m,c) (outcome acc) '
                         'exactly as in this file; they are not re-estimated.',
    'expected_direction': 'Activating class c should change tokens in the direction of '
                          'beta_tok(m,c): a POSITIVE Spearman correlation between beta_tok(m,c) '
                          'and Delta_tok(m,c) is predicted. For accuracy a POSITIVE correlation '
                          'between beta_acc(m,c) and Delta_acc(m,c) is predicted.',
    'primary_statistics': [
        'Spearman rank correlation rho_tok between beta_tok(m,c) and Delta_tok(m,c) over all '
        'usable cells',
        'Spearman rank correlation rho_acc between beta_acc(m,c) and Delta_acc(m,c) over all '
        'usable cells',
        'sign agreement rate: among cells whose pre-registered sign is + or - (CI excludes 0), '
        'the share where sign(Delta) equals the predicted sign (point estimate of Delta; cells '
        'with Delta exactly 0 count as disagreement), with an exact two-sided binomial test '
        'against 0.5; computed separately for tokens and accuracy'],
    'uncertainty': 'family-cluster bootstrap of the INTERVENTION side only: resample families '
                   'with replacement (same 2,000 draws / seed 20261017 as here, over the '
                   'families of the intervention study), recompute every Delta(m,c), then '
                   'rho and the agreement rate; 95% percentile intervals. The pre-registered '
                   'beta values are held fixed.',
    'support_criterion': 'the prediction is supported for tokens (accuracy) if the 95% interval '
                         'of rho_tok (rho_acc) lies above 0; it is not supported otherwise. The '
                         'sign-agreement rate is reported alongside, with its binomial p-value.',
    'secondary_statistics': [
        'the same rho and agreement statistics with the secondary predictors stored in this '
        'receipt: tok_total, acc_excl_trunc, and the between-question dataset_fe estimates '
        '(sensitivity.dataset_fe.cells[].tok_reason / .acc)',
        'rho_tok and rho_acc restricted to the cells whose primary sign is + or - and whose '
        "'opposite_sign_in' list is empty (sign not reversed by no_dedup, ipw or poscontrol)"],
    'excluded_from_test': 'target_deactivate arms, strengths other than central, and any '
                          'predictor not stored in this receipt. The secondary statistics are '
                          'reported but do not decide whether the prediction is supported.',
    'association_not_causation': 'beta values are native associations; a failed prediction is a '
                                 'result about the efficiency link, not about the estimator.',
}


def now_utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def atomic_text(path, text):
    """Write once; an existing file must be byte-identical (same rule as publish)."""
    path = Path(path)
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f'immutable file conflict: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            if path.read_text() != text:
                raise ValueError(f'concurrent file conflict: {path}')
    finally:
        Path(tmp).unlink(missing_ok=True)


# --------------------------------------------------------------------------- data


def load_model(model, directory, dedup):
    path = Path(directory) / f'native_{model}.parquet'
    frame = pd.read_parquet(path, columns=COLUMNS)
    info = {'rows': int(len(frame)), 'null_label_rows': int(frame.label.isna().sum())}
    frame = frame[frame.label.notna()].copy()
    if dedup:
        before = len(frame)
        frame = frame.sort_values(['trace_digest', 'sentence_index'])
        frame = frame.drop_duplicates(['trace_digest', 'context_hash'], keep='first')
        info['dedup_dropped_rows'] = int(before - len(frame))
    else:
        info['dedup_dropped_rows'] = 0
    if frame.truncated.to_numpy().tolist() != (frame.termination == 'length').to_numpy().tolist():
        raise ValueError(f'{model}: truncated != (termination == length)')
    end = frame.reasoning_tok_start.to_numpy() + frame.n_tokens.to_numpy()
    if (end > frame.n_reasoning_tokens.to_numpy()).any():
        raise ValueError(f'{model}: sentence end beyond n_reasoning_tokens')
    if (frame.n_reasoning_tokens.to_numpy() > frame.completion_tokens.to_numpy()).any():
        raise ValueError(f'{model}: n_reasoning_tokens > completion_tokens')
    frame['end'] = end
    frame['y_tok_reason'] = np.log1p(frame.n_reasoning_tokens.to_numpy() - end)
    frame['y_tok_total'] = np.log1p(frame.completion_tokens.to_numpy() - end)
    interior = np.asarray(POSITION_EDGES[1:-1], dtype=float)
    frame['bin'] = np.searchsorted(interior, frame.reasoning_tok_start.to_numpy(), side='right')
    probability = frame.trace_inclusion_probability.fillna(frame.arm_inclusion_probability_b)
    if probability.isna().any():
        raise ValueError(f'{model}: no inclusion probability for some rows')
    frame['ipw'] = 1.0 / probability.to_numpy()
    correct = frame.strict_correct.map({True: 1.0, False: 0.0}).astype('float64')  # None -> NaN
    frame['y_acc'] = np.where(frame.truncated.to_numpy(), 0.0, correct.to_numpy())
    info['truncated_traces'] = int(frame[frame.truncated].trace_digest.nunique())
    info['truncated_with_stored_strict_correct'] = int(
        frame[frame.truncated & correct.notna()].trace_digest.nunique())
    info['unscored_nontruncated_traces'] = int(
        frame[~frame.truncated & correct.isna()].trace_digest.nunique())
    return frame, info


def outcome_frames(frame):
    """Rows and outcome column for each outcome."""
    tok = frame[~frame.truncated]
    acc = frame[frame.truncated | frame.y_acc.notna()]
    acc_x = frame[~frame.truncated & frame.y_acc.notna()]
    return {'tok_reason': (tok, 'y_tok_reason'), 'tok_total': (tok, 'y_tok_total'),
            'acc': (acc, 'y_acc'), 'acc_excl_trunc': (acc_x, 'y_acc')}


# ------------------------------------------------------------------------ estimator


def family_sums(frame, ycol, family_index, n_families, ipw, poscontrol, group_key='question'):
    """Per-family additive sums behind beta_c for all classes (fixed effects absorbed)."""
    group = pd.factorize(frame[group_key].astype(str) + '|' + frame.bin.astype(str))[0]
    fam = frame.family_id.map(family_index).to_numpy()
    if np.isnan(fam.astype(float)).any():
        raise ValueError('family missing from the family index')
    fam = fam.astype(np.int64)
    w = frame.ipw.to_numpy() if ipw else np.ones(len(frame))
    y = frame[ycol].to_numpy(dtype=float)
    total = np.bincount(group, w)

    def center(v):
        return v - (np.bincount(group, w * v) / total)[group]

    sums = np.zeros((n_families, len(CLASSES), 3))  # Sxx, Sxy, Sxz
    common = np.zeros((n_families, 2))  # Szz, Szy
    z_c = None
    if poscontrol:
        z_c = center(np.log1p(frame.reasoning_tok_start.to_numpy(dtype=float)))
        common[:, 0] = np.bincount(fam, w * z_c * z_c, minlength=n_families)
        common[:, 1] = np.bincount(fam, w * z_c * y, minlength=n_families)
    labels = frame.label.to_numpy()
    informative = np.zeros(len(CLASSES), dtype=np.int64)
    for c, name in enumerate(CLASSES):
        x = (labels == name).astype(float)
        xc = center(x)
        sums[:, c, 0] = np.bincount(fam, w * xc * xc, minlength=n_families)
        sums[:, c, 1] = np.bincount(fam, w * xc * y, minlength=n_families)
        if poscontrol:
            sums[:, c, 2] = np.bincount(fam, w * xc * z_c, minlength=n_families)
        informative[c] = int((np.abs(xc) > 1e-12).sum())
    return sums, common, informative


def betas(multiplicity, sums, common, poscontrol):
    """beta for each row of ``multiplicity`` (draws x families) and each class."""
    flat = multiplicity @ sums.reshape(sums.shape[0], -1)
    flat = flat.reshape(multiplicity.shape[0], len(CLASSES), 3)
    sxx, sxy, sxz = flat[..., 0], flat[..., 1], flat[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        if not poscontrol:
            out = np.where(sxx > 1e-12, sxy / sxx, np.nan)
        else:
            szz = (multiplicity @ common[:, 0])[:, None]
            szy = (multiplicity @ common[:, 1])[:, None]
            det = sxx * szz - sxz * sxz
            out = np.where(det > 1e-12, (szz * sxy - sxz * szy) / det, np.nan)
    return out


def sign_of(lo, hi):
    if lo is None or hi is None or np.isnan(lo) or np.isnan(hi):
        return 'NA'
    if lo > 0:
        return '+'
    if hi < 0:
        return '-'
    return '0'


def interval(draws):
    """95% percentile interval over draws (axis 0), ignoring NaN draws."""
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    return lo, hi, np.isfinite(draws).sum(axis=0)


def fnum(v):
    v = float(v)
    return v if np.isfinite(v) else None


def run_variant(name, spec, directory, family_index, multiplicity, models):
    """Estimate every model x outcome x class; return dict and the bootstrap draws."""
    n_fam = len(family_index)
    point = np.ones((1, n_fam))
    est = {}
    infos = {}
    counts = {}
    for model in models:
        frame, info = load_model(model, directory, spec['dedup'])
        infos[model] = info
        for outcome, (rows, ycol) in outcome_frames(frame).items():
            sums, common, informative = family_sums(
                rows, ycol, family_index, n_fam, spec['ipw'], spec['poscontrol'], spec['group'])
            beta = betas(point, sums, common, spec['poscontrol'])[0]
            boot = betas(multiplicity, sums, common, spec['poscontrol'])
            est[(model, outcome)] = (beta, boot)
            labels = rows.label.to_numpy()
            traces = rows.trace_digest.to_numpy()
            fams = rows.family_id.to_numpy()
            cell_counts = []
            for c, cls in enumerate(CLASSES):
                mask = labels == cls
                cell_counts.append({
                    'n_sentences_class': int(mask.sum()),
                    'n_traces_class': int(pd.unique(traces[mask]).size),
                    'n_families_class': int(pd.unique(fams[mask]).size),
                    'n_sentences_in_groups_with_class_variation': int(informative[c])})
            counts[(model, outcome)] = {
                'n_sentences': int(len(rows)), 'n_traces': int(rows.trace_digest.nunique()),
                'n_families': int(rows.family_id.nunique()),
                'n_questions': int(rows.question.nunique()), 'classes': cell_counts}
            if outcome in ('acc', 'acc_excl_trunc'):
                per_q = rows.groupby('question').y_acc.nunique()
                counts[(model, outcome)]['n_questions_with_outcome_variation'] = int(
                    (per_q > 1).sum())
    return est, infos, counts


def summarise(est, models):
    """beta, interval, sign per (model, outcome, class) and pooled per (outcome, class)."""
    cells = {}
    pooled = {}
    for outcome in OUTCOMES:
        stack_point = []
        stack_boot = []
        for model in models:
            beta, boot = est[(model, outcome)]
            lo, hi, valid = interval(boot)
            for c, cls in enumerate(CLASSES):
                cells[(model, outcome, cls)] = {
                    'beta': fnum(beta[c]), 'ci_low': fnum(lo[c]), 'ci_high': fnum(hi[c]),
                    'sign': sign_of(lo[c], hi[c]), 'valid_draws': int(valid[c])}
            stack_point.append(beta)
            stack_boot.append(boot)
        boot = np.stack(stack_boot)  # models x draws x classes
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            point = np.nanmean(stack_point, axis=0)
            pooled_boot = np.nanmean(boot, axis=0)
        lo, hi, valid = interval(pooled_boot)
        for c, cls in enumerate(CLASSES):
            signs = [cells[(m, outcome, cls)]['sign'] for m in models]
            pooled[(outcome, cls)] = {
                'beta': fnum(point[c]), 'ci_low': fnum(lo[c]), 'ci_high': fnum(hi[c]),
                'sign': sign_of(lo[c], hi[c]), 'valid_draws': int(valid[c]),
                'models_plus': signs.count('+'), 'models_minus': signs.count('-'),
                'models_zero': signs.count('0'), 'models_na': signs.count('NA')}
    return cells, pooled


# ---------------------------------------------------------------------------- report


def consistency(cells, outcome, models):
    """Classes whose model-level signs agree (>=5 of 7 same non-zero sign, none opposite)."""
    result = {}
    for cls in CLASSES:
        signs = [cells[(m, outcome, cls)]['sign'] for m in models]
        plus, minus = signs.count('+'), signs.count('-')
        if plus >= 5 and minus == 0:
            result[cls] = '+'
        elif minus >= 5 and plus == 0:
            result[cls] = '-'
        else:
            result[cls] = 'mixed' if plus and minus else 'none'
    return result


def markdown(payload, models):
    cells = {(c['model'], c['class']): c for c in payload['cells']}
    pooled = {p['class']: p for p in payload['pooled']}

    def sgn(s):
        return {'-': '−', 'NA': 'n/a'}.get(s, s)

    def table(outcome, title):
        head = '| model | ' + ' | '.join(CLASSES) + ' |\n|---|' + '---|' * len(CLASSES) + '\n'
        rows = ''
        for m in models:
            rows += f'| {m} | ' + ' | '.join(
                sgn(cells[(m, c)][outcome]['sign']) for c in CLASSES) + ' |\n'
        rows += '| **pooled** | ' + ' | '.join(
            f"**{sgn(pooled[c][outcome]['sign'])}**" for c in CLASSES) + ' |\n'
        return f'### {title}\n\n{head}{rows}\n'

    def pooled_line(outcome):
        return f"| {'beta_tok' if outcome == 'tok_reason' else 'beta_acc'} | " + ' | '.join(
            f"{pooled[c][outcome]['beta']:+.3f} [{pooled[c][outcome]['ci_low']:+.3f}, "
            f"{pooled[c][outcome]['ci_high']:+.3f}]" for c in CLASSES) + ' |'

    summary = payload['data_summary']
    trunc_text = ', '.join(f"{m} {summary[m]['truncated_traces']}" for m in models)
    var_counts = [summary[m]['analysis']['acc']['n_questions_with_outcome_variation']
                  for m in models]
    var_text = f'{min(var_counts)}-{max(var_counts)}'
    acc_signs = [cells[(m, c)]['acc']['sign'] for m in models for c in CLASSES]
    acc_zero, acc_total = acc_signs.count('0'), len(acc_signs)
    def flipped(outcome):
        parts = []
        for cls in CLASSES:
            hit = [cells[(m, cls)][outcome]['opposite_sign_in'] for m in models]
            n = sum(1 for h in hit if h)
            if n:
                variants = sorted({v for h in hit for v in h})
                parts.append(f"{cls} {n}/{len(models)} ({', '.join(variants)})")
        return '; '.join(parts) if parts else 'none'

    between = {p['class']: p for p in payload['sensitivity']['dataset_fe']['pooled']}
    between_tok = ' '.join(f"{c} {sgn(between[c]['tok_reason']['sign'])}" for c in CLASSES)
    between_acc = ' '.join(f"{c} {sgn(between[c]['acc']['sign'])}" for c in CLASSES)
    cons_tok = payload['consistency']['tok_reason']
    cons_acc = payload['consistency']['acc']

    def describe(cons):
        parts = []
        for sign, word in (('-', 'negative'), ('+', 'positive')):
            names = [c for c in CLASSES if cons[c] == sign]
            parts.append(f"{word}: {', '.join(names) if names else 'none'}")
        return '; '.join(parts)

    text = f"""# Efficiency link pre-registration (outcome-blind)

Created {payload['created_utc']}. `intervention_outcomes_used: false`. Machine-readable receipt:
`EFFICIENCY_LINK_PREREG.json` (sha256 of the script and of every input recorded there).

**What is predicted.** For each model and class, the sign of the native association between the
class of the current sentence and (i) the log of the reasoning tokens that remain after it
(`beta_tok`, negative = shorter path) and (ii) trace correctness (`beta_acc`), at the same
absolute position of the same question (question x position-bin fixed effects, no relative
position, no trace effects). Sign `+`/`-` only if the 95% family-cluster bootstrap interval
(2,000 draws, seed 20261017) excludes 0; otherwise `0`. Later test: Spearman correlation between
`beta_tok(m,c)` and the measured intervention token effect over (model, class) cells, plus the
sign-agreement rate; analogously for accuracy (positive correlation predicted).

{table('tok_reason', 'Signs of beta_tok (remaining reasoning tokens)')}{table('acc', 'Signs of beta_acc (trace correct; truncated = incorrect)')}### Pooled estimates (mean of the seven models, 95% interval)

| | {' | '.join(CLASSES)} |
|---|{'---|' * len(CLASSES)}
{pooled_line('tok_reason')}
{pooled_line('acc')}

### Consistency across models (at least 5 of 7 models share the sign, none opposite)

- tokens, sign by class: {describe(cons_tok)}
- accuracy, sign by class: {describe(cons_acc)}

### Robustness (sensitivity variants; they never change the pre-registered signs)

- Models whose primary non-zero sign is reversed (opposite non-zero sign) by a variant, tokens:
  {flipped('tok_reason')}.
- Same for accuracy: {flipped('acc')}.
- Between-question variant (dataset x position-bin effects), pooled signs. Tokens: {between_tok}.
  Accuracy: {between_acc}.

### Caveats

- Association, not causation: class and remaining length are both outcomes of the same reasoning
  path; the later intervention test is what links them.
- Labelled sentences are a stratified SUBSET of each trace (median 8-33% of units); labels are
  single temperature-1.0 judge draws. Identical judge inputs inside a trace (repetition loops)
  are reduced to their first occurrence. Estimates are unweighted; 1/p design weights are a
  sensitivity analysis only.
- Truncation: for tokens, sentences of traces that hit the generation cap are excluded (their
  remaining length is censored); for accuracy they count as incorrect, with the variant excluding
  them reported as `acc_excl_trunc`. Truncated traces per model: {trunc_text}.
- Accuracy is a trace-level outcome: with question fixed effects it is identified only from
  questions that have several attempts with different outcomes ({var_text} questions per
  model), so `beta_acc` is far less precise than `beta_tok` and {acc_zero} of the
  {acc_total} model x class accuracy cells have sign `0`.
- `beta_tok` uses remaining REASONING tokens (the native tables do not store the completion
  token end position); the secondary `tok_total` adds the answer tokens.
- lfm25 has no native lexicon table: its cells have no prediction.
"""
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--lexical-dir', default=str(LEXICAL))
    parser.add_argument('--out', default=str(OUT_JSON))
    parser.add_argument('--models', default=','.join(MODELS))
    parser.add_argument('--draws', type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument('--seed', type=int, default=BOOTSTRAP_SEED)
    parser.add_argument('--dry-run', action='store_true', help='compute, print, do not publish')
    args = parser.parse_args()
    started = time.time()
    models = tuple(args.models.split(','))
    directory = Path(args.lexical_dir)
    if not args.dry_run and (tuple(models) != MODELS or args.draws != BOOTSTRAP_DRAWS
                             or args.seed != BOOTSTRAP_SEED):
        raise SystemExit('publication requires all seven models, 2000 draws and seed 20261017')

    families = {}
    for model in models:
        frame = pd.read_parquet(directory / f'native_{model}.parquet', columns=['family_id'])
        families[model] = set(frame.family_id.unique())
    union = sorted(set().union(*families.values()))
    family_index = {f: i for i, f in enumerate(union)}
    rng = np.random.default_rng(args.seed)
    multiplicity = rng.multinomial(len(union), np.full(len(union), 1.0 / len(union)),
                                   size=args.draws).astype(np.float64)

    results = {}
    for name, spec in VARIANTS.items():
        est, infos, counts = run_variant(name, spec, directory, family_index, multiplicity, models)
        cells, pooled = summarise(est, models)
        results[name] = (cells, pooled, infos, counts)
        print(f'{name} done {time.time() - started:.0f}s', flush=True)

    cells, pooled, infos, counts = results['primary']
    cell_rows = []
    for model in models:
        for cls in CLASSES:
            row = {'model': model, 'class': cls}
            for outcome in OUTCOMES:
                row[outcome] = cells[(model, outcome, cls)]
                c = counts[(model, outcome)]
                k = CLASSES.index(cls)
                row[outcome]['n'] = {
                    'n_sentences_analysis': c['n_sentences'], 'n_traces_analysis': c['n_traces'],
                    'n_families_analysis': c['n_families'], **c['classes'][k]}
                primary_sign = cells[(model, outcome, cls)]['sign']
                opposite = '-' if primary_sign == '+' else '+'
                row[outcome]['opposite_sign_in'] = [
                    v for v in ('no_dedup', 'ipw', 'poscontrol')
                    if primary_sign in ('+', '-')
                    and results[v][0][(model, outcome, cls)]['sign'] == opposite]
            cell_rows.append(row)
    pooled_rows = [{'class': cls, **{o: pooled[(o, cls)] for o in OUTCOMES}} for cls in CLASSES]

    sensitivity = {}
    for name in SENSITIVITY:
        vcells, vpooled, _, _ = results[name]
        agree = {}
        for outcome in OUTCOMES:
            same = other = 0
            for model in models:
                for cls in CLASSES:
                    a, b = cells[(model, outcome, cls)]['sign'], vcells[(model, outcome, cls)]['sign']
                    same += a == b
                    other += a != b
            agree[outcome] = {'cells_same_sign': same, 'cells_different_sign': other}
        sensitivity[name] = {
            'spec': VARIANTS[name],
            'sign_agreement_with_primary': agree,
            'cells': [{'model': m, 'class': cls,
                       **{o: {k: vcells[(m, o, cls)][k] for k in ('beta', 'ci_low', 'ci_high', 'sign')}
                          for o in OUTCOMES}} for m in models for cls in CLASSES],
            'pooled': [{'class': cls, **{o: {k: vpooled[(o, cls)][k]
                                            for k in ('beta', 'ci_low', 'ci_high', 'sign')}
                                         for o in OUTCOMES}} for cls in CLASSES]}

    consistent = {o: consistency(cells, o, models) for o in OUTCOMES}
    data_summary = {m: {**infos[m], 'n_families': len(families[m]),
                        'analysis': {o: {k: v for k, v in counts[(m, o)].items() if k != 'classes'}
                                     for o in OUTCOMES}} for m in models}
    class_share = {m: counts[(m, 'tok_reason')]['classes'] for m in models}

    script = Path(__file__).resolve()
    git = subprocess.run(['git', '-C', str(script.parents[2]), 'rev-parse', 'HEAD'],
                         capture_output=True, text=True)
    inputs = {'directory': str(directory),
              'native_parquet_sha256': {m: file_sha(directory / f'native_{m}.parquet')
                                        for m in models},
              'readme_sha256': file_sha(directory / 'README.md')}
    payload = {
        'kind': 'efficiency_link_preregistration', 'version': 1,
        'intervention_outcomes_used': False,
        'purpose': 'Outcome-blind prediction of the sign of the token / accuracy effect of '
                   'pushing a MoE reasoning model toward each Schoenfeld class, from native '
                   'associations only.',
        'models': list(models), 'classes': list(CLASSES),
        'unavailable_models': ['lfm25 (no native lexicon table)'],
        'estimator_spec': ESTIMATOR_SPEC, 'test': TEST_SPEC,
        'inputs': inputs,
        'script': {'path': 'scripts/study/efficiency_link.py', 'sha256': file_sha(script),
                   'git_head': git.stdout.strip() or None,
                   'git_tree_note': 'working tree had uncommitted changes when run'},
        'data_summary': data_summary, 'class_counts_tok_reason': class_share,
        'cells': cell_rows, 'pooled': pooled_rows, 'consistency': consistent,
        'sensitivity': sensitivity,
        'runtime_seconds': round(time.time() - started, 1)}

    out = Path(args.out)
    if out.exists():  # re-runs on identical inputs reuse the original timestamp and runtime
        old = json.loads(out.read_text())
        payload['created_utc'] = old.get('created_utc')
        payload['runtime_seconds'] = old.get('runtime_seconds')
    else:
        payload['created_utc'] = now_utc()
    if args.dry_run:
        for name, (vcells, vpooled, _, vcounts) in results.items():
            for outcome in OUTCOMES:
                print(f'== {name} {outcome}: beta (sign)')
                for model in models:
                    print(f'{model:9s}', ' '.join(
                        f"{vcells[(model, outcome, c)]['beta']:+.4f}({vcells[(model, outcome, c)]['sign']})"
                        for c in CLASSES))
                print(f'{"pooled":9s}', ' '.join(
                    f"{vpooled[(outcome, c)]['beta']:+.4f}({vpooled[(outcome, c)]['sign']})"
                    for c in CLASSES))
        print(json.dumps({'sens_agreement': {k: v['sign_agreement_with_primary']
                                             for k, v in sensitivity.items()},
                          'consistency': consistent}, indent=1))
        print(json.dumps(data_summary, indent=1)[:4000])
        return
    receipt = publish(out, payload)
    atomic_text(out.with_suffix('.md'), markdown(payload, models))
    print(f'published {out} sha256={receipt["sha256"]}')
    print(f'wrote {out.with_suffix(".md")}')


if __name__ == '__main__':
    sys.exit(main())
