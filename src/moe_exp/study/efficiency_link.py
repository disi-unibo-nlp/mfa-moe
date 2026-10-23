"""Efficiency link for all eight models: LFM2.5 gets its own native signs.

The pre-registration (scripts/study/efficiency_link.py -> driver/EFFICIENCY_LINK_PREREG.json) had no
native table for LFM2.5, so the whole-reasoning steering gave LFM the pooled seven-model signs. LFM's
labelled native sentences exist since 2026-10-17 22:22, but lexical/native_lfm25.parquet keeps only
labels and text (it was built for the lexical readout). This module

  table      joins those labels with LFM's sampled sources and native outcomes into the columns the
             estimator reads, in a new lexical directory next to links to the seven existing tables;
  publish    runs the frozen estimator unchanged on all eight models and publishes
             driver/EFFICIENCY_LINK_V2.json (+ .md). The seven models' cells must equal the prereg's;
  driver-test reruns the driver's frozen link test (efficiency_link) on the saved analysis-final
             pairs with the V2 cells, so LFM's seven cells join the 49.

LFM's signs are estimated after its interventions ran; the receipt records that. The estimator, draws
and seed are the pre-registered ones and the seven pre-registered models are unchanged.

    scripts/study/cpu.sh moe_exp.study.efficiency_link all
"""
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import sys
import tempfile

import pandas as pd

from moe_exp.final_routing.common import clean
from .common import REPO, file_sha, load, publish, require, verified
from . import driver

FROZEN_SCRIPT = REPO/'scripts/study/efficiency_link.py'
MODELS = ('gpt', 'nemotron', 'qwen35', 'qwen36', 'gemma', 'glm', 'qwen330b', 'lfm25')
PREREG = driver.DRIVER/'EFFICIENCY_LINK_PREREG.json'
LINK = driver.DRIVER/'EFFICIENCY_LINK_V2.json'
DRIVER_TEST = driver.DRIVER/'analysis-final-link-v2/EFFICIENCY_LINK_TEST.json'


def frozen():
    """The pre-registration script, loaded by path so that it stays byte-identical."""
    spec = importlib.util.spec_from_file_location('efficiency_link_v1', FROZEN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(file_sha(FROZEN_SCRIPT) == verified(PREREG)['script']['sha256'], 'frozen estimator script changed')
    return module


def same(a, b) -> bool:
    """Equal receipts up to the last bits of a float (the estimator sums in BLAS order)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


class CapturedReceipt:
    """Stands in for the frozen script's publish(): keeps the payload instead of writing it."""

    def __init__(self):
        self.payload = None

    def __call__(self, path, payload):
        self.payload = clean(payload)
        return {'sha256': 'captured'}


def lexical_dir(estimator) -> Path:
    return estimator.STUDY/'lexical-link-r2'


def build_table(estimator) -> Path:
    """Write lexical-link-r2/native_lfm25.parquet with the estimator's columns, plus links to the others.

    Rows are the 100,000 sampled LFM sentences in sampling order, labels from lexical/native_lfm25.parquet.
    Positions are ranks among the trace's reasoning tokens (as in the seven tables), the trace outcome
    is LFM's native strict correctness and termination (NATIVE_ESTIMANDS.json, scored in the main
    process), and the sentence inclusion probability is selected / supply sentences of the sampling
    cell (only the 1/p sensitivity variant reads it).
    """
    report = driver.ROOT/'lfm25'
    labels = pd.read_parquet(estimator.LEXICAL/'native_lfm25.parquet')
    sampled = verified(estimator.STUDY/'lfm25/SAMPLED_SENTENCES.json')['records']
    require(labels.context_hash.tolist() == [r['annotation_uid'] for r in sampled], 'LFM labels and samples differ')
    sources = {s['attempt_id']: s for s in verified(report/'SOURCES.json')['records']}
    natives = {r['uid']: r for r in verified(report/'NATIVE_ESTIMANDS.json')['records']}
    cells = verified(report/'SAMPLING_MANIFEST.json')['cells']
    rows = []
    for record, label in zip(sampled, labels.label):
        source = sources[record['attempt_id']]
        native = natives[source['native_uid']]
        ranks = record['token_ranks']
        require(ranks and max(ranks)-min(ranks)+1 == len(ranks), 'LFM sentence tokens are not contiguous')
        cell = cells[source['stratum']]
        rows.append({
            'label': label, 'family_id': record['family'], 'question': record['question'],
            'dataset': record['benchmark'], 'trace_digest': source['trace_sha256'],
            'sentence_index': int(record['sentence_index']), 'context_hash': record['annotation_uid'],
            'reasoning_tok_start': min(ranks), 'n_tokens': len(ranks),
            'n_reasoning_tokens': int(record['n_reasoning']), 'completion_tokens': int(source['completion_tokens']),
            'truncated': native['termination'] == 'length', 'termination': native['termination'],
            'strict_correct': native['strict'], 'trace_inclusion_probability': None,
            'arm_inclusion_probability_b': cell['selected']/cell['supply_sentences']})
    table = pd.DataFrame(rows)
    table['trace_inclusion_probability'] = table.trace_inclusion_probability.astype('float64')
    folder = lexical_dir(estimator)
    require(not folder.exists(), f'{folder} exists; the table is written once')
    folder.mkdir(mode=0o700)
    for model in MODELS[:-1]:
        (folder/f'native_{model}.parquet').symlink_to(estimator.LEXICAL/f'native_{model}.parquet')
    table.to_parquet(folder/'native_lfm25.parquet', index=False)
    (folder/'README.md').write_text(
        '# Native tables for the eight-model efficiency link\n\n'
        'The seven `native_<model>.parquet` files link to `../lexical/` (the pre-registered inputs).\n'
        '`native_lfm25.parquet` is built by `moe_exp.study.efficiency_link table` from LFM2.5\'s sampled\n'
        'native sentences (`lfm25/SAMPLED_SENTENCES.json`, labels from `../lexical/native_lfm25.parquet`),\n'
        'its sampled sources (`report/final-routing-20261015-r1/lfm25/SOURCES.json`) and native outcomes\n'
        '(`NATIVE_ESTIMANDS.json`). It has only the columns the estimator reads. Positions are ranks among\n'
        'the trace\'s reasoning tokens; `arm_inclusion_probability_b` is the sampling cell\'s selected /\n'
        'supply sentences and `trace_inclusion_probability` is empty. See ../lexical/README.md for columns.\n')
    return folder


def publish_link(estimator, directory: Path) -> dict:
    """Run the frozen estimator on eight models and publish EFFICIENCY_LINK_V2.json.

    The frozen main() refuses to publish anything but the seven-model receipt, so its MODELS is
    widened and its publish/atomic_text are captured; the estimation itself runs unchanged.
    """
    require(not LINK.exists(), f'{LINK} exists')
    captured = CapturedReceipt()
    estimator.MODELS = MODELS
    estimator.publish = captured
    estimator.atomic_text = lambda path, text: None
    with tempfile.TemporaryDirectory() as tmp:
        saved, sys.argv = sys.argv, ['efficiency_link.py', '--lexical-dir', str(directory),
                                     '--out', str(Path(tmp)/'link.json')]
        try:
            estimator.main()
        finally:
            sys.argv = saved
    payload = captured.payload
    prereg = verified(PREREG)
    old = {(c['model'], c['class']): c for c in prereg['cells']}
    for cell in payload['cells']:
        if cell['model'] != 'lfm25':
            require(same(cell, old[cell['model'], cell['class']]),
                    f"pre-registered cell changed: {cell['model']} {cell['class']}")
    for model, sha in prereg['inputs']['native_parquet_sha256'].items():
        require(payload['inputs']['native_parquet_sha256'][model] == sha, f'pre-registered {model} table changed')
    payload.update({
        'kind': 'efficiency_link_eight_models', 'version': 2, 'models': list(MODELS), 'unavailable_models': [],
        'created_utc': estimator.now_utc(), 'predecessor': {'path': str(PREREG), 'sha256': prereg['sha256']},
        'lfm25_estimated_after_interventions': True,
        'note': 'Frozen estimator, draws and seed of the pre-registration. The cells of the seven '
                'pre-registered models equal the predecessor\'s (checked to float rounding); LFM2.5 (no native '
                'table at pre-registration) is estimated after its interventions ran, and the pooled row is '
                'now the eight-model mean.',
        'builder': {'path': 'src/moe_exp/study/efficiency_link.py', 'sha256': file_sha(Path(__file__))}})
    payload['estimator_spec'] = {**payload['estimator_spec'],
                                 'pooled': payload['estimator_spec']['pooled'].replace('seven', 'eight')}
    payload['test'] = {**payload['test'], 'unit_of_analysis': 'cell = (model, class) with a nominated layer '
                       'set and a native estimate: at most 8 models x 7 classes = 56 cells'}
    receipt = publish(LINK, payload)
    text = estimator.markdown(payload, MODELS).replace('mean of the seven models', 'mean of the eight models').replace(
        'at least 5 of 7 models', 'at least 5 of 8 models').replace(
        '- lfm25 has no native lexicon table: its cells have no prediction.',
        '- lfm25 is estimated after its interventions ran (no native table existed at pre-registration);\n'
        '  the seven other models are identical to EFFICIENCY_LINK_PREREG.json.')
    LINK.with_suffix('.md').write_text(text.replace('# Efficiency link pre-registration (outcome-blind)',
                                                    '# Efficiency link, eight models (v2)'))
    return receipt


def driver_test() -> dict:
    """The driver's frozen link test on analysis-final pairs, with the V2 cells (56 instead of 49)."""
    from .effects import Bootstrap, efficiency_link
    require(not DRIVER_TEST.exists(), f'{DRIVER_TEST} exists')
    final = driver.DRIVER/'analysis-final'
    pairs = pd.read_parquet(final/'pairs.parquet')
    families = sorted(pd.read_parquet(final/'continuations.parquet', columns=['family']).family.unique())
    benchmarks = {r['family']: r['benchmark'] for r in load(driver.ROOT/'ALLOCATION.json')['records']}
    boot = Bootstrap(families, benchmarks)
    # analysis-final ran the plain test (it predates the bounded-outcome wrapper). Its intervals came from
    # an earlier version of the shared Bootstrap draws, so on the pre-registration today's code must
    # reproduce every point estimate exactly; the 49-cell test is republished with today's intervals so
    # that the 49- and 56-cell tests compare like for like.
    published = verified(final/'EFFECTS.json')['efficiency_link']
    reproduced = clean(efficiency_link(verified(PREREG), pairs, boot))
    require(same(point_estimates(reproduced), point_estimates(published)),
            'driver link test does not reproduce the point estimates of analysis-final')
    result = clean(efficiency_link(verified(LINK), pairs, boot))
    DRIVER_TEST.parent.mkdir(mode=0o700)
    return publish(DRIVER_TEST, {**result, 'prereg_sha256': verified(LINK)['sha256'],
                                 'preregistered_cells_same_code': reproduced,
                                 'pairs': str(final/'pairs.parquet'),
                                 'effects_sha256': verified(final/'EFFECTS.json')['sha256']})


def point_estimates(test: dict) -> dict:
    """A link-test result without its bootstrap intervals and the statistics derived from them."""
    interval_keys = ('spearman_ci', 'sign_agreement_ci', 'supported', 'prereg_sha256')
    return {key: point_estimates(value) if isinstance(value, dict) else value
            for key, value in test.items() if key not in interval_keys and key != 'tok_reasoning'}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog='python -m moe_exp.study.efficiency_link')
    parser.add_argument('command', choices=('table', 'publish', 'driver-test', 'all'))
    args = parser.parse_args(argv)
    estimator = frozen()
    if args.command in ('table', 'all'):
        print('table', build_table(estimator), flush=True)
    if args.command in ('publish', 'all'):
        print('link', publish_link(estimator, lexical_dir(estimator))['sha256'], flush=True)
    if args.command in ('driver-test', 'all'):
        print('driver test', driver_test()['sha256'], flush=True)


if __name__ == '__main__':
    main()
