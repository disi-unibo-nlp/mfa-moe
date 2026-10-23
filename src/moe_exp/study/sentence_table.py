"""A campaign-v3 model's labelled-sentence table, rebuilt from its saved traces and original judge labels.

The dynamics tables behind the paper's state-transition figure
(`dynamics-routing/results/<model>/A/sentences.parquet`) hold one row per judge-labelled sentence that owns
at least one reasoning token. The other models' tables cover all 1,547 attempts (1,486 families), but
Qwen3.6's covers only 35,484 sentences of 522 questions (509 families): its extraction kept the steering dev
and tune questions, the only ones with a router-tensor cache. The sentence layout needs no router tensor.
Offsets, reasoning tokens and token ownership are fixed by the saved trace (`saved_layout`), and the labels by
the original judge records, so the same table can be rebuilt for every attempt.

This module reads the traces and the 4 x 25,000-sentence judge records of one model, applies the matching
rules of the extraction (a record counts only when its offsets equal the current sentence, an invalid label
is skipped and a conflicting one removed), writes the table with the extraction's columns and checks it in two
ways: per attempt, the class token totals equal the nine saved class counts, and for every attempt that the
old table also holds, the rows are equal.

    STUDY_MODULE=moe_exp.study.sentence_table scripts/study/driver.sh rebuild --model qwen36   (CPU job)
    python -m moe_exp.study.sentence_table rebuild --model qwen36 --limit 8 --output PATH       (small debug run)
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import multiprocessing
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .common import CLASSES, file_sha, load, publish, require
from .driver import ROOT, binding

# Next to the final-routing report (not under the code tree, so a source snapshot finds it too).
CAMPAIGN = ROOT.parent/'research-campaign-v3-20261014-r5-r2'
SOURCE_BASE = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
PAIR_SOURCES = ('Read', 'Verify')
# Absolute reasoning-token position bins of the extraction: 0-1k, 1-2k, 2-4k, 4-8k, 8-16k, 16k+ (1k = 1024).
ABSOLUTE_BIN_EDGES = (1024, 2048, 4096, 8192, 16384)
COLUMNS = ('attempt_id', 'question', 'dataset', 'sentence_index', 'cls', 'segment', 'tok_start', 'tok_end',
           'n_tokens', 'has_prev_labelled', 'has_next_labelled', 'prev_same_segment', 'next_same_segment',
           'n_reasoning', 'cls_name', 'abs_bin', 'rel_pos')
DTYPES = {'sentence_index': 'int32', 'cls': 'int8', 'segment': 'int16', 'tok_start': 'int32', 'tok_end': 'int32',
          'n_tokens': 'int32', 'has_prev_labelled': 'bool', 'has_next_labelled': 'bool',
          'prev_same_segment': 'bool', 'next_same_segment': 'bool', 'abs_bin': 'int8', 'rel_pos': 'float64'}


def pairs_dir(model: str) -> Path:
    """Private bulk directory of this model's transition-pair work (tables, judge stores, merged labels)."""
    return Path(binding()['config']['bulk_root'])/'transition-pairs'/model


def receipt_dir(model: str) -> Path:
    """Compact receipts of this model's transition-pair work, inside the final-routing campaign report."""
    return ROOT/'transition-pairs'/model


def campaign_sources(model: str) -> list[dict]:
    """The 1,547 attempts (source trace location, trace digest, family, completion length) of a campaign-v3 model."""
    sources = load(CAMPAIGN/model/'SOURCES.json')['records']
    require(len(sources) == 1547, f'{model}: expected 1,547 campaign attempts')
    return sources


def trace_key(source: dict) -> tuple:
    """Key joining an attempt to its judge records."""
    return (source['benchmark'], source['problem_id'], int(source['sample_id']), source['trace_sha256'])


def label_files(model: str) -> list[Path]:
    """The four judge-record files (25,000 sentences each) of a campaign-v3 model."""
    return sorted((SOURCE_BASE/'campaign-v3/labels'/model).glob('part-*/annotations.json'))


def original_label_records(model: str, keep_inputs: tuple[str, ...] = PAIR_SOURCES) -> dict[tuple, list[tuple]]:
    """Original judge records by trace: key -> [(sentence index, start, end, label or None, judge inputs or None)].

    The judge inputs are kept only for sentences of the classes in `keep_inputs` (the pair completion audits
    the contexts it rebuilds against them); the rest would only cost memory. A record without a valid label
    (an `unknown` outcome) has label None.
    """
    records = defaultdict(list)
    for path in label_files(model):
        for row in json.loads(path.read_text()):
            identity, label = row['identity'], row.get('label')
            label = label if label in CLASSES else None
            key = (identity['dataset'], identity['problem_id'], int(identity['sample_id']), identity['trace_sha256'])
            records[key].append((int(identity['sentence_index']), int(identity['start']), int(identity['end']), label,
                                 row['inputs'] if label in keep_inputs else None))
    return records


def usable_labels(units: list[dict], records: Iterable[tuple]) -> tuple[dict[int, str], dict[int, tuple]]:
    """Judge labels that still match the current sentence segmentation, and every judged span.

    A record counts when its index exists and its offsets equal the current sentence's; an invalid label is
    skipped and conflicting labels of one sentence are both removed (the extraction's rules). The second
    result maps every sentence with a record at the right span to that span, labelled or not, so a sentence
    the judge already saw (even with an unknown outcome) is never judged again.
    """
    labels, conflicting, judged = {}, set(), {}
    for index, start, end, label, *_ in records:
        if not 0 <= index < len(units) or (units[index]['start'], units[index]['end']) != (start, end):
            continue
        judged[index] = (start, end)
        if label is not None and labels.setdefault(index, label) != label:
            conflicting.add(index)
    for index in conflicting:
        del labels[index]
    return labels, judged


def attempt_layout(source: dict) -> dict:
    """Sentences, reasoning-token ownership and segments of one saved trace.

    Returns a dict with `value` (the raw trace line), `units`, `segments` (reasoning range of each sentence),
    `tokens` and `ranks` (completion positions and reasoning ranks owned by each sentence) and `n_reasoning`.
    """
    from moe_exp.research_campaign_v3_r6.inputs import read_source
    from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
    from moe_exp.correlation_pipeline.spans import reasoning_ranges, trace_digest
    from moe_exp.schemas import TraceRecord
    value = read_source(source['source_location'])
    trace = TraceRecord(**value)
    require(trace_digest(trace) == source['trace_sha256'], f"{source['attempt_id']}: trace digest differs")
    layout = saved_layout(trace)
    require(layout['reasoning_tokens'] is not None, f"{source['attempt_id']}: saved trace lacks token offsets")
    ranges = reasoning_ranges(trace)
    rank_of = {token: rank for rank, token in enumerate(layout['reasoning_tokens'])}
    tokens = [owned or [] for owned in layout['unit_tokens']]
    return {'value': value, 'units': layout['units'], 'tokens': tokens, 'n_reasoning': len(rank_of),
            'ranks': [[rank_of[t] for t in owned] for owned in tokens],
            'segments': [next(j for j, (a, b) in enumerate(ranges) if a <= u['start'] < b) for u in layout['units']]}


def sentence_extent(layout: dict, index: int, limit: int | None = None) -> dict | None:
    """Token count and reasoning-rank extent of one sentence, or None when it owns no (captured) token.

    `limit` drops completion positions at or beyond it, for a trace whose replay captured only a prefix.
    """
    ranks = [r for t, r in zip(layout['tokens'][index], layout['ranks'][index]) if limit is None or t < limit]
    if not ranks:
        return None
    return {'n_tokens': len(ranks), 'tok_start': min(ranks), 'tok_end': max(ranks)+1}


def table_rows(source: dict, layout: dict, labels: dict[int, str]) -> list[dict]:
    """The table rows of one attempt: every labelled sentence that owns a reasoning token."""
    extents = {i: sentence_extent(layout, i) for i in sorted(labels)}
    present = {i: e for i, e in extents.items() if e is not None}
    segment = layout['segments']
    rows = []
    for index, extent in present.items():
        before, after = index-1 in present, index+1 in present
        rows.append({
            'attempt_id': source['attempt_id'], 'question': source['question'], 'dataset': source['benchmark'],
            'sentence_index': index, 'cls': CLASSES.index(labels[index]), 'segment': segment[index], **extent,
            'has_prev_labelled': before, 'has_next_labelled': after,
            'prev_same_segment': before and segment[index-1] == segment[index],
            'next_same_segment': after and segment[index+1] == segment[index],
            'n_reasoning': layout['n_reasoning'], 'cls_name': labels[index],
            'abs_bin': int(np.searchsorted(ABSOLUTE_BIN_EDGES, extent['tok_start'], side='right')),
            'rel_pos': extent['tok_start']/layout['n_reasoning']})
    return rows


def _attempt_rows(task: tuple) -> list[dict]:
    """Worker: table rows of one attempt from its judge records."""
    source, records = task
    layout = attempt_layout(source)
    labels, _ = usable_labels(layout['units'], records)
    return table_rows(source, layout, labels)


def as_frame(rows: list[dict]) -> pd.DataFrame:
    """Rows in the extraction's column order and dtypes, sorted by attempt and sentence."""
    frame = pd.DataFrame(rows, columns=COLUMNS).sort_values(['attempt_id', 'sentence_index'], ignore_index=True)
    return frame.astype(DTYPES)


def campaign() -> dict:
    """The campaign-v3 binding that names every saved input of the seven models."""
    return load(CAMPAIGN/'CAMPAIGN.json')


def check_class_counts(frame: pd.DataFrame, model: str, sources: list[dict]) -> int:
    """Require the class token totals of every attempt to equal the model's saved class counts.

    The saved counts are the 7 class totals, the reasoning tokens and the labelled tokens of
    `class_counts.npz`, which the full-A token-ownership receipt already reproduced from the same labels.
    The reasoning length is compared for attempts that have a row; the others have no sentence to carry it.

    Returns:
        The number of attempts checked.
    """
    with np.load(campaign()['saved_inputs'][model]['counts'], allow_pickle=False) as archive:
        require(list(archive['classes']) == [*CLASSES, 'ALL_REASONING', 'ALL_LABELLED'], 'class order differs')
        saved = dict(zip(archive['attempt_id'].tolist(), archive['tokens'].astype(np.int64).tolist()))
    totals = frame.groupby(['attempt_id', 'cls']).n_tokens.sum().unstack(fill_value=0)
    totals = totals.reindex(columns=range(len(CLASSES)), fill_value=0)
    reasoning = frame.groupby('attempt_id').n_reasoning.first()
    for source in sources:
        attempt = source['attempt_id']
        classes = totals.loc[attempt].tolist() if attempt in totals.index else [0]*len(CLASSES)
        expected = saved[attempt]
        require(classes == expected[:7] and sum(classes) == expected[8], f'{model}: class tokens differ for {attempt}')
        require(attempt not in reasoning.index or int(reasoning[attempt]) == expected[7],
                f'{model}: reasoning tokens differ for {attempt}')
    return len(sources)


def compare_with_existing(frame: pd.DataFrame, model: str) -> dict:
    """Rows of the attempts that the extraction's own table also holds must be identical."""
    existing = pd.read_parquet(campaign()['saved_inputs'][model]['sentences'])
    shared = frame[frame.attempt_id.isin(set(existing.attempt_id))].reset_index(drop=True)
    old = existing.sort_values(['attempt_id', 'sentence_index'], ignore_index=True)
    equal = shared.shape == old.shape and all(
        shared[c].to_numpy().tolist() == old[c].to_numpy().tolist() for c in COLUMNS)
    return {'attempts_in_existing_table': int(existing.attempt_id.nunique()), 'rows_in_existing_table': len(existing),
            'rows_rebuilt_for_those_attempts': len(shared), 'identical': bool(equal)}


def rebuild(model: str, workers: int, limit: int | None = None, output: Path | None = None) -> dict:
    """Rebuild a model's labelled-sentence table for all of its attempts and publish the receipt.

    Args:
        model: A campaign-v3 model key.
        workers: Processes reading traces and laying out sentences (the work is CPU-bound per trace).
        limit: Debug only: the first `limit` attempts; no receipt is published and no count check is made.
        output: Parquet path; defaults to this model's private transition-pairs directory.

    Returns:
        The published SENTENCES.json receipt (counts, checks, hashes), or a summary dict when `limit` is set.
    """
    require(not limit or output, '--limit writes a partial table: name an --output')
    sources = campaign_sources(model)
    records = original_label_records(model)
    selected = sources[:limit] if limit else sources
    tasks = [(s, records.get(trace_key(s), [])) for s in selected]
    with multiprocessing.get_context('fork').Pool(workers) as pool:
        rows = [row for part in pool.imap_unordered(_attempt_rows, tasks, chunksize=4) for row in part]
    frame = as_frame(rows)
    output = Path(output) if output else pairs_dir(model)/'sentences.parquet'
    require(not output.exists(), f'never overwritten: {output}')
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = output.with_name('.pending-'+output.name)
    frame.to_parquet(temporary, index=False)
    temporary.rename(output)
    summary = {'attempts': int(frame.attempt_id.nunique()), 'questions': int(frame.question.nunique()),
               'sentences': len(frame), 'path': str(output)}
    if limit:
        return summary
    attempts = set(frame.attempt_id)
    families = {s['family'] for s in sources if s['attempt_id'] in attempts}
    return publish(receipt_dir(model)/'SENTENCES.json', {
        **summary, 'model': model, 'families': len(families), 'sentences_sha256': file_sha(output),
        'class_token_counts_checked_attempts': check_class_counts(frame, model, sources),
        'existing_table': compare_with_existing(frame, model),
        'class_sentences': frame.groupby('cls_name').size().to_dict(),
        'label_files': {str(p): file_sha(p) for p in label_files(model)},
        'sources_sha256': file_sha(CAMPAIGN/model/'SOURCES.json'), 'campaign_sha256': campaign()['sha256'],
        'module_sha256': file_sha(Path(__file__))})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog='moe_exp.study.sentence_table', description=__doc__.split('\n\n')[0])
    parser.add_argument('command', choices=['rebuild'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    parser.add_argument('--limit', type=int, help='debug: first N attempts, no receipt')
    parser.add_argument('--output', type=Path, help='debug: parquet path')
    args = parser.parse_args(argv)
    print(json.dumps(rebuild(args.model, args.workers, args.limit, args.output), indent=1, default=str))


if __name__ == '__main__':
    main()
