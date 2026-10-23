"""Native attempts with their routing saved: the data the class-conditioned selection learns from.

A capture release asks every model the same 300 questions six times each, with unedited routing,
and keeps, for every reasoning token, the native top-k expert ids and their weights at every routed
layer (``capture_mode: native_ids``). The six attempts per question are also the unedited arm of the
later nudge releases, so nothing is generated twice.

- ``roster()`` fixes the questions: a fixed number per benchmark, drawn by a digest over the
  1,547-question pool every model shares, one question per family. It never reads an outcome.
- ``capture_manifest(model, spec)`` is called by ``moe_exp.study.whole`` prepare when the release
  holds ``CAPTURE.json``; it publishes the model's manifest and its SELECTED receipt.
"""
from __future__ import annotations

from moe_exp.final_routing.common import digest, load, publish, require

from . import REPORTS

# Questions per benchmark: every AIME/AMC question (where right and wrong attempts of the same
# question are common), plus a stratified share of the larger benchmarks.
PER_BENCHMARK = {'aime24': 30, 'aime25': 30, 'amc23': 40, 'olympiad': 100, 'minerva': 50, 'math500': 50}
ATTEMPTS = 6                 # native attempts per question
CAP_TOKENS = 32768           # completion allowance of every row (the card-v3 generation budget)
FOLDS = REPORTS/'research-campaign-v3-20261014-r6/FOLDS.json'
SOURCES = {'lfm25': REPORTS/'final-routing-20261015-r1/lfm25/SOURCES.json'}
SOURCE_CAMPAIGN = REPORTS/'research-campaign-v3-20261014-r6'
ROSTER_TAG = 'outcome-capture-roster-v1'


def sources(model: str) -> dict[str, dict]:
    """One source record per question (dataset|problem) for a model; its prompt tokens are the original."""
    path = SOURCES.get(model, SOURCE_CAMPAIGN/model/'SOURCES.json')
    return {r['question']: r for r in load(path)['records']}


def roster(models: list[str]) -> list[str]:
    """The questions every model is asked, identical for all models and independent of any outcome.

    Questions are ranked within each benchmark by digest([ROSTER_TAG, question]); a question whose
    family was already taken is skipped, so each family appears once.
    """
    folds = load(FOLDS)
    pools = [set(sources(m)) for m in models]
    shared = set.intersection(*pools)
    chosen, families = [], set()
    for benchmark, count in sorted(PER_BENCHMARK.items()):
        ranked = sorted((q for q in shared if q.split('|')[0] == benchmark), key=lambda q: digest([ROSTER_TAG, q]))
        taken = 0
        for question in ranked:
            family = folds['question_families'][question]
            if taken == count or family in families:
                continue
            chosen.append(question)
            families.add(family)
            taken += 1
        require(taken == count, f'only {taken} of {count} {benchmark} questions available')
    return sorted(chosen)


def capture_rows(model: str, spec: dict, profile: dict, prompts: dict[str, list[int]]) -> list[dict]:
    """Six native rows per roster question, each with its own seed and the native-id capture.

    Args:
        model: Study model.
        spec: CAPTURE.json entry: questions, attempts, cap_tokens.
        profile: The model's binding profile.
        prompts: Question -> original prompt token ids.
    """
    from moe_exp.final_routing.runtime import remaining
    folds = load(FOLDS)
    table = sources(model)
    rows = []
    for question in spec['questions']:
        source = table[question]
        family = folds['question_families'][question]
        prompt = prompts[question]
        allowance = min(remaining(profile, prompt, []), int(spec['cap_tokens']))
        for attempt in range(int(spec['attempts'])):
            rows.append({
                'model': model, 'family': family, 'benchmark': question.split('|')[0],
                'fold': int(folds['folds'][family]), 'question': question, 'attempt': attempt,
                'seed': int(digest([model, question, 'capture-native', attempt])[:8], 16),
                'profile_sha256': profile['profile_sha256'],
                'source': {'attempt_id': source['attempt_id'], 'prefix_tokens': 0, 'question': question},
                'original_prompt': True, 'prompt_sha256': digest(prompt), 'whole': True, 'prompt_scope': True,
                'pulse_tokens': 0, 'horizon': allowance, 'total_allowance': allowance, 'cap_tokens': allowance,
                'prefix_status': 'enrolled', 'capture_layers': profile['layers'], 'action': None, 'dose': 'main',
                'depth_third': None, 'arm': 'native', 'class': None, 'operation': 'native', 'actions': [],
                'nomination_status': 'nominated', 'audit_capture': False, 'capture_mode': 'native_ids'})
    for row in rows:
        row['uid'] = digest(['outcome-capture-v1', row])
    require(len({r['seed'] for r in rows}) == len(rows), 'capture seed collision')
    return rows


def capture_manifest(model: str, spec: dict) -> dict:
    """Publish one model's capture manifest and select it (no canary: native routing only)."""
    from moe_exp.study import driver
    from moe_exp.study.runtime import prompt_prefix
    from moe_exp.study.whole import ROOT, SWEEP, manifest_path
    dose = SWEEP[model]
    b = driver.binding()
    profile = b['models'][model]
    table = sources(model)
    prompts = {}
    for question in spec['questions']:
        row = {'original_prompt': True, 'source': {'attempt_id': table[question]['attempt_id'], 'prefix_tokens': 0}}
        prompt, prefix, _ = prompt_prefix(b, model, row, None)
        require(not prefix, 'original prompt row retained a prefix')
        prompts[question] = prompt
    rows = capture_rows(model, spec, profile, prompts)
    expected = len(spec['questions'])*int(spec['attempts'])
    require(len(rows) == expected and len({r['uid'] for r in rows}) == expected, 'capture row count/UID differs')
    value = publish(manifest_path(model, dose), {
        'model': model, 'dose': dose, 'design': 'outcome-capture', 'scope': 'native routing, ids captured',
        'cells': [], 'records': rows, 'planned': expected, 'executable': expected, 'audit_units': 0,
        'questions': spec['questions'], 'outcomes_used_for_selection': False})
    publish(ROOT/model/'SELECTED.json', {'dose': dose, 'manifest_sha256': value['sha256'], 'canary_job': None,
                                         'sweep': True, 'capture': True})
    return {key: value[key] for key in ('sha256', 'planned', 'executable', 'scope')}
