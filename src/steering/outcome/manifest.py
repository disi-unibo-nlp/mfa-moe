"""Freeze the rows of an outcome-steering release (called by steering.whole prepare).

Each evaluation family of the base whole-reasoning release gets:

- ``target_nudge`` and ``random_nudge`` at every delta: the selected experts (or their matched random
  control) have ``delta`` subtracted from their router logits at every prompt and reasoning token;
- ``budget`` rows at every fraction: the family's finished native run is cut at that fraction of its
  reasoning, the closing suffix is appended, and only the answer is generated (native routing).

Native and native-repeat rows are not rerun: the base release already ran them with the same prompts,
seeds and allowances, and the analysis reads them from there.
"""
from __future__ import annotations

import math
from pathlib import Path

from store import digest, file_sha, load, publish, require, verified

# GPT-OSS closes its analysis channel with this full harmony transition; the runtime closure only
# matches its last three tokens (<|channel|>final<|message|>).
GPT_TRANSITION = [200007, 200006, 173781, 200005, 17196, 200008]


def evaluation_questions(model: str, spec: dict) -> set[str]:
    """Question identities (dataset|source_problem_id) of the base release's evaluation families."""
    return {row['source']['question'] for row in base_natives(model, spec)}


def base_natives(model: str, spec: dict) -> list[dict]:
    """Native rows of the base whole-reasoning manifest (one per evaluation family)."""
    from steering.whole import manifest_path
    base = verified(manifest_path(model, spec['base_dose'], Path(spec['base'])))
    rows = sorted((r for r in base['records'] if r['arm'] == 'native'), key=lambda r: r['family'])
    require(rows and len({r['family'] for r in rows}) == len(rows), 'base native roster differs')
    return rows


def default_spec(model: str, reports: Path) -> dict:
    """Where each model's base natives ran: its base release, at the dose that release selected, plus
    the top-ups and resumes that finished them. GPT-OSS's base release is r8; every other model's is r7.

    Args:
        reports: The report/ folder of the working repository.
    """
    release = {name: str(reports/f'whole-reasoning-20261021-{name}') for name in ('r7', 'r8', 'r11', 'r15')}
    base = release['r8'] if model == 'gpt' else release['r7']
    dose = load(Path(base)/model/'SELECTED.json')['dose']
    return {'base': base, 'base_dose': dose, 'stores': [base, release['r11'], release['r15']]}


def nudge_actions(pairs: list[dict], delta: float) -> list[dict]:
    """One score_shift action per layer that holds nudged experts."""
    by_layer = {}
    for pair in pairs:
        by_layer.setdefault(pair['layer'], []).append(pair['expert'])
    return [{'layer': layer, 'activate': [], 'deactivate': [], 'shift': sorted(experts), 'delta': float(delta),
             'operator': 'score_shift', 'epsilon': 0., 'pulse_tokens': 0}
            for layer, experts in sorted(by_layer.items())]


def finished_output(model: str, spec: dict, native: dict) -> Path | None:
    """OUTPUT.json of the store where this native row finished, or None if it never finished."""
    for root in spec['stores']:
        folder = Path(root)/'stores'/model/spec['base_dose']/'units'/native['uid']
        if (folder/'RESULT.json').exists() and (folder/'OUTPUT.json').exists():
            return folder/'OUTPUT.json'
    return None


def reasoning_length(tokens: list[int], closures: list[list[int]]) -> tuple[int, bool]:
    """Position where the reasoning closes, or (all tokens, True) if it never closed (capped run)."""
    starts = [i for seq in closures for i in range(len(tokens)-len(seq)+1) if tokens[i:i+len(seq)] == seq]
    return (min(starts), False) if starts else (len(tokens), True)


def outcome_manifest(root: Path, model: str, spec: dict) -> dict:
    """Publish this release's selection, manifest and SELECTED receipt for one model.

    Args:
        root: The folder of this release.
        model: Study model.
        spec: OUTCOME.json entry: base release/dose, stores holding its natives, deltas, budget
            fractions and the answer allowance of budget rows.
    """
    from transformers import AutoTokenizer
    from steering import pulse as driver
    from steering.runtime import closure, prompt_prefix, remaining
    from steering.whole import manifest_path, sweep_doses
    from steering.outcome.select import selection

    dose = sweep_doses(root)[model]
    natives = base_natives(model, spec)
    chosen = publish(root/model/'OUTCOME_SELECTION.json',
                     selection(model, evaluation_questions(model, spec), root.parent))
    b = driver.binding()
    profile = b['models'][model]
    tokenizer = AutoTokenizer.from_pretrained(profile['local_path'], local_files_only=True)
    closures = closure(tokenizer, model)
    suffix = GPT_TRANSITION if model == 'gpt' else list(closures[0])
    require(any(suffix[-len(seq):] == list(seq) for seq in closures), 'budget suffix does not close the reasoning')

    rows = []
    for native in natives:
        expert_set = native['source']['question'] if native['source']['question'] in chosen['sets'] else 'all'
        chosen_set = chosen['sets'][expert_set]
        base = {**{k: v for k, v in native.items() if k not in ('uid', 'audit_capture')},
                'dose': dose, 'native_dose': spec['base_dose'], 'native_uid': native['uid'],
                'expert_set': expert_set, 'audit_capture': False, 'nomination_status': 'nominated'}
        for delta in spec['deltas']:
            for role, pairs in (('target', chosen_set['targets']), ('random', chosen_set['controls'])):
                rows.append({**base, 'arm': role+'_nudge', 'operation': 'nudge', 'class': f'delta-{delta}',
                             'actions': nudge_actions(pairs, delta), 'delta': float(delta)})
        output = finished_output(model, spec, native)
        tokens = load(output)['token_ids'] if output else []
        length, capped = reasoning_length(tokens, closures) if output else (0, False)
        prompt, _, _ = prompt_prefix(b, model, native, tokenizer)
        for fraction in spec['fractions']:
            cut = math.floor(fraction*length)
            row = {**base, 'arm': 'budget', 'operation': 'budget', 'class': f'fraction-{fraction}', 'actions': [],
                   'budget_fraction': float(fraction), 'budget_cut': cut, 'source_reasoning_tokens': length,
                   'source_capped': capped}
            if output is None or cut < 1:
                rows.append({**row, 'nomination_status': 'source_unavailable'})
                continue
            prefix = {'path': str(output), 'sha256_file': file_sha(output), 'cut': cut, 'suffix': suffix}
            allowance = min(int(spec['answer_tokens']), remaining(profile, prompt, tokens[:cut]+suffix))
            rows.append({**row, 'budget_prefix': prefix, 'horizon': allowance, 'total_allowance': allowance})
    for row in rows:
        row['uid'] = digest(['outcome-v1', chosen['sha256'], row])
    expected = len(natives)*(2*len(spec['deltas'])+len(spec['fractions']))
    require(len(rows) == expected and len({r['uid'] for r in rows}) == expected, 'outcome row count/UID differs')
    value = publish(manifest_path(model, dose, root), {
        'model': model, 'dose': dose, 'design': 'outcome-nudge', 'scope': 'prompt-and-generated-reasoning',
        'selection_sha256': chosen['sha256'], 'native_manifest': spec['base'], 'native_dose': spec['base_dose'],
        'cells': [], 'records': rows, 'planned': expected,
        'executable': sum(r['nomination_status'] == 'nominated' for r in rows), 'audit_units': 0,
        'outcomes_used_for_selection': 'cohort-B attempts only; evaluation families held out by question'})
    publish(root/model/'SELECTED.json', {'dose': dose, 'manifest_sha256': value['sha256'], 'canary_job': None,
                                         'sweep': True, 'outcome': True})
    return {key: value[key] for key in ('sha256', 'planned', 'executable', 'scope')}
