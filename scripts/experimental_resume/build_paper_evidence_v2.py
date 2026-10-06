"""Append completed clean R3-D/B1 claims to the immutable first paper snapshot.

This is a bounded saved-result packaging step. The original ledger and figure
snapshot remain unchanged; v2 records the newly completed predictive analyses.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess


REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
DOC = REPO / 'report/experimental-resume-v1'
R3D = ROOT / 'forum/tests/r3_context'
B1 = ROOT / 'steering-v1/runs/resume-v1/r3e-b1-clean-v2/result.json'
PRIMARY = ('next', 'next-contained', 'switch', 'switch-contained', 'dest', 'dest-contained')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(f'broken saved-result seal: {path}')
    return value


def claim(identifier, population, intervention, estimate, uncertainty, family,
          status, source, note):
    return {'id': identifier, 'population': population,
            'intervention': intervention, 'estimate': estimate,
            'uncertainty': uncertainty, 'multiplicity_family': family,
            'status': status, 'source': str(source), 'interpretation': note}


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    answer = [None] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        answer[index] = min(1.0, running)
    return answer


def slurm_record(job_id):
    lines = subprocess.run(
        ['sacct', '-j', str(job_id), '-X', '-P', '-n', '-o',
         'JobID,State,ExitCode,Elapsed,End'],
        check=True, capture_output=True, text=True).stdout.strip().splitlines()
    if len(lines) != 1:
        raise ValueError(f'cannot verify Slurm job {job_id}: {lines}')
    return dict(zip(('job_id', 'state', 'exit_code', 'elapsed', 'end'),
                    lines[0].split('|')))


def main():
    old_ledger_path = DOC / 'CLAIM_LEDGER.json'
    old_pointer_path = DOC / 'PAPER_SNAPSHOT.json'
    ledger_path = DOC / 'CLAIM_LEDGER_v2.json'
    snapshot_path = DOC / 'PAPER_SNAPSHOT_v2.json'
    if ledger_path.exists() or snapshot_path.exists():
        raise FileExistsError('v2 evidence package already exists')
    old = json.loads(old_ledger_path.read_text())
    if not isinstance(old, list) or len(old) != 209:
        raise ValueError('historical claim ledger inventory changed')
    pointer = json.loads(old_pointer_path.read_text())
    old_snapshot = Path(pointer['path'])
    if not (old_snapshot / 'FROZEN.json').is_file():
        raise ValueError('first immutable paper snapshot missing')

    readout_path = R3D / 'readout-results.v3.json'
    anticipation_path = R3D / 'anticipation-results.json'
    readout, anticipation, b1 = sealed(readout_path), sealed(anticipation_path), sealed(B1)
    audit = sealed(DOC / 'R3D_MERGED_INVENTORY_AUDIT_v2.json')
    final = sealed(DOC / 'R3D_FINALIZATION_AFTER_MERGE_v1.json')
    retirement = sealed(DOC / 'R3D_GUARD_CHAIN_RETIREMENT_v1.json')
    if (audit['status'] != 'PASS' or audit['fit_total'] != 1650 or
        final['result_sha256'] != anticipation['sha256'] or
        retirement['frozen_result_sha256'] != anticipation['sha256'] or
        retirement['status'] != 'RETIRED_ZERO_CPU' or
        b1['status'] != 'COMPLETE' or len(readout['models']) != 7):
        raise ValueError('R3-D/B1 completion binding differs')
    deactivation = slurm_record(59205653)
    positive_screen = slurm_record(59204242)
    held = subprocess.run(['scontrol', 'show', 'job', '59204242'], check=True,
                          capture_output=True, text=True).stdout
    if (deactivation['state'] != 'CANCELLED by 133943' or
        deactivation['elapsed'] != '00:00:00' or
        positive_screen['state'] != 'PENDING' or
        positive_screen['elapsed'] != '00:00:00' or
        not re.search(r'\bReason=JobHeldUser\b', held)):
        raise ValueError('deactivation cancellation or positive-screen hold changed')

    additions = []
    readout_table = {}
    for model, cells in readout['models'].items():
        readout_table[model] = {}
        for mode in ('token_identity', 'judge_context'):
            cell = cells[mode]
            if cell['status'] != 'CLEAN':
                raise ValueError(f'incomplete seven-model readout: {model}/{mode}')
            routing = cell['routing']
            excess = cell['routing_excess_over_lexical_null']
            population = (f"{cell['n_questions']} exact-ID-clean dev+tune questions; "
                          f"{cell['n_families']} frozen duplicate families; "
                          f"{cell['n_sentences']} sentences")
            additions.append(claim(
                f'R3D:S3:v2:{model}:{mode}', population,
                f'observational routing features added to {mode} baseline',
                routing['P'], routing['simultaneous_ci7'],
                'seven-model Bonferroni percentile family within this baseline mode',
                'CLEAN_OBSERVATIONAL_PREDICTIVE_GAIN', readout_path,
                'The completed seven-model clean family supersedes earlier incomplete-cell status. '
                'Held-out routing association does not establish causal steering; '
                'judge-context control is retrospective, not an online detector.'))
            if mode == 'judge_context':
                additions.append(claim(
                    f'R3D:S3:v2:{model}:{mode}:lexical_excess', population,
                    'routing gain minus class-within-token/position lexical-null gain',
                    excess['P'], excess['simultaneous_ci7'],
                    'seven-model Bonferroni percentile family; separately reported excess contrast',
                    'CLEAN_OBSERVATIONAL_EXCESS_OVER_LEXICAL_NULL', readout_path,
                    'Retrospective judge-context control and within-stratum null; '
                    'not a deployable class trigger or causal effect.'))
            readout_table[model][mode] = {
                'questions': cell['n_questions'], 'families': cell['n_families'],
                'routing_P': routing['P'], 'routing_simultaneous_CI7': routing['simultaneous_ci7'],
                'lexical_excess_P': excess['P'],
                'lexical_excess_simultaneous_CI7': excess['simultaneous_ci7']}

    gpt = anticipation['models']['gpt']
    pvalues = [gpt[tag]['routing']['composition16' if tag.startswith('next') else 'k16']['p_bootstrap']
               for tag in PRIMARY]
    adjusted = holm(pvalues)
    primary_table = {}
    for index, tag in enumerate(PRIMARY):
        response = 'A1 next class' if tag.startswith('next') else ('K2 switch' if tag.startswith('switch') else 'K2 destination')
        measure = 'composition16' if tag.startswith('next') else 'k16'
        cell = gpt[tag]
        result = cell['routing'][measure]
        if tag == 'next':
            status = 'CLEAN_OBSERVATIONAL_ADVANCES_PREDICTIVE_GATE'
        elif tag == 'dest':
            status = 'CLEAN_OBSERVATIONAL_IMPRECISE_SENSITIVITIES'
        elif tag in ('switch', 'switch-contained'):
            status = 'CLEAN_OBSERVATIONAL_SWITCH_PRACTICAL_GATE_FAIL'
        else:
            status = 'CLEAN_OBSERVATIONAL_CONTAINED_SENSITIVITY'
        population = (f"GPT exact-ID-clean A; {cell['questions']} questions, "
                      f"{cell['pairs']} adjacent sentence pairs; frozen family folds")
        additions.append(claim(
            f'R3D:{response.split()[0]}:v2:gpt:{tag}:{measure}', population,
            'observational routing augmentation beyond past text/history and training-only current-class estimate',
            result['P'], result['simultaneous_ci6'],
            f'six GPT primary contrasts; Bonferroni percentile CI6; Holm bootstrap p={adjusted[index]:.6g}',
            status, anticipation_path,
            f"P is candidate-minus-baseline loss in nats per pair; negative favors routing. "
            f"The registered advancement threshold is P≤−0.005; {result['gain_positive_repeats']}/5 repeats favor routing. "
            'An observational prediction improvement cannot identify useful causal routing action.'))
        primary_table[tag] = {'response': response, 'measure': measure,
                              'questions': cell['questions'], 'pairs': cell['pairs'],
                              'P': result['P'], 'simultaneous_CI6': result['simultaneous_ci6'],
                              'bootstrap_p': pvalues[index], 'Holm_p_six_primary': adjusted[index],
                              'positive_gain_repeats': result['gain_positive_repeats']}

    qwen_table = {}
    for tag, measure in (('next', 'composition16'), ('next-contained', 'composition16'),
                         ('dest', 'k16')):
        cell = anticipation['models']['qwen36'][tag]
        result = cell['routing'][measure]
        qwen_table[tag] = {'measure': measure, 'questions': cell['questions'],
                           'pairs': cell['pairs'], 'P': result['P'],
                           'simultaneous_CI6': result['simultaneous_ci6']}
        additions.append(claim(
            f'R3D:replication:v2:qwen36:{tag}:{measure}',
            f"Qwen3.6 exact-ID-clean A; {cell['questions']} questions, {cell['pairs']} adjacent pairs",
            'observational same-direction routing augmentation beyond past text/history',
            result['P'], result['simultaneous_ci6'],
            'model-specific descriptive replication; CI6 from saved driver',
            'SAME_DIRECTION_REPLICATION' if result['simultaneous_ci6'][1] < 0 else 'SAME_DIRECTION_IMPRECISE_REPLICATION',
            anticipation_path,
            'Replication is population-specific. GPT destination contained and Qwen destination intervals span zero; '
            'same direction does not establish a transportable causal controller.'))

    antic = gpt['next']['routing']['composition_antic']
    additions.append(claim(
        'R3D:A1:v2:gpt:next:composition_antic',
        f"GPT exact-ID-clean A; {gpt['next']['questions']} questions, {gpt['next']['pairs']} pairs",
        'secondary observational anticipation-composition augmentation',
        antic['P'], antic['simultaneous_ci6'],
        'secondary profile; CI6 descriptive and not an additional primary contrast',
        'BELOW_PRACTICAL_GATE_REPLICATION_INCOMPLETE', anticipation_path,
        'P does not reach −0.005, and saved Qwen A1_antic replication is marked incomplete.'))

    clustered = b1['family_clustered']
    lexical = b1['result']['lexical']
    additions.append(claim(
        'R3E:B1:v2:gpt:clean',
        f"{clustered['questions']} exact-ID-clean GPT A questions; {clustered['families']} frozen families; "
        f"{b1['population']['labelled_tokens']} labelled tokens before the frequent-ID filter",
        'observational class-associated routing gain after clean grouped fold refit',
        clustered['estimate_nats_per_token_pair'], clustered['family_cluster_ci95'],
        'family-clustered 5,000-bootstrap nominal 95% interval; five CV repeats',
        'CLEAN_OBSERVATIONAL_PREDICTIVE_GAIN', B1,
        f"All five repeats favor routing. The frequent-ID filter drops {lexical['tokens_dropped']} "
        f"tokens ({lexical['fraction_dropped']:.1%}); no correctness outcome was read. "
        'This is not a causal steering effect or endpoint accuracy improvement.'))

    ids = [row['id'] for row in old + additions]
    if len(ids) != len(set(ids)):
        raise ValueError('claim ID collision with historical ledger')
    ledger = old + additions
    ledger_path.write_text(json.dumps(ledger, indent=1, ensure_ascii=False) + '\n')
    source_paths = [old_ledger_path, old_pointer_path, old_snapshot / 'FROZEN.json',
                    readout_path, anticipation_path, B1,
                    DOC / 'R3D_MERGED_INVENTORY_AUDIT_v2.json',
                    DOC / 'R3D_FINALIZATION_AFTER_MERGE_v1.json',
                    DOC / 'R3D_GUARD_CHAIN_RETIREMENT_v1.json']
    body = {
        'schema': 'resume-paper-evidence-addendum-v2',
        'status': 'SAVED_RESULT_SNAPSHOT_NO_NEW_CAUSAL_CLAIM',
        'prior_immutable_figure_snapshot': str(old_snapshot),
        'prior_snapshot_pointer_sha256': file_sha(old_pointer_path),
        'source_file_sha256': {str(path): file_sha(path) for path in source_paths},
        'claim_ledger_v2': str(ledger_path),
        'claim_ledger_v2_file_sha256': file_sha(ledger_path),
        'claim_count_prior': len(old), 'claim_count_added': len(additions),
        'claim_count_total': len(ledger),
        'r3d': {
            'seven_model_readout': readout_table,
            'six_GPT_primary_anticipation': primary_table,
            'Qwen_same_direction_sensitivities': qwen_table,
            'merged_fit_count': audit['fit_total'],
            'frozen_result_sha256': anticipation['sha256'],
            'finalization_receipt_sha256': final['sha256'],
            'obsolete_guard_chain_retirement_sha256': retirement['sha256'],
            'scope': 'observational held-out prediction; no causal routing intervention'},
        'clean_B1': {
            'estimate_nats_per_token_pair': clustered['estimate_nats_per_token_pair'],
            'family_cluster_ci95': clustered['family_cluster_ci95'],
            'positive_repeats': b1['result']['observed']['positive_repeats'],
            'questions': clustered['questions'], 'families': clustered['families'],
            'labelled_tokens': b1['population']['labelled_tokens'],
            'frequent_ID_tokens_dropped': lexical['tokens_dropped'],
            'frequent_ID_fraction_dropped': lexical['fraction_dropped'],
            'source_result_sha256': b1['sha256'],
            'scope': 'observational class-associated routing; no correctness effect'},
        'still_missing': [
            'causal trajectory-control and accuracy-token endpoint estimates',
            'legacy X3 semantic and endpoint results',
            'independent source-grounded semantic validation beyond available LLM audits'],
        'figure_status': 'prior immutable figures retained; no R3-D/B1 figure regenerated in this addendum',
        'operational_job_state_at_snapshot': {
            'invalid_deactivation_cancelled_zero_work': deactivation,
            'positive_micro_screen_held_pending_review': positive_screen,
            'generator_model': 'Qwen3.6',
            'independent_semantic_reader_model': 'Qwen3.8'},
    }
    snapshot_path.write_text(json.dumps({**body, 'sha256': digest(body)},
                                        indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'snapshot': str(snapshot_path), 'sha256': digest(body),
                      'ledger': str(ledger_path), 'prior_claims': len(old),
                      'added_claims': len(additions)}))


if __name__ == '__main__':
    main()
