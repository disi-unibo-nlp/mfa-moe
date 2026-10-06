"""Record manual findings on a small, sealed sample of saved completions.

This is a qualitative readout, not a judge or an estimate of incidence. It
requires no model calls and does not alter frozen scoring or study policies.
"""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'report/experimental-resume-v1/completion-audit-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    assert value['sha256'] == digest({k: v for k, v in value.items() if k != 'sha256'}), str(path)
    return value


REVIEWS = [
    (1, 'Exponential equation', 'x = 4',
     'The saved prefix already solves and verifies x=4. All six continuations finish with the correct boxed answer. Additional reasoning mostly checks wording, formatting and the same calculation.',
     ['repeated_verification', 'response_redrafting', 'correct_finished_answer'],
     [('force', 'Output Generation')]),
    (2, 'Balloon and rope geometry', 'maximum saving = 160',
     'All six continuations are capped. They restate coordinate conventions and constraints, but also make valid progress towards the height. Some interventions progress farther than native. Capping here does not demonstrate mathematical divergence.',
     ['repeated_setup', 'useful_unfinished_work'], []),
    (3, 'Equilateral hexagon', None,
     'All six continuations are capped while revisiting directions, opposite sides and vector representations. Much of the available budget is spent on setup rather than using the given triangle side lengths. The native duplicate drops a minus sign; the random reweight control rejects a valid telescoping vector identity.',
     ['repeated_setup', 'local_math_error', 'unresolved_doubt'],
     [('native_duplicate', r'\vec{E} - \vec{D} = \vec{B} - \vec{A}'),
      ('random_reweight', r'\vec{AB} + \vec{BC} + \vec{CD} = \vec{AD}$? No.')]),
    (4, 'Radical equation', 'x = 6',
     'The 1,826-token saved prefix already solves the equation, rejects x=3 and verifies x=6. The continuations repeat those checks and consider unsupported alternate interpretations. Native and reweight close reasoning but their answers are cut off; the other four remain in reasoning at the cap.',
     ['repeated_verification', 'response_redrafting', 'unsupported_alternative', 'correct_candidate_without_finished_answer'],
     [('native', 'Wait, what if complex numbers were allowed?')]),
    (5, 'Arithmetic sequence', 'x = 45',
     'All six reach the correct candidate. Native, native duplicate, force and random force finish correctly. Bias closes reasoning but its final answer is cut off; reweight never closes reasoning. Hypotheses about geometric sequences, indices and numeral bases are raised and rejected rather than established as facts.',
     ['repeated_verification', 'response_redrafting', 'unsupported_alternative', 'mixed_operator_effect'],
     [('bias', 'What if the user meant $3^2, x, 3^4$ as indices?'),
      ('bias', 'What if $3^2$ is not base 10?')]),
    (6, 'Complex rotation', 'w = 6 - 5i',
     'Every continuation reaches the correct final value near its start, repeats the calculation, and then checks coordinates, angles, magnitude or the diagram code. All six spend 1,024 new tokens without closing reasoning. This is concrete evidence of delayed finalization despite a correct candidate.',
     ['repeated_verification', 'correct_candidate_without_finished_answer'],
     [('native', '$w = 6 - 5i$.'),
      ('native', 'The solution appears robust. No obvious pitfalls found.')]),
    (7, 'Common tangents', 'area = 15',
     'All six continuations correctly obtain the two tangent slopes but do not reach the area within the budget. Native and reweight make false statements about vertical tangency; native first recognizes x=0 as the vertex tangent and later calls it a normal. Force writes and immediately rejects an incorrect contact-point formula. The trigger labelled candidate_to_verify is only the circle radius, before an answer candidate exists.',
     ['useful_unfinished_work', 'local_math_error', 'unresolved_doubt', 'questionable_transition_trigger'],
     [('native', 'vertical line cuts at one point but is normal.'),
      ('reweight', "vertical tangents don't exist (parabola opens sideways).")]),
]


def reference_checks():
    # Independent short calculations, not grades of the generated outputs.
    assert 2 ** 8 == 4 ** 4
    assert 2 + 2 == 6 - 2  # sqrt(6-2) = 2
    assert 2 + 1 != 3 - 2  # sqrt(3-2) = 1
    assert Fraction(9 + 81, 2) == 45 and 45 - 9 == 81 - 45
    # (sqrt(2)-3sqrt(2)i)*(1+i)/sqrt(2) = (1-3i)*(1+i).
    assert complex(2, -3) + complex(1, -3) * complex(1, 1) == complex(6, -5)
    cp = Fraction(150 ** 2 + 140 ** 2 - 130 ** 2, 2 * 140)
    assert cp == 90 and 0 < cp < 140 and 150 ** 2 - cp ** 2 == 120 ** 2
    assert 150 + 130 - 120 == 160
    # Tangencies at (-1,+/-1) and (2,+/-4), vertical parallel sides 2 and 8.
    for sign in (-1, 1):
        assert (-1) ** 2 + sign ** 2 == 2
        assert (4 * sign) ** 2 == 8 * 2
        assert sign == sign * (-1) + 2 * sign
        assert 4 * sign == sign * 2 + 2 * sign
    assert Fraction(2 + 8, 2) * (2 - (-1)) == 15
    return {'checked': [1, 2, 4, 5, 6, 7],
            'case_3': 'No final reference answer asserted; local vector errors checked algebraically.',
            'status': 'PASS_SHORT_INDEPENDENT_REFERENCE_CALCULATIONS'}


def main():
    sample = sealed(OUT / 'SAMPLE.json')
    cases = {c['case']: c for c in sample['cases']}
    assert len(cases) == len(REVIEWS) == 7
    findings = []
    for number, title, answer, observation, patterns, evidence in REVIEWS:
        case = cases[number]
        rows = {r['arm']: r for r in case['completions']}
        assert len(rows) == 6
        excerpts = []
        for arm, quote in evidence:
            row = rows[arm]
            assert quote in row['text'], (number, arm, quote)
            excerpts.append({'arm': arm, 'uid': row['uid'], 'verbatim': quote})
        if number == 6:
            assert all('$w = 6 - 5i$.' in r['text'] for r in rows.values())
            assert all(r['generated_tokens'] == 1024 and not r['reasoning_closed'] for r in rows.values())
        findings.append({'case': number, 'title': title, 'question': case['question'],
                         'reference_answer': answer, 'manual_observation': observation,
                         'patterns_observed_in_case': patterns, 'evidence': excerpts,
                         'continuation_lengths': {a: {k: r[k] for k in
                             ('reasoning_tokens', 'answer_tokens', 'generated_tokens', 'finish', 'reasoning_closed')}
                             for a, r in rows.items()}})
    completions = [r for c in cases.values() for r in c['completions']]
    counts = Counter((r['finish'], r['reasoning_closed']) for r in completions)
    assert len(completions) == 42 and counts == {('stop', True): 10, ('length', False): 29, ('length', True): 3}
    value = {'schema': 'saved-completion-manual-readout-v1',
             'recorded_utc': datetime.now(timezone.utc).isoformat(),
             'sample_path': str(OUT / 'SAMPLE.json'), 'sample_sha256': sample['sha256'],
             'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'selection_rule': sample['selection_rule'], 'cases_reviewed': 7, 'completions_reviewed': 42,
             'finish_counts_in_selected_sample': {'natural_stop': 10, 'unclosed_capped': 29, 'closed_but_capped': 3},
             'reference_checks': reference_checks(), 'findings': findings,
             'interpretation': 'Repeated checking and delayed finalization occur in native controls and intervention arms. Genuine local mathematical errors coexist with correct candidates and useful unfinished work. The sample does not establish a general hallucination rate or an operator ranking.',
             'limitations': ['Selection deliberately covers native finish/boundary strata; sample counts are not population rates.',
                            'These are 1,024-token continuations after saved prefixes, not full original-prompt 16,384-token runs.',
                            'Reasoning containing a correct candidate is not a completed answer and does not override frozen grading.',
                            'No causal diagnosis of a sampler, template or EOS implementation defect is established.'],
             'source_inspection': {'branch_builder': 'scripts/experimental_resume/overnight_routing_runner_v2.py:285',
                                   'observed': 'Exact prompt-plus-prefix IDs, common seeds across arms, restore_presence=True and presence=0.0 for the branch request. A zero request penalty alone does not establish disabled presence penalties.',
                                   'engine_source': '/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/code/s1-9a61e32f48c04c24/moe_steer/engine.py:140'},
             'next_steps': ['Prioritize the already authorized full original-prompt accuracy/length panel and its frozen strict/J1 grading; retain cohort selection from measured costs before examining its accuracy outcomes.',
                            'Keep correct reasoning candidates, finished answers, reasoning closure and capped outputs separate in the diagnostic readout; retain operationally-wrong cap/error scoring.',
                            'Use a small manual sample of long native and intervention traces to assess repeated checking and delayed finalization; reuse existing classifications rather than restarting dense relabelling.',
                            'Check existing replay/termination qualification and manually validate a few trigger contexts. If full native runs still loop, propose a separate controlled sampler/prompt/termination experiment without changing frozen operator comparisons.'],
             'operations': 'No new Slurm jobs or inference. Deferred dense classification remains deferred; no frozen source, scoring rule, policy or result edited.'}
    value['sha256'] = digest(value)
    path = OUT / 'FINDINGS.json'
    if path.exists():
        prior = sealed(path)
        assert prior['script_sha256'] == value['script_sha256'] and prior['findings'] == findings
        value = prior
    else:
        with path.open('x') as stream:
            json.dump(value, stream, indent=1, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    assert sealed(path)['sha256'] == value['sha256']
    print(json.dumps({'path': str(path), 'sha256': value['sha256'],
                      'cases': 7, 'completions': 42, 'reference_checks': value['reference_checks']['status']}))


if __name__ == '__main__':
    main()
