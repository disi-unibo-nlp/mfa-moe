"""Seal legacy v0.3 with explicit precedence and byte-level evidence inventory.

The current user specifically requested completion and sealing. This receipt names
that authority; it never impersonates the former Claude controller/reviewer.
Historical protocols and raw failed qualifications are preserved.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path(__file__).resolve().parents[2]
S = ROOT / 'steering-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def run():
    if not (ROOT / 'swarm/DRAIN').exists():
        raise ValueError('previous keeper must remain drained')
    external_cap = S / 'addenda/s2/PROPOSAL.md'
    external_cap.write_text((REPO / 'report/experimental-resume-v1/CAP_FIX.md').read_text())
    (ROOT / 'swarm/reports/s2prop.md').write_text(external_cap.read_text())
    old = S / 'PREREG_steering_v1.md'
    amendment = REPO / 'report/experimental-resume-v1/LEGACY_AMENDMENT_v0.3.md'
    split = json.loads((S / 'manifests/split-v1.json').read_text())
    disc = [f"{q['dataset']}|{q['source_problem_id']}" for q in split['questions']
            if q.get('subsplit') == 'dev-disc']
    if len(disc) != 96:
        raise ValueError(f'expected 96 dev-disc questions, found {len(disc)}')
    order = sorted(disc, key=lambda q: hashlib.sha256(('forum-v1|' + q).encode()).hexdigest())
    subset = order[:64]
    appendix = '\n\n## Frozen X3 enrollment appendix\n\n'
    appendix += 'Ordered first-64 IDs; SHA256 of canonical JSON list: `' + digest(subset) + '`.\n\n'
    appendix += '\n'.join(f'{i}. `{q}`' for i, q in enumerate(subset, 1)) + '\n'
    appendix += '\n## Historical v0.2 reference appendix\n\n'
    appendix += ('The following text is preserved verbatim as historical reference. All conflicts\n'
                 'are governed by the normative v0.3 amendment above. Its historical status,\n'
                 'approval requirements, estimates and retired wording do not override v0.3.\n\n')
    source_body = amendment.read_text() + appendix + old.read_text()
    draft = S / 'PREREG_steering_v1.v0.3-draft.md'
    frozen = S / 'PREREG_steering_v1.v0.3.md'
    draft_content = '# PREREG steering-v1 v0.3\n\nStatus: DRAFT v0.3, NOT YET FROZEN\n\n' + source_body
    frozen_content = draft_content.replace('Status: DRAFT v0.3, NOT YET FROZEN',
        'Status: FROZEN v0.3 (2026-10-01); design sealed, launch holds retained', 1)
    if frozen.exists() and frozen.read_text() != frozen_content:
        raise ValueError('sealed v0.3 bytes already exist and differ')
    draft.write_text(draft_content)
    changes = ['# PREREG v0.3 checklist and changes', '',
        'Authority: the current user requested implementation, including finishing and sealing v0.3.',
        'The former executor/controller review flow is superseded by that instruction. No reviewer signature is fabricated.', '',
        '| Resolution | Normative v0.3 section |', '|---|---|',
        '| A1–A3 | 3–4 (sets, equal-layer dose, matching) |',
        '| A4–A5 | 6 (G4 and claim-level rules, escalation) |',
        '| A6–A7 | 1,5 (split/families, fixed occupancy) |',
        '| A8 | 2 (Q13; MTP Q12 not run) |',
        '| A9–A11 | 7 (rescue, deferred P*, A12 MDE) |',
        '| A12–A13 | 7–8 and inventory (frozen files, R1-C4 correction) |',
        '| B | 2,6 (CRN, retries, build/shard, X1/G2 facts) |',
        '| C1–C3 | 3–4 (exact enumeration, 1024 requirement, all-in eight-hour accounting) |',
        '| C4–C6; C4–C6 primes | 5–6 and enrollment appendix (X3, secondary status, power) |',
        '| C7–C9 | 2,4 (G3, immutable s2, separate validated native NLL) |',
        '| C10–C13 | 1,6,8 (G2 reading, M7 missing, family sensitivity, inventory) |',
        '| G4 prime; C14–C15 | 6 (retired direction gate, six claims, LLM audit) |',
        '| C16–C17 | 2,8 and inventory (holes, snapshots, status at seal) |',
        '| C18–C21 | 2–5,8 (ordered64, branch audit, revised cap tree, patch inventory) |', '',
        'Resolved conflicts: fixed user requirement retains 1024-token X2 continuations; the old 768 rescue is retired.',
        'The 39-question cached enumeration replaces 48-question paper arithmetic; 128 policies includes zero.',
        'The old cap API and test assertion are superseded only through the inventoried two-hunk patch and separate test copy.',
        'The new native NLL path uses actual saved parity arrays; there is no fallback to unqualified branches.',
        'The first-64 R3-C simulation used steering-v1 salt; registered X3 uses forum-v1. Their mismatch is disclosed.',
        'No Fable/controller approval, engine equivalence, semantic steering or equivalence from an MDE is claimed.', '',
        'Open launch holds are explicit states, not missing design placeholders: H1–H4 recovery; native parity/NLL;',
        'X3 builder and complete cost; new qualified worker/detectors, behavioral eligibility and stage prices.',
        'No [controller: value] placeholders remain in the normative amendment.', '']
    change_path = S / 'PREREG_v0.3_CHANGELOG.md'
    change_path.write_text('\n'.join(changes))
    required = [old, draft, frozen, amendment, change_path, external_cap,
        S / 'PREREG_open_decisions.md', S / 'DECISIONS.md', S / 'R1_ADJUDICATION.md',
        S / 'provenance/vllm-j1-59025605.log', S / 'runs/x1/score/items.jsonl.manifest.json',
        S / 'qualification/Q3.json', S / 'qualification/H1-H4.json',
        S / 'qualification/h14/proposal/FROZEN_SPEC.json',
        S / 'qualification/h14/proposal/PROPOSED_TREE.json',
        S / 'addenda/s2/manifests.py.patch', S / 'addenda/s2/PROPOSED_TREE.json',
        S / 'addenda/x2/x2_build.py', S / 'runs/x2prep/enumeration.json',
        S / 'runs/x2prep/cost_model.json', S / 'runs/x2prep/NLL_FEASIBILITY.md',
        S / 'runs/s2prop/x2-dryrun-s2.json', S / 'runs/s2prop/dry-verification.json',
        S / 'runs/s2prop/FROZEN_ADDENDA.json', S / 'runs/resume-v1/RESOURCE_AUTHORIZATION.json',
        REPO / 'report/experimental-resume-v1/family-freeze.json',
        ROOT / 'forum/tests/r3_integrity/out/near_dup.v2.json',
        ROOT / 'forum/moderator/schedule_v2.md', ROOT / 'forum/moderator/verdict_r3.md']
    required += [S / 'manifests' / (name + '.json') for name in
                 ('split-v1', 'targets-v1', 'random-sets-v2', 'lexicon-v1', 'x1-v1', 'G1-gate-v1', 'G2-gate-v1')]
    required += [ROOT / 'forum/tests/r3_x3_power' / name for name in
                 ('FROZEN.json', 'RESULTS.md', 'DETAILS.md', 'estimates.json',
                  'estimates.addendum.json', 'AMENDMENT_RECOMMENDATION.md')]
    required += [REPO / 'results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3' / name for name in
                 ('selected_program_20260827_173300.json', 'results_20260827_173300.json')]
    required += list((S / 'addenda/x2/tests').glob('*.py'))
    required += list((S / 'qualification/parity').glob('*.tf*.npz'))
    required += list((S / 'qualification/parity').glob('*.tf*.json'))
    required += list((S / 'qualification/h14/resume-v1').glob('H*.json'))
    identities = json.loads((S / 'runs/s2prop/FROZEN_ADDENDA.json').read_text())['snapshots']
    for path in [S / 'code/s1-f2ded3957eb54fd5',
                 Path(identities['s2']['snapshot']), Path(identities['h14']['snapshot']),
                 Path(identities['nll']['path'])]:
        required += [p for p in path.rglob('*') if p.is_file()]
    # Check all references before publishing frozen bytes or appending a seal receipt.
    missing = [str(p) for p in required if p != frozen and not p.exists()]
    if missing:
        raise FileNotFoundError('required seal inputs absent: ' + json.dumps(missing))
    if not frozen.exists():
        frozen.write_text(frozen_content)
    inventory = {'schema': 'prereg-v0.3-byte-inventory', 'protocol': str(frozen),
        'protocol_sha256': sha(frozen), 'historical_v0.2_sha256': sha(old),
        'draft_sha256': sha(draft), 'authority': 'current user implementation and sealing instruction',
        'X3_first64_sha256': digest(subset), 'snapshots': identities,
        'files': {str(p): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(set(required))}}
    inventory['sha256'] = digest(inventory)
    inventory_path = S / 'PREREG_v0.3_INVENTORY.json'
    if inventory_path.exists() and json.loads(inventory_path.read_text()) != inventory:
        raise ValueError('sealed inventory already differs')
    inventory_path.write_text(json.dumps(inventory, indent=1) + '\n')
    record = {'ts': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'version': 'v0.3',
        'path': str(frozen), 'sha256': sha(frozen), 'draft_sha256': sha(draft),
        'approved_by': 'current user: implement resume plan including finish and seal v0.3',
        'sealed_by': 'Codex', 'inventory_sha256': inventory['sha256'],
        'status': 'design sealed; execution holds retained'}
    receipt = S / 'provenance/prereg-freeze.jsonl'
    already_recorded = receipt.exists() and any(json.loads(line).get('sha256') == record['sha256']
                                                for line in receipt.read_text().splitlines())
    if not already_recorded:
        with receipt.open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        with (S / 'ledger.jsonl').open('a') as handle:
            handle.write(json.dumps({**record, 'task': 'prereg3'}) + '\n')
    assert sha(frozen) == inventory['protocol_sha256']
    reread = json.loads(inventory_path.read_text())
    expected = reread.pop('sha256')
    assert digest(reread) == expected
    report = '# PREREG v0.3 sealed\n\n' + json.dumps(record, indent=1) + '\n\n'
    report += ('Exact protocol and inventory bytes verified after publication. Historical v0.2 and raw failed\n'
               'qualification records remain unchanged. The current user explicitly requested sealing;\n'
               'this record does not impersonate the former controller approval.\n\n'
               'X2: 6630 requests, each 1024 tokens, 4.94 GPU-hour central all-in estimate (4.49–6.75),\n'
               'eight-hour aggregate ceiling. H1–H4 must pass before generation. G3 requires qualified\n'
               'native NLL. X3 has no launch permission until its builder and complete cost pass.\n')
    (ROOT / 'swarm/reports/prereg3.md').write_text(report)
    (REPO / 'report/experimental-resume-v1/PREREG_SEAL.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(record, indent=1))


if __name__ == '__main__':
    run()
