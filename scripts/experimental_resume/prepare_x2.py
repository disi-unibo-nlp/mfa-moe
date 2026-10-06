"""Prepare sealed production X2 from the already frozen cached enumeration.

No corpus preprocessing, engine load or Slurm submission. All-in price and H1–H4
receipts are verified before the manifest is sealed; output names bind its digest.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
S = ROOT / 'steering-v1'
REPO = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    import x2_build as X
    from moe_steer import manifests as M, engine
    from moe_steer.spec import seal, verify
    qualification = S / 'qualification/h14/resume-v1-recovery/H14.json'
    verdict = json.loads(qualification.read_text())
    if not verdict['pass'] or not all(verdict['evidence']['holes'][h]['pass'] for h in ('H1','H2','H3','H4')):
        raise ValueError('all four qualification holes must pass')
    tree = json.loads((S / 'runs/s2prop/FROZEN_ADDENDA.json').read_text())['snapshots']['s2']['tree_sha256']
    if engine.code_tree_sha256() != tree:
        raise ValueError('production X2 requires the frozen cap tree')
    if not (S / 'PREREG_steering_v1.v0.3.md').exists():
        raise ValueError('v0.3 design must be sealed before production X2')
    family = json.loads((REPO / 'report/experimental-resume-v1/family-freeze.json').read_text())
    verify(family)
    world, split = M.load_world(), M.load_split()
    eligible = json.loads((S / 'runs/s2prop/eligibility.json').read_text())
    manifest, cells = X.build_x2_manifest(world, eligible['rows'], name='x2-resume-v1',
        n_shards=1, max_new_tokens=1024, n_policy='sham', code_tree=tree)
    X.check_manifest(manifest, eligible['rows'], cells, split=split, infos=world.infos,
                     n_policy='sham', max_new_tokens=1024, expect_tree=tree)
    out = S / 'runs/x2-resume-v1'
    out.mkdir(exist_ok=True)
    path = out / 'x2-resume-v1.json'
    M.write_manifest(path, manifest)
    assert M.load_manifest(path) == manifest
    guard_source = REPO / 'scripts/experimental_resume/guarded_runner.py'
    guard_sha = sha(guard_source)
    immutable = S / 'addenda/run' / guard_sha[:16]
    immutable.mkdir(parents=True, exist_ok=True)
    guard = immutable / 'guarded_runner.py'
    if guard.exists() and sha(guard) != guard_sha:
        raise ValueError('immutable guard changed')
    if not guard.exists():
        shutil.copy2(guard_source, guard)
        guard.chmod(0o400)
    cost = json.loads((S / 'runs/x2prep/cost_model.json').read_text())
    planning = cost['rule_walk']['central|shards=1']
    members = family['new_parent_pools']['families']
    enrolled = set(manifest['questions'])
    freeze = seal({'schema': 'x2-production-resume-freeze-v1', 'manifest': str(path),
        'manifest_sha256': manifest['sha256'], 'manifest_file_sha256': sha(path),
        'guard_path': str(guard), 'guard_sha256': guard_sha,
        'output': str(out / 'outputs' / manifest['sha256'] / guard_sha),
        'prereg_path': str(S / 'PREREG_steering_v1.v0.3.md'),
        'prereg_sha256': sha(S / 'PREREG_steering_v1.v0.3.md'),
        'qualification': {'path': str(qualification), 'sha256': sha(qualification), 'job_id': 59092583,
                          'state': 'COMPLETED', 'exit_code': '0:0', 'all_holes_pass': True},
        'family_source_sha256': family['sha256'],
        'families': {f: [q for q in questions if q in enrolled] for f, questions in members.items()
                     if enrolled & set(questions)},
        'analysis_units': {'primary': 'question', 'sensitivity': 'frozen duplicate family'},
        'price': {'all_in_central_GPUh': planning['total_gpu_h'],
                  'generation_max_reserved_GPUh': 6., 'pending_NLL_pessimistic_GPUh': 1.5,
                  'contingency_GPUh': .5, 'total_ceiling_GPUh': 8.,
                  'source': str(S / 'runs/x2prep/cost_model.json'),
                  'source_sha256': sha(S / 'runs/x2prep/cost_model.json'),
                  'loads_NLL_retries_all_charged': True},
        'requests': len(manifest['requests']), 'horizon': 1024,
        'enforced_checks': ['cap or natural stop', 'routed rows and native top-k8',
                            'force membership on every active layer before accepting request'],
        'post_generation_checks': ['both ranks same UID/pulses', 'sham zero active rows',
                                  'every E/M request positive actual TV dose', 'all assigned UIDs accounted'],
        'authorizations': 'current user: implement legacy X2 within its stated eight GPU-hour ceiling'})
    if freeze['price']['generation_max_reserved_GPUh'] + freeze['price']['pending_NLL_pessimistic_GPUh'] + freeze['price']['contingency_GPUh'] > 8.:
        raise ValueError('complete commitments exceed X2 ceiling')
    frozen_path = out / 'FROZEN.json'
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != freeze:
        raise ValueError('production experiment freeze changed')
    frozen_path.write_text(json.dumps(freeze, indent=1) + '\n')
    assert verify(json.loads(frozen_path.read_text()))
    print(json.dumps({'manifest': str(path), 'manifest_sha256': manifest['sha256'],
                      'freeze_sha256': freeze['sha256'], 'guard': str(guard), 'requests': freeze['requests']}, indent=1))


if __name__ == '__main__':
    run()
