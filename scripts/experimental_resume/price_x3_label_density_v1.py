"""Seal the bounded CPU-only native X3 label-count projection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
S = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
SNAP = S / 'code/s1-9a61e32f48c04c24'
REPORT = REPO / 'report/experimental-resume-v1'
OUT = REPORT / 'X3_LABEL_DENSITY_PRICE_v1.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(f'invalid seal: {path}')
    return value


def main():
    eligible = sealed(REPORT / 'X3_ELIGIBILITY_v1.json')
    selected = sealed(REPORT / 'X3_G3_SELECTION_v2.json')
    if eligible['summary']['eligible'] != 77 or selected['registered_passing_cells'] != 2:
        raise ValueError('X3 frozen enrollment or dose selection changed')
    sys.path.insert(0, str(SNAP))
    sys.path.insert(0, str(SNAP / 'moe_exp_src'))
    from moe_steer import engine
    if engine.code_tree_sha256() != '9a61e32f48c04c242acccc89c529bd750776c553cdfc776347151d359bc53430':
        raise ValueError('frozen X3 engine tree changed')
    tokenizer = Path(engine.snapshot_path()) / 'tokenizer.json'
    inputs = [REPORT / 'X3_ELIGIBILITY_v1.json', REPORT / 'X3_G3_SELECTION_v2.json',
              SNAP / 'moe_steer/label_units.py', SNAP / 'moe_exp_src/moe_exp/correlation_pipeline/spans.py',
              tokenizer]
    body = {'schema': 'legacy-X3-native-label-density-price-v1',
            'status': 'READY_CPU_COUNT_ONLY',
            'source_sha256': sha(REPO / 'scripts/experimental_resume/x3_label_density_projection_v1.py'),
            'sampler_tree_sha256': engine.code_tree_sha256(),
            'eligibility_sha256': eligible['sha256'],
            'selection_sha256': selected['sha256'],
            'tokenizer_sha256': sha(tokenizer),
            'input_sha256': {str(p): sha(p) for p in inputs},
            'scope': '77 registered X3 firing native sample_00 traces; original-prefix cumulative segmentation; count only sentences starting in 512-token reasoning window; no judge/inference/intervention reads',
            'resources': {'account': 'iscrc_miosr', 'partition': 'boost_usr_prod',
                          'qos': 'normal', 'cpus': 1, 'memory': '16G',
                          'wall_seconds': 3600, 'max_core_hours': 1.0, 'gpus': 0},
            'interpretation': '10x scaling for two seeds and five arms is a density proxy; reprice from actual X3 branch units after generation.'}
    body['sha256'] = digest(body)
    if OUT.exists():
        if json.loads(OUT.read_text()) != body:
            raise ValueError('existing X3 label-density price differs')
    else:
        OUT.write_text(json.dumps(body, indent=1) + '\n')
    print(json.dumps({'price': str(OUT), 'sha256': body['sha256']}))


if __name__ == '__main__':
    main()
