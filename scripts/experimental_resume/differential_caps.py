"""Compare real no-cap request dictionaries across the immutable s1 and s2 runtimes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

S = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
PY = '/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/correlation-client-3.11/bin/python'

PROGRAM = '''
import inspect,json
from moe_steer import manifests as M, policies
import x2_build as X
world=M.load_world()
table,_=X.x2_table(world.inputs)
rows=[r for r in json.loads((policies.CAMPAIGN/'runs/x2prep/eligibility.json').read_text())['rows'] if r['eligible']][:3]
out=[]
for row in [None,*rows]:
 info=world.infos[rows[0]['question'] if row is None else row['question']]
 kw=dict(arm='N',policy_name='zero',seed_k=0)
 if row is not None: kw['parent']={'trace_ref':row['trace_ref'],'prefix_len':row['prefix_len']}
 request=M.make_request('no-cap-parity',table,info,**kw)
 if 'max_new_tokens' in inspect.signature(M.make_request).parameters:
  assert request==M.make_request('no-cap-parity',table,info,max_new_tokens=None,**kw)
 out.append(request)
print(json.dumps(out,sort_keys=True,separators=(',',':')))
'''


def main():
    old = S / 'code/s1-f2ded3957eb54fd5'
    new = S / 'code/s1-9a61e32f48c04c24'
    outputs = []
    for snap in (old, new):
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONPATH': f'{snap}:{snap}/moe_exp_src:{S}/addenda/x2'}
        outputs.append(subprocess.run([PY, '-B', '-c', PROGRAM], env=env,
                                      check=True, text=True, capture_output=True).stdout)
    assert json.loads(outputs[0]) == json.loads(outputs[1])
    evidence = {'status': 'PASS', 'real_original_requests': 1, 'real_branch_requests': 3,
        'all_dictionary_fields_equal_s1_default_s2_default_and_s2_None': True,
        'canonical_request_json_sha256': hashlib.sha256(outputs[0].strip().encode()).hexdigest()}
    (S / 'runs/s2prop/default-request-parity.json').write_text(json.dumps(evidence, indent=1) + '\n')
    # One legacy assertion represented the exact old cap invariant. Retain its original file
    # unchanged and replace only that assertion in the separately named cap follow-up suite.
    dest = S / 'runs/s2prop/legacy-cap-followup'
    dest.mkdir(exist_ok=True)
    source = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/steering/tests')
    for name in ('test_engine_manifests.py', 'test_runner.py', 'e5_world.py'):
        text = (source / name).read_text()
        if name == 'test_engine_manifests.py':
            anchor = 'check(lambda b: b["requests"][0].update(max_tokens=5), "uid does not match|cumulative cap")'
            assert text.count(anchor) == 1
            text = text.replace(anchor, 'check(lambda b: b["requests"][0].update(max_tokens=0), "outside")')
        (dest / name).write_text(text)
    print(json.dumps(evidence, indent=1))


if __name__ == '__main__':
    main()
