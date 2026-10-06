"""Seal the measurement UID compatibility correction; preserve the original addendum."""
from __future__ import annotations
import datetime
import hashlib
import json
from pathlib import Path
import shutil

REPO=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
S=Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')


def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def run():
    old=json.loads((S/'runs/s2prop/FROZEN_ADDENDA.json').read_text())['snapshots']['nll']
    sources={'native_nll.py':REPO/'scripts/experimental_resume/native_nll.py',
        'nll_helpers.py':REPO/'src/moe_exp/routing_control/nll.py',
        'receipts.py':REPO/'src/moe_exp/routing_control/receipts.py'}
    hashes={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in sources.items()}
    tree=digest(hashes);dest=S/'addenda/nll'/('nll-'+tree[:16]);dest.mkdir(parents=True,exist_ok=True)
    for name,source in sources.items():
        p=dest/name
        if p.exists() and p.read_bytes()!=source.read_bytes():raise ValueError('sealed NLL code changed')
        if not p.exists():shutil.copy2(source,p);p.chmod(0o400)
    manifest={'schema':'native-nll-addendum-v2','tree_sha256':tree,'files':hashes,
        'generation_tree':json.loads((S/'runs/s2prop/FROZEN_ADDENDA.json').read_text())['snapshots']['s2']['tree_sha256'],
        'measurement':'same native teacher-forced logprobs, identical Q3 fixtures and tolerances',
        'qualification':'PENDING actual identical-fixture rescore',
        'supersedes_measurement_code_only':old,'uid_format':'explicit legacy SHA256-prefix32 binding',
        'method_change':False,'sampler_change':False,'reason':'Legacy request UIDs contain 32 hex digits; the generic new-study receipt store requires 64 unless an explicit legacy binding is present.'}
    manifest={**manifest,'sha256':digest(manifest)};p=dest/'MANIFEST.json'
    if p.exists() and json.loads(p.read_text())!=manifest:raise ValueError('NLL inventory changed')
    if not p.exists():p.write_text(json.dumps(manifest,indent=1)+'\n');p.chmod(0o400)
    dest.chmod(0o500)
    body={'schema':'native-nll-UID-amendment-v1','recorded_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'old_addendum':old,'new_addendum':{'path':str(dest),'tree_sha256':tree,'seal':manifest['sha256']},
        'authority':'current user instruction to implement and verify the separate native measurement driver',
        'prospective':'before any native NLL model load or outcomes; pending old job is replaced with a separately receipted job',
        'method_change':False,'sampler_change':False,'budget_change':False,
        'output_binding':'manifest digest AND new measurement tree digest',
        'old_protocol_inventory':'preserved unchanged; this is an additional engineering receipt'}
    path=S/'runs/resume-v1/NLL_UID_AMENDMENT.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if previous['new_addendum']!=body['new_addendum']:raise ValueError('a different NLL amendment exists')
        body=previous
    else:path.write_text(json.dumps({**body,'sha256':digest(body)},indent=1)+'\n')
    print(json.dumps(body['new_addendum']))


if __name__=='__main__':run()
