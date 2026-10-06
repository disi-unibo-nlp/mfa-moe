"""Seal a small read-only GPT sidecar/canonical checkpoint parity audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
REPORT = REPO / 'report/experimental-resume-v1'
SOURCE = REPO / 'scripts/experimental_resume/audit_r3d_gpt_overlap_v1.py'
VERIFY = REPORT / 'R3D_SIDECAR_VERIFY_GPT_DEST_v2.json'
OUT = REPORT / 'R3D_GPT_OVERLAP_AUDIT_PRICE_v1.json'


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def main():
    verified = json.loads(VERIFY.read_text())
    if verified['sha256'] != digest({k:v for k,v in verified.items() if k!='sha256'}):
        raise ValueError('GPT sidecar verifier receipt changed')
    body = {'schema':'r3d-gpt-overlap-audit-price-v1','status':'READY',
            'driver_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'verifier_sha256':verified['sha256'],
            'source_fits':300, 'canonical_overlap_max_fits':300,
            'cpus':2,'memory':'16G','wall_seconds':1200,'gpus':0,
            'max_core_hours':2/3,
            'slurm':{'account':'iscrc_miosr','partition':'boost_usr_prod','qos':'normal'},
            'output':str(REPORT/'R3D_GPT_OVERLAP_AUDIT_v1.json'),
            'scope':'read-only overlap numeric/fold/hash comparison; does not stop canonical or merge'}
    value={**body,'sha256':digest(body)}
    if OUT.exists():
        if json.loads(OUT.read_text()) != value:
            raise ValueError('existing overlap price differs')
    else:
        OUT.write_text(json.dumps(value,indent=1)+'\n')
    print(json.dumps({'price':str(OUT),'seal':value['sha256']}))


if __name__=='__main__':
    main()
