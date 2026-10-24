"""Recheck reviewed live job scripts before retiring authorized unused weights."""
import argparse, hashlib, importlib.util, json, os, subprocess, time
from pathlib import Path

def digest(b):return hashlib.sha256(b).hexdigest()
def live():return set(subprocess.check_output(['squeue','-h','-u','lmolfett','-o','%i'],text=True).split())-{os.environ.get('SLURM_JOB_ID','')}
def script(job):
 b=subprocess.check_output(['scontrol','write','batch_script',job,'-'])
 if not b.startswith(b'#!/'):raise ValueError('Cannot verify submitted script '+job)
 return b

def main():
 p=argparse.ArgumentParser();p.add_argument('proof',type=Path);a=p.parse_args()
 if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Requires CPU Slurm allocation')
 d=json.loads(a.proof.read_text());jobs=live()
 if not jobs.issubset(d['jobs']):raise ValueError('New unaudited jobs appeared')
 for j in jobs:
  if digest(script(j))!=d['jobs'][j]['script_sha256']:raise ValueError('Submitted script changed')
 for f,h in d['bound_files'].items():
  if digest(Path(f).read_bytes())!=h:raise ValueError('Reviewed manifest/code changed: '+f)
 if d['candidate_dependencies']!=[]:raise ValueError('Candidate required by job')
 audit=a.proof.with_name('live-audit.json')
 audit.write_text(json.dumps(dict(at=time.time(),manifest_sha256=d['manifest_sha256'],checked_job_ids=sorted(jobs),candidate_dependencies=[],proof_sha256=digest(a.proof.read_bytes())),indent=2)+'\n')
 spec=importlib.util.spec_from_file_location('retirement',d['retirement_script']);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.retire(Path(d['manifest']),audit)
if __name__=='__main__':main()
