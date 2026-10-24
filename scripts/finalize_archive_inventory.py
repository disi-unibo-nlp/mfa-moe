"""Bind a transfer inventory to an audited inventory, excluding mutable archive-job logs."""
import hashlib,json
from pathlib import Path
ROOT=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study')
def finalize(source):
 base=ROOT/'results/measurement-study';parent=base/'archive-v1'/source/'inventory.json';out=base/'archive-v2'/source/'inventory.json'
 if out.exists():raise ValueError('Final inventory exists')
 d=json.loads(parent.read_text());excluded=[r for r in d['files'] if Path(r['source']).name.startswith('mfa-archive-') and '/logs/' in r['source']]
 d['files']=[r for r in d['files'] if r not in excluded]
 for r in d['files']:
  if r['copy_required']:r['destination']=str(base/'archive-v2/external'/r['source'].lstrip('/'))
 d.update(parent_inventory=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),excluded_operational_logs=[r['source'] for r in excluded],total_bytes=sum(r['bytes'] for r in d['files']),copy_bytes=sum(r['bytes'] for r in d['files'] if r['copy_required']))
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(d,indent=2)+'\n')
 (out.parent/'transfer-files.txt').write_text(''.join(r['source'].lstrip('/')+'\n' for r in d['files'] if r['copy_required']))
 print(source,len(d['files']),d['copy_bytes']/1024**3)
if __name__=='__main__':
 import sys
 for source in sys.argv[1:]:
  if source not in ['gpt','glm']:raise ValueError('Unexpected source')
  finalize(source)
