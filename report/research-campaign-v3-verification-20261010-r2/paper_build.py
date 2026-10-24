"""Bounded snapshot-only asset completion; historical builders stay untouched."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
repo=Path(__file__).resolve().parents[2]
stage=Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp')
old=stage/'research-campaign-v3-paper-20261010-r1'
binding=json.loads((old/'BUILD.json').read_text())
assert all(sha(p)==h for p,h in binding['canonical_source_files'].items())
target=stage/'research-campaign-v3-paper-20261010-r2';target.mkdir(exist_ok=False)
snapshot=target/'source';shutil.copytree(old/'source',snapshot)
paper=repo/'paper/6a510c6db0d9b08aa438c142'
assets={}
for path in (paper/'generated').rglob('*'):
    if path.is_file() and path.suffix in ('.pdf','.svg','.png','.jpg'):
        dest=snapshot/path.relative_to(paper);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest);assets[str(path)]=sha(path)
        assert sha(dest)==assets[str(path)]
output=target/'output';output.mkdir();temporary=target/'tmp';temporary.mkdir()
engine=Path(binding['toolchain']['binary_path'])
assert sha(engine)==binding['toolchain']['binary_sha256']
env=dict(os.environ,XDG_CACHE_HOME=str(stage/'verification-paper-tectonic-0.17.0/cache'),
    XDG_CONFIG_HOME=str(target/'config'),TMPDIR=str(temporary),TMP=str(temporary),TEMP=str(temporary))
for name in ('TEXINPUTS','TEXMFHOME','TEXMFVAR','TEXMFCONFIG'):env.pop(name,None)
command=[str(engine),'--only-cached','--untrusted','--color','never','--keep-intermediates','--keep-logs',
    '--makefile-rules',str(output/'inputs.mk'),'--outdir',str(output),'main.tex']
started=time.monotonic()
with (output/'compiler.stdout').open('w') as stream:
    result=subprocess.run(command,cwd=snapshot,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=45)
receipt={'exit_code':result.returncode,'seconds':time.monotonic()-started,'source_files':binding['canonical_source_files'],
    'generated_assets':assets,'previous_failed_receipt_sha256':sha(old/'BUILD.json'),
    'script_sha256':sha(__file__),'command':command,'only_cached':True,'runtime_modified':False}
if result.returncode==0:receipt.update(pdf=str(output/'main.pdf'),pdf_sha256=sha(output/'main.pdf'))
else:receipt['errors']=[s for s in (output/'compiler.stdout').read_text().splitlines() if 'error:' in s]
assert all(sha(p)==h for p,h in {**binding['canonical_source_files'],**assets}.items())
(target/'BUILD.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k in ('exit_code','seconds','pdf','errors')}))
raise SystemExit(result.returncode)
