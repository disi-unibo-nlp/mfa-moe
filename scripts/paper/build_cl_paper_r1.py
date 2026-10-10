"""Compile a Computational Linguistics source snapshot with the verified TeX engine.

Each invocation has a 45-second development bound. It never installs into the
site environment, edits paper sources or submits scientific work. A populated
private resource cache can subsequently be used offline with --only-cached.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from build_verification_paper_r1 import PAPER, STAGING, private, publish, sha

TOOLCHAIN = STAGING / "verification-paper-tectonic-0.17.0/TOOLCHAIN.json"
CACHE = STAGING / "verification-paper-tectonic-0.17.0/cache"
EXPECTED = "a98aa59ad5c1df39a6c9e56cbfc5088f2b11d6c179c0130b97998e4bd46a46da"


def build(directory, only_cached=False):
    toolchain = json.loads(TOOLCHAIN.read_text())
    engine = private(Path(toolchain['binary_path']))
    if toolchain['binary_sha256'] != EXPECTED or sha(engine) != EXPECTED:
        raise ValueError("standalone compiler differs from the verified release")
    directory = private(directory)
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = directory / "source"; snapshot.mkdir()
    sources = {}
    for path in PAPER.rglob("*"):
        if not path.is_file() or not (path.suffix in {".tex", ".cls", ".bib", ".bst", ".sty", ".bbx", ".cbx", ".dbx"}
                or "figures" in path.relative_to(PAPER).parts
                or ("generated" in path.relative_to(PAPER).parts and path.suffix in {".pdf", ".png"})):
            continue
        sources[str(path)] = sha(path)
        target = snapshot / path.relative_to(PAPER)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha(path) != sources[str(path)] or sha(target) != sources[str(path)]:
            raise ValueError("canonical paper source changed during snapshot")
    font_path = TOOLCHAIN.with_name('FONT_ASSETS.json')
    font_assets = json.loads(font_path.read_text()) if font_path.exists() else {'files':{}}
    for filename, value in font_assets['files'].items():
        origin = private(Path(value['path']))
        if Path(filename).name != filename or not filename.endswith('.otf') or sha(origin) != value['sha256']:
            raise ValueError('font asset filename or hash differs')
        shutil.copyfile(origin,snapshot/filename)
    output = directory / 'output'; output.mkdir()
    temporary = directory / 'tmp'; temporary.mkdir()
    environment = dict(os.environ)
    environment.update(XDG_CACHE_HOME=str(CACHE), XDG_CONFIG_HOME=str(directory/'config'),
        TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary))
    for name in ('TEXINPUTS','TEXMFHOME','TEXMFVAR','TEXMFCONFIG'):
        environment.pop(name,None)
    command = [str(engine),'--untrusted','--color','never','--keep-intermediates','--keep-logs',
        '--makefile-rules',str(output/'inputs.mk'),'--outdir',str(output),'main.tex']
    if only_cached:
        command.insert(1,'--only-cached')
    start = time.time(); state = 'BUILD_FAILED'; error = None; code = None
    with (output/'compiler.stdout').open('w') as stdout:
        try:
            process = subprocess.run(command,cwd=snapshot,env=environment,
                stdout=stdout,stderr=subprocess.STDOUT,timeout=45)
            code = process.returncode
            if code == 0 and (output/'main.pdf').exists():
                state = 'COMPILED_FULL_CLASS'
        except subprocess.TimeoutExpired:
            state = 'DEVELOPMENT_TIME_BOUND'
            error = '45-second bound reached; inspect retained cache before another invocation'
    changed = [path for path,expected in sources.items() if sha(Path(path)) != expected]
    if changed:
        state = 'CANONICAL_SOURCE_CHANGED_DURING_BUILD'
    chatter = (output/'compiler.stdout').read_text(errors='replace')
    cache_files = {str(p):sha(p) for p in CACHE.rglob('*') if p.is_file()}
    result = {'schema':'computational-linguistics-paper-build-v1','state':state,'observed_epoch':time.time(),
        'elapsed_seconds':time.time()-start,'exit_code':code,'error':error,
        'canonical_source_files':sources,'source_snapshot':str(snapshot),'changed_sources':changed,
        'toolchain_receipt_sha256':sha(TOOLCHAIN),'toolchain':toolchain,'build_script_sha256':sha(Path(__file__)),
        'helper_sha256':sha(Path(__file__).with_name('build_verification_paper_r1.py')),
        'font_assets':font_assets,
        'command':command,'only_cached':only_cached,'cache_files':cache_files,
        'cache_bytes':sum(Path(p).stat().st_size for p in cache_files),
        'stdout_path':str(output/'compiler.stdout'),
        'diagnostics':[line.strip() for line in chatter.splitlines() if 'error:' in line or 'warning:' in line],
        'installed_runtime_modified':False,'scientific_dispatch':'NONE'}
    if state == 'COMPILED_FULL_CLASS':
        pdf = output/'main.pdf'
        result.update(pdf_path=str(pdf),pdf_sha256=sha(pdf),pdf_bytes=pdf.stat().st_size)
    publish(directory/'BUILD.json',result)
    brief = {key:result[key] for key in ('state','elapsed_seconds','exit_code','error','cache_bytes')}
    brief['diagnostic_count'] = len(result['diagnostics'])
    brief['errors'] = [line for line in result['diagnostics'] if 'error:' in line][:10]
    print(json.dumps(brief,indent=2))
    return 0 if state == 'COMPILED_FULL_CLASS' else 1


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--only-cached',action='store_true')
    args=parser.parse_args()
    raise SystemExit(build(args.output,args.only_cached))
