"""Build an isolated active-source snapshot, excluding review/history artifacts."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / 'paper/6a510c6db0d9b08aa438c142'
STAGING = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp')
TOOLS = STAGING / 'verification-paper-tectonic-0.17.0'
EXPECTED = 'a98aa59ad5c1df39a6c9e56cbfc5088f2b11d6c179c0130b97998e4bd46a46da'
REFERENCES = re.compile(r'\\(input|include|subfile|bibliography|addbibresource|documentclass|bibliographystyle|usepackage|RequirePackage|includegraphics)\*?(?:\[[^\]]*\])?\s*\{([^}]+)\}')
EXTENSIONS = {'input':('.tex',), 'include':('.tex',), 'subfile':('.tex',),
    'bibliography':('.bib',), 'addbibresource':('.bib',), 'documentclass':('.cls',),
    'bibliographystyle':('.bst',), 'usepackage':('.sty',), 'RequirePackage':('.sty',),
    'includegraphics':('.pdf','.png','.jpg','.jpeg','.eps')}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def active_sources(paper=PAPER):
    paper = Path(paper).resolve(); queue = [paper / 'main.tex']; found = set()
    while queue:
        path = queue.pop(0).resolve()
        if path in found:
            continue
        if paper not in path.parents or any(part.startswith('.') for part in path.relative_to(paper).parts):
            raise ValueError('Active source escapes the canonical input tree')
        if not path.is_file():
            raise ValueError('Required active manuscript source is absent: ' + str(path))
        found.add(path)
        if path.suffix not in ('.tex', '.cls', '.sty'):
            continue
        text = '\n'.join(re.sub(r'(?<!\\)%.*', '', line) for line in path.read_text(errors='replace').splitlines())
        for command, raw in REFERENCES.findall(text):
            for name in raw.split(','):
                name = name.strip()
                if not name:
                    continue
                if '#' in name:
                    # Locally defined icon macros choose a filename at runtime.
                    directory = (paper / name.rsplit('/', 1)[0]).resolve() if '/' in name else paper
                    if directory == paper or paper in directory.parents:
                        queue.extend(p for p in directory.iterdir() if p.is_file() and p.suffix in EXTENSIONS[command])
                    continue
                options = []
                for directory in (paper, path.parent):
                    base = directory / name
                    options.extend([base] if base.suffix else [Path(str(base) + ext) for ext in EXTENSIONS[command]])
                target = next((p for p in options if p.is_file()), None)
                if target is not None:
                    queue.append(target)
                elif command not in ('documentclass', 'bibliographystyle', 'usepackage', 'RequirePackage'):
                    raise ValueError('Required dependency is absent: ' + name)
    return sorted(found)


def source_state(paper=PAPER):
    return {str(path): sha(path) for path in active_sources(paper)}


def build(directory, only_cached=True):
    directory = Path(directory).resolve()
    if STAGING not in directory.parents:
        raise ValueError('Snapshot must use the authorized private staging area')
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    snapshot = directory / 'source'; snapshot.mkdir()
    before = source_state()
    for name, expected in before.items():
        origin = Path(name); target = snapshot / origin.relative_to(PAPER)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
        if sha(origin) != expected or sha(target) != expected:
            raise ValueError('Canonical input changed during snapshot')
    toolchain = json.loads((TOOLS / 'TOOLCHAIN.json').read_text())
    engine = Path(toolchain['binary_path'])
    if toolchain['binary_sha256'] != EXPECTED or sha(engine) != EXPECTED:
        raise ValueError('Verified compiler changed')
    fonts = json.loads((TOOLS / 'FONT_ASSETS.json').read_text())
    for name, record in fonts.get('files', {}).items():
        if Path(name).name != name or sha(Path(record['path'])) != record['sha256']:
            raise ValueError('Verified font changed')
        shutil.copyfile(record['path'], snapshot / name)
    output = directory / 'output'; output.mkdir()
    temporary = directory / 'tmp'; temporary.mkdir()
    env = dict(os.environ)
    env.update(XDG_CACHE_HOME=str(TOOLS / 'cache'), XDG_CONFIG_HOME=str(directory / 'config'),
               TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary))
    for key in ('TEXINPUTS','TEXMFHOME','TEXMFVAR','TEXMFCONFIG'):
        env.pop(key, None)
    command = [str(engine), '--untrusted', '--color', 'never', '--keep-intermediates', '--keep-logs',
               '--makefile-rules', str(output / 'inputs.mk'), '--outdir', str(output), 'main.tex']
    if only_cached:
        command.insert(1, '--only-cached')
    start = time.monotonic(); code = None; error = None
    with (output / 'compiler.stdout').open('w') as stream:
        try:
            result = subprocess.run(command, cwd=snapshot, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=45)
            code = result.returncode
        except subprocess.TimeoutExpired:
            error = '45-second compiler bound reached'
    after = source_state(); changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    pdf = output / 'main.pdf'
    state = 'COMPILED_FULL_CLASS' if code == 0 and pdf.exists() and not changed else 'BUILD_FAILED'
    receipt = {'state': state, 'exit_code': code, 'error': error, 'elapsed_seconds': time.monotonic() - start,
        'canonical_source_files': before, 'source_snapshot': str(snapshot), 'changed_sources': changed,
        'source_scope': 'active TeX, local class/style/bibliography and referenced graphics; review artifacts excluded',
        'build_script_sha256': sha(Path(__file__)), 'toolchain_receipt_sha256': sha(TOOLS / 'TOOLCHAIN.json'),
        'font_assets': fonts, 'command': command, 'only_cached': only_cached,
        'canonical_manuscript_writes': 0, 'scientific_dispatch': 'NONE'}
    if pdf.exists():
        receipt.update(pdf_path=str(pdf), pdf_sha256=sha(pdf), pdf_bytes=pdf.stat().st_size)
    (directory / 'BUILD.json').write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('state','exit_code','elapsed_seconds','changed_sources')}))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only-cached', action='store_true')
    args = parser.parse_args()
    raise SystemExit(build(args.output, args.only_cached)['state'] != 'COMPILED_FULL_CLASS')
