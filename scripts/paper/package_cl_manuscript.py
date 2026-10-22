"""Package the current journal source and verify a bounded unpacked offline build."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import time
import zipfile

import pymupdf

from build_cl_manuscript import PAPER, SKILL, TOOLS, EXPECTED, sha
from cl_paths import BUILD


def page_proofs(path):
    with pymupdf.open(path) as doc:
        count = len(doc)
        texts = [p.get_text() for p in doc]
    renders = []
    for i in range(count):
        with pymupdf.open(path) as doc:
            pixels = doc[i].get_pixmap(matrix=pymupdf.Matrix(2, 2)).samples
            renders.append(hashlib.sha256(pixels).hexdigest())
    return texts, renders


def link_proofs(path):
    with pymupdf.open(path) as document:
        names = document.resolve_names()
        links = []
        for page in document:
            current = []
            for link in page.get_links():
                fields = {key: link[key] for key in
                          ('kind', 'page', 'nameddest', 'uri', 'file') if key in link}
                for key in ('from', 'to'):
                    if key in link:
                        value = link[key]
                        fields[key] = (tuple(round(float(x), 6) for x in value)
                                       if isinstance(value, (pymupdf.Point, pymupdf.Rect)) else value)
                current.append(fields)
            links.append(current)
    return names, links


def main():
    review = BUILD
    record = json.loads((review / 'BUILD.json').read_text())
    current_pdf = review / 'build/main.pdf'
    if record['exit_code'] or sha(current_pdf) != record['pdf_sha256']:
        raise ValueError('current build is not verified')
    for name, expected in record['source_hashes'].items():
        if sha(PAPER / name) != expected:
            raise ValueError('source changed since build: ' + name)
    root = review / 'delivery'; root.mkdir(exist_ok=True)
    nonce = str(time.time_ns())
    source = root / ('source-' + nonce); source.mkdir()
    spec = importlib.util.spec_from_file_location('paper_writer_rules', SKILL / 'scripts/rules.py')
    rules = importlib.util.module_from_spec(spec); spec.loader.exec_module(rules)
    files = rules.active_files(PAPER / 'main.tex')
    files += [PAPER / 'clv2025.cls', PAPER / 'compling.bst']
    # The source package is the manuscript folder: sections, tables, figure PDFs, icons, class,
    # bibliography style and references.
    files += [PAPER / 'references.bib']
    files += [p for p in (PAPER / 'figures/journal').rglob('*') if p.is_file()]
    bindings = {}
    for origin in dict.fromkeys(files):
        name = str(origin.relative_to(PAPER))
        target = source / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target); bindings[name] = sha(origin)
    manifest = {'entry': 'main.tex', 'files': bindings,
                'build_pdf_sha256': record['pdf_sha256'],
                'compiler': 'Tectonic 0.17.0', 'compiler_sha256': EXPECTED,
                'science_recomputed': False, 'remote_publication': False}
    (source / 'SOURCE_PACKAGE.json').write_text(json.dumps(manifest, indent=2) + '\n')
    archive = root / 'cl-submission-source.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(source.rglob('*')):
            if p.is_file(): z.write(p, p.relative_to(source))
    unpacked = root / ('unpacked-' + nonce); unpacked.mkdir()
    with zipfile.ZipFile(archive) as z: z.extractall(unpacked)
    for name, expected in bindings.items():
        if sha(unpacked / name) != expected: raise ValueError('archive readback differs: ' + name)
    engine = TOOLS / 'tectonic'
    if sha(engine) != EXPECTED: raise ValueError('verified compiler changed')
    output = unpacked / 'output'; output.mkdir()
    temporary = unpacked / 'tmp'; temporary.mkdir()
    env = dict(os.environ)
    env.update(XDG_CACHE_HOME=str(TOOLS / 'cache'), XDG_CONFIG_HOME=str(review / 'config'),
               TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    for key in ('TEXINPUTS', 'TEXMFHOME', 'TEXMFVAR', 'TEXMFCONFIG'): env.pop(key, None)
    command = [str(engine), '--untrusted', '--only-cached', '--keep-logs',
               '--keep-intermediates', '--reruns', '0',
               '--outdir', str(output), 'main.tex']
    start = time.monotonic()
    passes = []
    for number in range(1, 5):
        # Keep each compiler invocation bounded while allowing bibliography
        # and cross-reference convergence in the independently unpacked copy.
        for suffix in ('aux', 'bbl', 'out', 'toc'):
            auxiliary = output / ('main.' + suffix)
            if auxiliary.exists():
                shutil.copyfile(auxiliary, unpacked / auxiliary.name)
        pass_start = time.monotonic()
        log = output / f'compiler-pass-{number}.log'
        with log.open('w') as stream:
            run = subprocess.run(command, cwd=unpacked, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, timeout=45)
        if run.returncode: raise ValueError(log.read_text()[-3000:])
        text = (output / 'main.log').read_text(errors='replace')
        pending = bool(re.search(r'Rerun to get cross-references right|Label\(s\) may have changed|There were undefined (?:references|citations)|Citation .+ undefined', text))
        passes.append({'pass': number, 'exit_code': run.returncode,
                       'elapsed_seconds': time.monotonic() - pass_start,
                       'rerun_needed': pending})
        if not pending:
            break
    else:
        raise ValueError('unpacked references did not converge in four bounded passes')
    before_text, before_render = page_proofs(current_pdf)
    after_text, after_render = page_proofs(output / 'main.pdf')
    if (before_text, before_render) != (after_text, after_render):
        raise ValueError('unpacked source differs in page text or rendering')
    if link_proofs(current_pdf) != link_proofs(output / 'main.pdf'):
        raise ValueError('unpacked source differs in link targets or named destinations')
    delivered = root / 'cl-submission-draft.pdf'; shutil.copyfile(current_pdf, delivered)
    result = {'source_package': str(archive), 'source_package_sha256': sha(archive),
              'pdf': str(delivered), 'pdf_sha256': sha(delivered),
              'unpacked_build_exit_code': run.returncode, 'offline': True,
              'elapsed_build_seconds': time.monotonic() - start,
              'bounded_compile_passes': passes,
              'page_count': len(before_text), 'all_page_text_equal': True,
              'all_page_renders_equal': True, 'all_link_targets_equal': True,
              'source_files': len(bindings),
              'remote_publication': False, 'scientific_computation': False}
    (root / 'PACKAGE.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
