"""Compile editable native figures to standalone vector PDFs, 45 s per figure.

Names given on the command line are built; with none, every name is built. Figure sources and
their data live in the support folder (cl_paths.FIGURES) and are compiled in a snapshot that places
them beside the manuscript's figures and icons, under the figures/journal paths they reference; the
PDF goes to the manuscript. A requested figure whose source is missing is an error before anything
compiles.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time

from build_cl_manuscript import TOOLS, EXPECTED, sha
from cl_paths import BUILD, DATA, FIGURES, PAPER

NAMES = ('overview', 'conditioning', 'readout', 'selection', 'occupancy', 'next_state', 'state_transitions', 'transition_summary', 'routing', 'whole_dose', 'whole_states')


def build(name, allow_cache_fill=False):
    if not (FIGURES / (name + '.tex')).is_file():
        raise FileNotFoundError('figure source missing: ' + str(FIGURES / (name + '.tex')))
    engine = TOOLS / 'tectonic'
    if sha(engine) != EXPECTED:
        raise ValueError('verified compiler changed')
    review = BUILD
    root = review / 'figure-builds' / name
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / ('source-' + str(time.time_ns()))
    snapshot.mkdir()
    (snapshot / 'journal').mkdir()
    shutil.copyfile(PAPER / 'journal/style.tex', snapshot / 'journal/style.tex')
    shutil.copytree(PAPER / 'figures/journal', snapshot / 'figures/journal')
    shutil.copytree(FIGURES, snapshot / 'figures/journal', dirs_exist_ok=True)
    source = snapshot / 'main.tex'
    source.write_text(r'''\documentclass[border=1pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{palatino}
\input{journal/style}
\begin{document}
\setlength{\linewidth}{384pt}
\input{figures/journal/''' + name + r'''}
\end{document}
''')
    output = snapshot / 'output'; output.mkdir()
    temporary = snapshot / 'tmp'; temporary.mkdir()
    env = dict(os.environ)
    env.update(XDG_CACHE_HOME=str(TOOLS / 'cache'), XDG_CONFIG_HOME=str(review / 'config'),
               TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for key in ('TEXINPUTS', 'TEXMFHOME', 'TEXMFVAR', 'TEXMFCONFIG'):
        env.pop(key, None)
    cmd = [str(engine), '--untrusted', '--keep-logs', '--keep-intermediates',
           '--outdir', str(output), 'main.tex']
    if not allow_cache_fill:
        cmd.insert(1, '--only-cached')
    started = time.monotonic()
    with (output / 'compiler.log').open('w') as stream:
        run = subprocess.run(cmd, cwd=snapshot, env=env, stdout=stream,
                             stderr=subprocess.STDOUT, timeout=45)
    if run.returncode:
        raise ValueError((output / 'compiler.log').read_text()[-3000:])
    pdf = PAPER / 'figures/journal' / (name + '.pdf')
    shutil.copyfile(output / 'main.pdf', pdf)
    import pymupdf
    with pymupdf.open(pdf) as document:
        if len(document) != 1:
            raise ValueError('figure is not one page')
        document[0].get_pixmap(dpi=200).save(root / 'preview.png')
        dimensions = [document[0].rect.width, document[0].rect.height]
    record = {'figure': name, 'pdf_sha256': sha(pdf), 'dimensions_pt': dimensions,
              'source_sha256': sha(FIGURES / (name + '.tex')),
              'style_sha256': sha(PAPER / 'journal/style.tex'),
              'data_files': {p.name: sha(p) for p in DATA.glob('*') if p.is_file()},
              'elapsed_seconds': time.monotonic() - started, 'offline': not allow_cache_fill,
              'new_inference': False, 'source_snapshot': str(snapshot)}
    (FIGURES / (name + '.build.json')).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ('figure', 'elapsed_seconds', 'dimensions_pt')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('names', nargs='*', metavar='NAME', help='figures to build (default: all of ' + ', '.join(NAMES) + ')')
    parser.add_argument('--figure', action='append', default=[], metavar='NAME', help='same as a positional NAME')
    parser.add_argument('--allow-cache-fill', action='store_true')
    args = parser.parse_args()
    requested = args.figure + args.names
    unknown = [name for name in requested if name not in NAMES]
    if unknown:
        parser.error('unknown figure(s): ' + ', '.join(unknown) + '; choose from ' + ', '.join(NAMES))
    todo = list(dict.fromkeys(requested)) or list(NAMES)
    missing = [name for name in todo if not (FIGURES / (name + '.tex')).is_file()]
    if missing:
        parser.error('figure source missing for requested figure(s): ' + ', '.join(missing))
    for name in todo:
        build(name, args.allow_cache_fill)
