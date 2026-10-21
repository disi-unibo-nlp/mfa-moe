"""Compile editable native figures to standalone vector PDFs, 45 s per figure."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time

from build_cl_manuscript import PAPER, TOOLS, EXPECTED, sha

NAMES = ('overview', 'conditioning', 'readout', 'selection', 'occupancy', 'next_state', 'state_transitions', 'transition_summary', 'routing_mock')


def build(name, allow_cache_fill=False):
    engine = TOOLS / 'tectonic'
    if sha(engine) != EXPECTED:
        raise ValueError('verified compiler changed')
    review = PAPER / '.paper-review'
    root = review / 'figure-builds' / name
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / ('source-' + str(time.time_ns()))
    snapshot.mkdir()
    (snapshot / 'journal').mkdir()
    shutil.copyfile(PAPER / 'journal/style.tex', snapshot / 'journal/style.tex')
    shutil.copytree(PAPER / 'figures/journal', snapshot / 'figures/journal')
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
              'source_sha256': sha(PAPER / 'figures/journal' / (name + '.tex')),
              'style_sha256': sha(PAPER / 'journal/style.tex'),
              'data_files': {p.name: sha(p) for p in (PAPER / 'figures/journal/data').glob('*') if p.is_file()},
              'elapsed_seconds': time.monotonic() - started, 'offline': not allow_cache_fill,
              'new_inference': False, 'source_snapshot': str(snapshot)}
    (PAPER / 'figures/journal' / (name + '.build.json')).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ('figure', 'elapsed_seconds', 'dimensions_pt')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', choices=NAMES)
    parser.add_argument('--allow-cache-fill', action='store_true')
    args = parser.parse_args()
    for name in ([args.figure] if args.figure else NAMES):
        build(name, args.allow_cache_fill)
