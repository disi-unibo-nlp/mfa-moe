"""Compile the native figure sources (figures/journal/NAME.tex) to one-page PDFs with the offline tectonic.

Names on the command line are built; with none, every figure is built. Each figure is compiled in a copy of the
sources; the PDF, a preview image and a receipt go to BUILD/figure-builds/NAME. The manuscript folder is only read.

  python -m paper.build_cl_figures [NAME ...] [--timeout-seconds 120]
"""
import argparse
import json
import shutil
import subprocess
import time

import pymupdf

from paper.build_cl_manuscript import compile_env, tectonic
from paper.cl_paths import BUILD, DATA, FIGURES, PAPER
from store import file_sha

NAMES = ('overview', 'conditioning', 'readout', 'selection', 'occupancy', 'next_state', 'state_transitions',
         'transition_summary', 'routing', 'whole_dose', 'whole_states')

PAGE = r'''\documentclass[border=1pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{palatino}
\input{journal/style}
\begin{document}
\setlength{\linewidth}{384pt}
\input{figures/journal/%s}
\end{document}
'''


def build(name, engine, timeout):
    """Compile one figure with `engine`; writes its PDF, preview and receipt."""
    root = BUILD / 'figure-builds' / name
    snapshot = root / ('source-' + str(time.time_ns()))
    (snapshot / 'journal').mkdir(parents=True)
    shutil.copyfile(PAPER / 'journal/style.tex', snapshot / 'journal/style.tex')
    shutil.copytree(FIGURES, snapshot / 'figures/journal')
    (snapshot / 'main.tex').write_text(PAGE % name)
    output, temporary = snapshot / 'output', snapshot / 'tmp'
    output.mkdir()
    temporary.mkdir()
    command = [str(engine), '--only-cached', '--untrusted', '--keep-logs', '--keep-intermediates',
               '--outdir', str(output), 'main.tex']
    started = time.monotonic()
    with (output / 'compiler.log').open('w') as stream:
        run = subprocess.run(command, cwd=snapshot, env=compile_env(temporary), stdout=stream,
                             stderr=subprocess.STDOUT, timeout=timeout)
    if run.returncode:
        raise ValueError((output / 'compiler.log').read_text()[-3000:])
    pdf = root / (name + '.pdf')
    shutil.copyfile(output / 'main.pdf', pdf)
    with pymupdf.open(pdf) as document:
        if len(document) != 1:
            raise ValueError('figure is not one page')
        document[0].get_pixmap(dpi=200).save(root / 'preview.png')
        dimensions = [document[0].rect.width, document[0].rect.height]
    record = {
        'figure': name,
        'pdf_path': str(pdf),
        'pdf_sha256': file_sha(pdf),
        'dimensions_pt': dimensions,
        'source_sha256': file_sha(FIGURES / (name + '.tex')),
        'style_sha256': file_sha(PAPER / 'journal/style.tex'),
        'data_files': {p.name: file_sha(p) for p in DATA.glob('*') if p.is_file()},
        'elapsed_seconds': time.monotonic() - started,
        'timeout_seconds': timeout,
        'offline': True,
        'new_inference': False,
        'source_snapshot': str(snapshot),
    }
    (root / 'BUILD.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ('figure', 'elapsed_seconds', 'dimensions_pt')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('names', nargs='*', metavar='NAME', help='figures to build (default: ' + ', '.join(NAMES) + ')')
    parser.add_argument('--timeout-seconds', type=int, default=120, help='time limit of each figure (the slowest takes about 65 s)')
    args = parser.parse_args()
    if not 0 < args.timeout_seconds <= 600:
        parser.error('--timeout-seconds must be between 1 and 600')
    unknown = [name for name in args.names if name not in NAMES]
    if unknown:
        parser.error('unknown figure(s): ' + ', '.join(unknown) + '; choose from ' + ', '.join(NAMES))
    todo = list(dict.fromkeys(args.names)) or list(NAMES)
    missing = [name for name in todo if not (FIGURES / (name + '.tex')).is_file()]
    if missing:
        parser.error('figure source missing for requested figure(s): ' + ', '.join(missing))
    engine = tectonic()
    for name in todo:
        build(name, engine, args.timeout_seconds)
