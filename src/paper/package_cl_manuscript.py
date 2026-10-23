"""Package the current manuscript sources as a zip and check that the unpacked copy builds the same PDF.

Needs a finished build (python -m paper.build_cl_manuscript). Writes BUILD/delivery/cl-submission-source.zip, the
current PDF as cl-submission-draft.pdf and PACKAGE.json. The unpacked sources are compiled offline in passes of
at most --timeout-seconds; their page texts, page images and link targets must equal those of the current build.

  python -m paper.package_cl_manuscript [--timeout-seconds 300]
"""
import argparse
import hashlib
import json
import shutil
import time
import zipfile

import pymupdf

from paper.build_cl_manuscript import PASS_SECONDS, compile_bounded, compile_env, copy_sources, seconds, source_files, tectonic
from paper.cl_paths import BUILD, PAPER, TECTONIC_SHA256
from store import file_sha


def page_proofs(path):
    """Text and a hash of the pixels of every page."""
    with pymupdf.open(path) as doc:
        texts = [p.get_text() for p in doc]
        renders = [hashlib.sha256(p.get_pixmap(matrix=pymupdf.Matrix(2, 2)).samples).hexdigest() for p in doc]
    return texts, renders


def link_proofs(path):
    """Named destinations and the links of every page."""
    with pymupdf.open(path) as document:
        names = document.resolve_names()
        links = []
        for page in document:
            current = []
            for link in page.get_links():
                fields = {key: link[key] for key in ('kind', 'page', 'nameddest', 'uri', 'file') if key in link}
                for key in ('from', 'to'):
                    if key in link:
                        value = link[key]
                        fields[key] = (tuple(round(float(x), 6) for x in value)
                                       if isinstance(value, (pymupdf.Point, pymupdf.Rect)) else value)
                current.append(fields)
            links.append(current)
    return names, links


def main(timeout):
    record = json.loads((BUILD / 'BUILD.json').read_text())
    current_pdf = BUILD / 'build/main.pdf'
    if record['exit_code'] or file_sha(current_pdf) != record['pdf_sha256']:
        raise ValueError('current build is not verified')
    for name, expected in record['source_hashes'].items():
        if file_sha(PAPER / name) != expected:
            raise ValueError('source changed since build: ' + name)
    root = BUILD / 'delivery'
    root.mkdir(exist_ok=True)
    nonce = str(time.time_ns())
    source = root / ('source-' + nonce)
    source.mkdir()
    # The original figure PDFs and the review notes travel with the sources but are not used to compile.
    bindings = copy_sources(source_files(PAPER / 'figures'), source)
    manifest = {'entry': 'main.tex', 'files': bindings, 'build_pdf_sha256': record['pdf_sha256'],
                'compiler': 'Tectonic 0.17.0', 'compiler_sha256': TECTONIC_SHA256,
                'science_recomputed': False, 'remote_publication': False}
    (source / 'SOURCE_PACKAGE.json').write_text(json.dumps(manifest, indent=2) + '\n')
    archive = root / 'cl-submission-source.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(source.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(source))
    unpacked = root / ('unpacked-' + nonce)
    unpacked.mkdir()
    with zipfile.ZipFile(archive) as z:
        z.extractall(unpacked)
    for name, expected in bindings.items():
        if file_sha(unpacked / name) != expected:
            raise ValueError('archive readback differs: ' + name)
    (unpacked / 'tmp').mkdir()
    start = time.monotonic()
    run, passes = compile_bounded(unpacked, tectonic(), compile_env(unpacked / 'tmp'), timeout)
    if run.returncode:
        raise ValueError('unpacked sources did not build in four bounded passes: ' + str(passes))
    rebuilt = unpacked / '.paper-review/build/main.pdf'
    before_text, before_render = page_proofs(current_pdf)
    if (before_text, before_render) != page_proofs(rebuilt):
        raise ValueError('unpacked source differs in page text or rendering')
    if link_proofs(current_pdf) != link_proofs(rebuilt):
        raise ValueError('unpacked source differs in link targets or named destinations')
    delivered = root / 'cl-submission-draft.pdf'
    shutil.copyfile(current_pdf, delivered)
    result = {'source_package': str(archive), 'source_package_sha256': file_sha(archive),
              'pdf': str(delivered), 'pdf_sha256': file_sha(delivered),
              'unpacked_build_exit_code': run.returncode, 'offline': True,
              'elapsed_build_seconds': time.monotonic() - start, 'bounded_compile_passes': passes,
              'page_count': len(before_text), 'all_page_text_equal': True,
              'all_page_renders_equal': True, 'all_link_targets_equal': True,
              'source_files': len(bindings), 'remote_publication': False, 'scientific_computation': False}
    (root / 'PACKAGE.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--timeout-seconds', type=seconds, default=PASS_SECONDS, help='time limit of each compiler pass')
    main(parser.parse_args().timeout_seconds)
