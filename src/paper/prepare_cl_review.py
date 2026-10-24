"""Add the journal's figure sources, data, notes and evidence files to a paper-writer review packet.

Run `pw packet` first; this only extends BUILD/packet.json and BUILD/packet.md with those files and points the
packet at the page images of the current build (BUILD/BUILD.json). It also writes the PDF text to BUILD/pdf-text.md.
It changes no tool and no rule.

  python -m paper.prepare_cl_review
"""
from pathlib import Path
import json
import os

import pymupdf

from paper.cl_paths import BUILD, DATA, FIGURES, NOTES, PAPER, SAVED_REPORTS

EVIDENCE_FILES = (
    'archive/20261014-pre-journal/SOURCE_SNAPSHOT.json', 'guiding_weighting_table.tex',
    'probe_grouped_summary_table.tex', 'expert_selection_audit.tex', 'generated/seven_model_evidence.tex',
    'generated/verification_status.tex')


def main():
    build = json.loads((BUILD / 'BUILD.json').read_text())
    pages = Path(build['page_renders'])
    with pymupdf.open(BUILD / 'build/main.pdf') as document:
        text = '\n\n'.join(f'## page-{i + 1:02d}.png\n' + p.get_text() for i, p in enumerate(document))
    (BUILD / 'pdf-text.md').write_text(text)
    extra = list(FIGURES.glob('*.tex')) + list(FIGURES.glob('*.json'))
    extra += [p for p in DATA.glob('*') if p.is_file()]
    extra += list((NOTES / 'journal').glob('*.md')) + list((NOTES / 'journal').glob('*.json'))
    extra += list(PAPER.glob('*.md'))
    extra += [BUILD / 'author-revision-20261014/PRIMARY_RECORDS.json', BUILD / 'bib/OFFLINE_VERIFICATION.json',
              BUILD / 'bib/report.md', BUILD / 'bib-verify.log', BUILD / 'delivery/PACKAGE.json',
              BUILD / 'BUILD.json', BUILD / 'pdf-text.md']
    extra += [SAVED_REPORTS / name for name in EVIDENCE_FILES]
    metadata = json.loads((BUILD / 'packet.json').read_text())
    metadata['files'] = list(dict.fromkeys(metadata['files'] + [str(p) for p in extra]))
    metadata['review_pages'] = str(pages)
    (BUILD / 'packet.json').write_text(json.dumps(metadata, indent=1) + '\n')
    packet = (BUILD / 'packet.md').read_text()
    packet = packet.replace('Pages: .paper-review/pages/', 'Pages: ' + os.path.relpath(pages, PAPER) + '/')
    packet += '\n\n## Additional registered artifacts\n'
    packet += '\n'.join('- ' + os.path.relpath(p, PAPER) for p in extra)
    packet += '\n\nUse the unique current page directory above, not an older cached image.\n'
    (BUILD / 'packet.md').write_text(packet)
    print(json.dumps({'judge_items': len(metadata['rules']), 'page_directory': str(pages),
                      'page_count': build['page_count'], 'registered_text_files': len(metadata['files'])}))


if __name__ == '__main__':
    main()
