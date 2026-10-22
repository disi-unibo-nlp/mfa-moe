"""Register the journal's external editable exhibits and source evidence.

Run paper-writer packet first. This only augments that native packet with the
standalone TikZ sources, copied data, current build and fresh per-page renders.
It never changes installed review tools or judgment rules.
"""
from pathlib import Path
import json
import pymupdf

from build_cl_manuscript import PAPER
from cl_paths import BUILD, DATA, FIGURES, LEGACY, NOTES


def main():
    review = BUILD
    build = json.loads((review / 'BUILD.json').read_text())
    pages = Path(build['page_renders'])
    with pymupdf.open(review / 'build/main.pdf') as document:
        text = '\n\n'.join(f'## page-{i + 1:02d}.png\n' + p.get_text()
                           for i, p in enumerate(document))
    (review / 'pdf-text.md').write_text(text)
    extra = list(FIGURES.glob('*.tex'))
    extra += list(FIGURES.glob('*.json'))
    extra += [p for p in DATA.glob('*') if p.is_file()]
    extra += list((NOTES / 'journal').glob('*.md'))
    extra += list((NOTES / 'journal').glob('*.json'))
    extra += [review / 'author-revision-20261014/PRIMARY_RECORDS.json',
              review / 'bib/OFFLINE_VERIFICATION.json',
              review / 'bib/report.md', review / 'bib-verify.log']
    extra += [review / 'delivery/PACKAGE.json',
              review / 'BUILD.json', review / 'pdf-text.md',
              LEGACY / 'archive/20261014-pre-journal/SOURCE_SNAPSHOT.json',
              LEGACY / 'guiding_weighting_table.tex', LEGACY / 'probe_grouped_summary_table.tex',
              LEGACY / 'expert_selection_audit.tex',
              LEGACY / 'generated/seven_model_evidence.tex',
              LEGACY / 'generated/verification_status.tex']
    metadata = json.loads((review / 'packet.json').read_text())
    metadata['files'] = list(dict.fromkeys(metadata['files'] + [str(p) for p in extra]))
    metadata['review_pages'] = str(pages)
    (review / 'packet.json').write_text(json.dumps(metadata, indent=1) + '\n')
    packet = (review / 'packet.md').read_text()
    packet = packet.replace('Pages: .paper-review/pages/',
                            'Pages: ' + str(pages.relative_to(PAPER)) + '/')
    packet += '\n\n## Additional registered artifacts\n'
    packet += '\n'.join('- ' + str(p.relative_to(PAPER)) for p in extra)
    packet += '\n\nUse the unique current page directory above, not an older cached image.\n'
    (review / 'packet.md').write_text(packet)
    print(json.dumps({'judge_items': len(metadata['rules']), 'page_directory': str(pages),
                      'page_count': build['page_count'], 'registered_text_files': len(metadata['files'])}))


if __name__ == '__main__':
    main()
