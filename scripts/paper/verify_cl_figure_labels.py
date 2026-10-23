"""Check the routing figure's actual PDF text against bar fills and interval strokes.

The expected labels are the numeric nodes of figures/journal/routing.tex, each checked against the
routing.dat value it annotates. This supplements inspection of the rendered figure and integrated
page. It measures the typeset output, including rotated text bounds and cap strokes.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import re

import pymupdf

from cl_paths import BUILD, FIGURES


LABEL = re.compile(r'[\u2212-]?\d+(?:\.\d+)?')


def expected_labels(root):
    """Label texts of routing.tex, each matched to the routing.dat entry it annotates."""
    rows = (root / 'data/routing.dat').read_text().splitlines()
    names = rows[0].split()
    table = {name: [] for name in names}
    for line in rows[1:]:
        for name, value in zip(names, line.split()):
            table[name].append(float(value))
    column, index, expected = None, 0, []
    for line in (root / 'routing.tex').read_text().splitlines():
        plot = re.search(r'\\addplot.*table\[x=x,y=(\w+),', line)
        if plot:
            column, index = plot[1], 0
            continue
        node = re.search(r'\\node\[rotate=90,anchor=(?:east|west),inner sep=1pt.*\{\$([^$]+)\$\};', line)
        if node:
            text = node[1]
            decimals = len(text.split('.')[1])
            value = table[column][index]
            assert abs(float(text) - value) <= .5 * 10 ** -decimals + 1e-5 * max(1, abs(value)), \
                ('label differs from routing.dat', column, index, text, value)
            expected.append(text)
            index += 1
    return expected


def main():
    root = FIGURES
    pdf = BUILD / 'figure-builds/routing/routing.pdf'
    expected = Counter(expected_labels(FIGURES))
    labels, bars, strokes = [], [], []
    with pymupdf.open(pdf) as document:
        page = document[0]
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                value = ''.join(span['text'] for span in line['spans']).strip().replace('\u2212', '-')
                if line['dir'][1] < -.9 and LABEL.fullmatch(value):
                    labels.append((value, pymupdf.Rect(line['bbox'])))
        for drawing in page.get_drawings():
            for item in drawing['items']:
                if item[0] == 're' and drawing['fill'] is not None:
                    box = item[1]
                    if 4 < box.width < 8:
                        bars.append(box)
                elif item[0] == 'l' and drawing['color'] is not None:
                    a, b = item[1:]
                    box = pymupdf.Rect(min(a.x, b.x), min(a.y, b.y),
                                       max(a.x, b.x), max(a.y, b.y))
                    if box.width < 6 and box.height < 10:
                        pad = max(drawing.get('width', .35), .35) / 2
                        strokes.append(box + (-pad, -pad, pad, pad))
    assert len(labels) == len(bars) == sum(expected.values()) == 108, (len(labels), len(bars), sum(expected.values()))
    found = Counter(value for value, _ in labels)
    assert found == expected, ('numeric label multiset differs', dict(expected - found), dict(found - expected))
    bar_hits = [value for value, box in labels for bar in bars
                if (box & bar).get_area() > .01]
    stroke_hits = [value for value, box in labels for stroke in strokes
                   if (box & stroke).get_area() > .01]
    assert not bar_hits, ('numbers intersect bar fills', bar_hits)
    assert not stroke_hits, ('numbers intersect interval or short strokes', stroke_hits)
    clearance = min(min(abs(box.x0 - bar.x1), abs(bar.x0 - box.x1))
                    for _, box in labels for bar in bars
                    if min(box.y1, bar.y1) > max(box.y0, bar.y0))
    rotated = []
    for source in root.glob('*.tex'):
        for number, line in enumerate(source.read_text().splitlines(), 1):
            if re.search(r'xticklabel style=\{[^}]*rotate=(?!0(?:,|\}))', line):
                rotated.append((source.name, number))
    assert not rotated, ('nonhorizontal x-label declarations', rotated)
    result = {'status': 'pass', 'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
              'numeric_labels': len(labels), 'bars': len(bars),
              'expected_numeric_multiset_present': True,
              'short_stroke_segments_checked': len(strokes),
              'number_bar_overlaps': len(bar_hits),
              'number_interval_or_short_stroke_overlaps': len(stroke_hits),
              'minimum_number_to_bar_clearance_pt': clearance,
              'nonhorizontal_x_label_declarations': len(rotated),
              'visual_inspection_still_required': True}
    receipt = BUILD / 'figure-labels-routing/LABEL_GEOMETRY.json'
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
