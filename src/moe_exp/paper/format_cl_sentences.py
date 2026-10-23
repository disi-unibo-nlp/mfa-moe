"""Apply the author's LaTeX sentence separators without changing prose tokens.

Math, comments, identifiers, literal blocks and abbreviations are protected.
Wrapped prose is joined so each sentence occupies one source line.
The formatter checks both text-token preservation and paragraph preservation.
"""
from pathlib import Path
import argparse
import json
import re

PAPER = Path(__file__).resolve().parents[3] / 'paper/6a510c6db0d9b08aa438c142'
MATH_ENV = re.compile(r'\\begin\{(equation\*?|align\*?|alignat\*?|gather\*?|multline\*?|displaymath|verbatim|lstlisting)\}.*?\\end\{\1\}', re.S)
IDENTIFIER = re.compile(r'\\(?:label|ref|eqref|pageref|cite[tp]?|citeauthor|citeyear|url|path|nolinkurl|texttt|input|includegraphics|bibliography|bibliographystyle|documentclass|usepackage)(?:\[[^\]]*\])?\{')
ABBREVIATION = re.compile(r'(?:\b(?:Fig|Figs|Eq|Eqs|Sec|Secs|App|Tab|Tabs|Dr|Prof|Mr|Ms|vs|etc|Min)|\be\.g|\bi\.e|\bet al|\b[A-Z])\.$')
LITERAL_ENV = re.compile(r'\\begin\{(tabular\*?|longtable|verbatim|lstlisting)\}.*?\\end\{\1\}', re.S)
PROSE_COMMANDS = {'caption', 'item', 'textbf', 'textit', 'textsc', 'textrm',
                  'emph', 'noindent', 'cid', 'bestcid', 'secondcid', 'modelname',
                  'best', 'second', 'footnote', 'cite', 'citet', 'citep', 'url',
                  'texttt'}


def escaped(text, index):
    count = 0
    while index > 0 and text[index - 1] == '\\':
        count += 1
        index -= 1
    return count % 2 == 1


def protected(text):
    mask = bytearray(len(text))
    def mark(a, b):
        mask[a:b] = b'\1' * (b-a)
    for match in MATH_ENV.finditer(text):
        mark(*match.span())
    for match in IDENTIFIER.finditer(text):
        end, depth = match.end(), 1
        while end < len(text) and depth:
            if text[end] == '{' and not escaped(text, end): depth += 1
            if text[end] == '}' and not escaped(text, end): depth -= 1
            end += 1
        mark(match.start(), end)
    i = 0
    while i < len(text):
        if mask[i]:
            i += 1
            continue
        if text[i] == '%' and not escaped(text, i):
            end = text.find('\n', i)
            end = len(text) if end < 0 else end
            mark(i, end)
            i = end
        elif text[i] == '$' and not escaped(text, i):
            delimiter = '$$' if text.startswith('$$', i) else '$'
            end = i + len(delimiter)
            while end < len(text):
                if text.startswith(delimiter, end) and not escaped(text, end):
                    end += len(delimiter)
                    break
                end += 1
            mark(i, end)
            i = end
        elif text.startswith(('\\(', '\\['), i):
            close = '\\)' if text[i+1] == '(' else '\\]'
            end = text.find(close, i+2)
            end = len(text) if end < 0 else end+2
            mark(i, end)
            i = end
        else:
            i += 1
    return mask


def normalized(text):
    return re.sub(r'\s+', ' ', re.sub(r'(?m)^[ \t]*%[ \t]*\n', '', text)).strip()


def paragraph_count(text):
    text = re.sub(r'(?m)^[ \t]*%[ \t]*\n', '', text)
    return len(re.findall(r'\n[ \t]*\n', text))


def unwrap_prose(text):
    """Join prose wraps, keeping structural lines, comments and displays intact."""
    mask = bytearray(len(text))
    for pattern in (MATH_ENV, LITERAL_ENV):
        for match in pattern.finditer(text):
            mask[match.start():match.end()] = b'\1' * len(match[0])
    abstract = text.find('\\begin{abstract}')
    if abstract >= 0:
        mask[:abstract] = b'\1' * abstract

    def structural(line):
        value = line.strip()
        if not value or value.startswith('%'):
            return True
        # Literal comments must keep their original line endings.
        if any(value[i] == '%' and not escaped(value, i) for i in range(len(value))):
            return True
        command = re.match(r'\\([A-Za-z]+)', value)
        return bool(command and command[1] not in PROSE_COMMANDS)

    changes = []
    for match in re.finditer(r'[^\S\n]*\n[^\S\n]*', text):
        a, b = match.span()
        if any(mask[a:b]):
            continue
        left = text[text.rfind('\n', 0, a)+1:a]
        end = text.find('\n', b)
        right = text[b:len(text) if end < 0 else end]
        if structural(left) or structural(right):
            continue
        changes.append((a, b, ' '))
    for a, b, replacement in reversed(changes):
        text = text[:a]+replacement+text[b:]
    return text, len(changes)


def format_text(text):
    mask = protected(text)
    changes = []
    candidates = []
    for match in re.finditer(r'([.!?])([ \t\r\n]+)', text):
        candidates.append((match.start(), match[1], match.start(2), match.end(2), match[2]))
    # A footnote belongs to the preceding sentence; its literal contents must
    # stay attached, with the separator inserted after the balanced argument.
    for match in re.finditer(r'([.!?])\\footnote(?:\[[^\]]*\])?\{', text):
        end, depth = match.end(), 1
        while end < len(text) and depth:
            if text[end] == '{' and not escaped(text, end): depth += 1
            if text[end] == '}' and not escaped(text, end): depth -= 1
            end += 1
        whitespace = re.match(r'[ \t\r\n]+', text[end:])
        if depth == 0 and whitespace:
            candidates.append((match.start(), match[1], end, end+whitespace.end(), whitespace[0]))
    for index, punctuation, start, end, whitespace in candidates:
        if mask[index] or escaped(text, index): continue
        if punctuation == '.' and ABBREVIATION.search(text[max(0,index-20):index+1]): continue
        if text.startswith('\n%\n', start): continue
        following = text[end:]
        if following.startswith('%'): continue
        newlines = whitespace.count('\n')
        replacement = '\n%\n' + '\n' * max(0, newlines-1)
        changes.append((start, end, replacement))
    result = text
    for a,b,replacement in sorted(changes, reverse=True): result = result[:a]+replacement+result[b:]
    assert normalized(text) == normalized(result), 'formatting changed prose tokens'
    assert paragraph_count(text) == paragraph_count(result), 'formatting changed paragraph boundaries'
    result, joined = unwrap_prose(result)
    assert normalized(text) == normalized(result), 'unwrapping changed prose tokens'
    assert paragraph_count(text) == paragraph_count(result), 'unwrapping changed paragraph boundaries'
    return result, len(changes) + joined


def main(check=False):
    files = [PAPER/'main.tex'] + [p for p in (PAPER/'journal').rglob('*.tex') if p.name != 'style.tex']
    pending = {}
    for path in files:
        before = path.read_text()
        after, count = format_text(before)
        if count:
            pending[str(path.relative_to(PAPER))] = count
            if not check: path.write_text(after)
    print(json.dumps({'mode':'check' if check else 'format', 'files_checked':len(files),
                      'formatting_changes':sum(pending.values()), 'files':pending},indent=2))
    return bool(pending) if check else False


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    raise SystemExit(main(parser.parse_args().check))
