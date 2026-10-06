"""Conservative prefix-only candidate adapter. No gold, future text, or final fields enter it.

This parser is a candidate detector, not a correctness judge or semantic class labeler.
Only closed boxes, closed math delimiters, or a full 12-character lookahead are accepted.
The supported expression grammar is exact numeric arithmetic; symbolic answers abstain.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction
import re
from typing import Mapping

VERSION = 'prefix-candidate-v1'
ALLOWLIST = frozenset({'problem', 'emitted_token_ids', 'emitted_text'})
LOOKAHEAD = 12
SCAN_LIMIT = 2000


def numeric_expression(text: str) -> str | None:
    """Return an exact rational value within a bounded, explicitly supported grammar."""
    text = text.strip().replace('−', '-').replace('×', '*').replace('÷', '/')
    for _ in range(4):
        updated = re.sub(r'\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', text)
        if updated == text:
            break
        text = updated
    text = text.replace(r'\cdot', '*').replace(r'\times', '*').replace('^', '**')
    if len(text) > 160 or not re.fullmatch(r'[\d\s.+*/()\-]+', text):
        return None
    try:
        tree = ast.parse(text, mode='eval')
        if len(list(ast.walk(tree))) > 64:
            return None

        def evaluate(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                return Fraction(ast.get_source_segment(text, node))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                return evaluate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            if not isinstance(node, ast.BinOp):
                raise ValueError('unsupported expression')
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            elif isinstance(node.op, ast.Div):
                value = left / right
            elif isinstance(node.op, ast.Pow) and right.denominator == 1 and abs(right) <= 8:
                value = left ** int(right)
            else:
                raise ValueError('unsupported operator')
            if max(value.numerator.bit_length(), value.denominator.bit_length()) > 256:
                raise ValueError('expression exceeds numeric bound')
            return value

        value = evaluate(tree.body)
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > 256:
            return None
        return str(value)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, RecursionError):
        return None


@dataclass(frozen=True)
class Candidate:
    start: int
    end: int  # includes all evidence needed to recognize completeness
    expression: str
    value: str
    kind: str


def candidates(text: str) -> list[Candidate]:
    """All complete supported candidates, with recognition offsets inside this prefix."""
    reasoning = text.split('</think>', 1)[0]
    found, boxed_spans = [], []
    for match in re.finditer(r'\\boxed\s*\{', reasoning):
        left, depth = match.end(), 1
        for index in range(left, min(len(reasoning), left + SCAN_LIMIT)):
            depth += (reasoning[index] == '{') - (reasoning[index] == '}')
            if depth == 0:
                boxed_spans.append((match.start(), index + 1))
                expr = reasoning[left:index]
                value = numeric_expression(expr)
                if value is not None:
                    found.append(Candidate(match.start(), index + 1, expr, value, 'boxed'))
                break
    for match in re.finditer(r'\banswer\s+is\s*:?[ \t]*', reasoning, re.I):
        start = match.end()
        if any(a <= start < b for a, b in boxed_spans) or reasoning[start:].startswith(r'\boxed'):
            continue
        tail = reasoning[start:]
        delimiters = (('$$', '$$'), ('$', '$'), (r'\(', r'\)'), (r'\[', r'\]'))
        wrapped = False
        for opening, closing in delimiters:
            if not tail.startswith(opening):
                continue
            wrapped = True
            stop = tail.find(closing, len(opening))
            if stop >= 0:
                expr = tail[len(opening):stop]
                value = numeric_expression(expr)
                if value is not None:
                    found.append(Candidate(match.start(), start + stop + len(closing),
                                           expr, value, 'answer_math'))
            break
        if wrapped:
            continue
        number = re.match(r'[-+]?\d+(?:\.\d+)?', tail)
        if not number:
            continue
        end = start + number.end()
        # Never accept a partial lookahead as the old parser did.
        window = reasoning[end:end + LOOKAHEAD]
        if len(window) != LOOKAHEAD:
            continue
        rest = window.lstrip(' ')
        if not rest or rest[0] not in '.!;\n':
            continue
        if rest[0] == '.' and (len(rest) < 2 or rest[1].isdigit()):
            continue
        value = numeric_expression(number.group())
        if value is not None:
            found.append(Candidate(match.start(), end + LOOKAHEAD, number.group(), value, 'answer_is'))
    return sorted(found, key=lambda row: (row.end, row.start))


@dataclass(frozen=True)
class PrefixInput:
    problem: str
    emitted_token_ids: tuple[int, ...]
    emitted_text: str

    @classmethod
    def from_record(cls, record: Mapping) -> 'PrefixInput':
        # Copy only these keys. Forbidden fields are never inspected, even for defaults.
        problem, ids, text = (record[key] for key in ('problem', 'emitted_token_ids', 'emitted_text'))
        if not isinstance(problem, str) or not isinstance(text, str):
            raise ValueError('problem and emitted_text must be strings')
        if not isinstance(ids, (list, tuple)) or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError('emitted_token_ids must contain nonnegative integers')
        return cls(problem, tuple(ids), text)


class StreamingAdapter:
    """Append-only input validation; decisions are reproducible from the allowed prefix."""
    def __init__(self):
        self.previous: PrefixInput | None = None
        self.last_candidate_end = -1

    def observe_tokens(self, problem: str, token_ids, decode) -> dict:
        """Production adapter: derive prefix text exclusively from already emitted tokens."""
        ids = tuple(token_ids)
        return self.observe({'problem': problem, 'emitted_token_ids': ids,
                             'emitted_text': decode(ids)})

    def observe(self, record: Mapping) -> dict:
        prefix = PrefixInput.from_record(record)
        if self.previous is not None:
            old = self.previous
            if prefix.problem != old.problem or prefix.emitted_token_ids[:len(old.emitted_token_ids)] != old.emitted_token_ids:
                raise ValueError('stream changed its problem or emitted token history')
            if not prefix.emitted_text.startswith(old.emitted_text):
                raise ValueError('decoded stream is not append-only; replay with a fresh adapter')
        self.previous = prefix
        closed = '</think>' in prefix.emitted_text
        parsed = candidates(prefix.emitted_text)
        latest = parsed[-1] if parsed else None
        new_candidate = latest is not None and latest.end > self.last_candidate_end
        if latest is not None:
            self.last_candidate_end = max(self.last_candidate_end, latest.end)
        return {'version': VERSION, 'closure': closed, 'candidate': latest,
                'can_trigger': new_candidate and not closed,
                'coverage': 'supported_candidate' if parsed else 'abstain'}
