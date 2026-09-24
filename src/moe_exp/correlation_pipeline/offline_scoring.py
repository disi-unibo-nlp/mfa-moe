"""Versioned answer-equivalence audit, independent of historical generation scoring.

Use report/rescoring_requirements.txt. Only the extracted final candidate is
compared, never an arbitrary earlier expression from a reasoning trace.
"""
from functools import lru_cache
import re
from math_verify import LatexExtractionConfig, parse, verify
from math_verify.errors import TimeoutException
from sympy import Basic, Float, Rational, N, simplify
from moe_exp.utils import extract_model_answer

CONTRACT = 'extracted_answer_equivalence_v1'
_TEXT = re.compile(r'\\(?:text|textrm|mathrm|mbox|operatorname)\s*\{([^{}]*)\}')
_SCI = re.compile(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))[eE]([+-]?\d+)')
# Text in a mathematical expression is accepted only for explicit units.
_UNITS = {'m', 'cm', 'mm', 'km', 's', 'ms', 'kg', 'g', 'n', 'j', 'w', 'v', 'a',
          'k', 'pa', 'hz', 'c', 't', 'f', 'ohm', 'rad', 'radians', 'degrees', 'degree',
          'ev', 'mev', 'gev', 'mol', 'l', 'newtons', 'watts', 'joules', 'seconds',
          'minutes', 'minute', 'hours', 'hour', 'inches', 'inch', 'feet', 'foot',
          'dollars', 'dollar', 'cents', 'cent', 'per', 'can', 'second',
          'nm', 'erg', 'arcseconds', 'arcsecond'}


def clean_annotations(value):
    value = re.sub(r'\\text\s*\{\s*(?:and|or)\s*\}', ',', value)
    value = re.sub(r'\\text\s*\{\s*\(?approximately\)?\s*\}', '', value)
    return value


def normalized_display(value):
    value = clean_annotations(value).strip().rstrip('.').replace(r'\$', '')
    value = value.replace('$', '').replace(r'\(', '').replace(r'\)', '')
    value = value.replace(r'\[', '').replace(r'\]', '')
    value = value.replace(r'\left', '').replace(r'\right', '')
    value = value.replace(r'\dfrac', r'\frac').replace(r'\tfrac', r'\frac')
    value = value.replace('−', '-')
    value = re.sub(r'\\[,!;: ]', '', value)
    # Preserve mathematical symbol case; case folding is only for plain text.
    text = _TEXT.fullmatch(value)
    word = text.group(1).strip() if text else value
    if re.fullmatch(r'[A-Za-z][A-Za-z -]+', word) and not (word.isupper() and len(word) <= 3):
        return 'text:' + ' '.join(word.lower().split())
    return re.sub(r'\s+', '', value)


@lru_cache(maxsize=50000)
def parsed(value):
    # E notation is not LaTeX: without this, 1.2e-8 parses as 1.2*e - 8.
    value = clean_annotations(value).strip().rstrip('.').replace(r'\$', '').replace('$', '')
    if re.fullmatch(r'[+-]?\d+', value) and len(value) < 1000:
        return Rational(value)
    if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', value):
        return Rational(str(Float(value, 50)))
    assignment = re.fullmatch(r'[A-Za-z](?:_\{[^{}]*\}|_\w+)?(?:\([^=]+\))?\s*=\s*([^=]+)', value)
    if assignment:
        value = assignment[1].strip()
    if len(value) > 2048:
        return None
    if _SCI.fullmatch(value):
        value = _SCI.sub(lambda m: m[1]+r'\times10^{'+m[2]+'}', value)
    result = parse('$'+value+'$', extraction_config=[LatexExtractionConfig()],
                   fallback_mode='no_fallback', extraction_mode='first_match',
                   parsing_timeout=2, raise_on_error=True)
    # Never retain raw-string parser fallbacks as mathematical equivalence.
    if len(result) != 1 or isinstance(result[0], str):
        return None
    obj = result[0]
    # Exact decimal representations prevent absolute-rounding false positives
    # for answers such as 1e-27 versus 0 inside structured expressions.
    if isinstance(obj, Basic):
        obj = obj.xreplace({x: Rational(str(x)) for x in obj.atoms(Float)})
    return obj


@lru_cache(maxsize=50000)
def equivalent(prediction, gold):
    if not gold.strip():
        raise ValueError('Missing gold answer')
    if not prediction.strip():
        return False, 'empty_prediction'
    prediction, gold = clean_annotations(prediction), clean_annotations(gold)
    a, b = normalized_display(prediction), normalized_display(gold)
    if a == b:
        return True, 'normalized_display'
    if a.startswith('text:') or b.startswith('text:'):
        return False, 'different_text'
    # Do not allow a symbolic parser to discard prose describing multiple answers.
    for value in (prediction, gold):
        for match in _TEXT.finditer(value):
            if value[:match.start()].rstrip().endswith(('_', '_{')):
                continue
            block = match[1]
            words = re.findall(r'[A-Za-z]+', block.lower())
            if any(w not in _UNITS for w in words):
                return False, 'unsupported_embedded_text'
    try:
        p, g = parsed(prediction), parsed(gold)
        if p is None or g is None:
            return False, 'unparsed_expression'
        if getattr(p, 'is_number', False) and getattr(g, 'is_number', False):
            if p == g or simplify(p-g) == 0:
                return True, 'exact_number'
            if p.is_finite is True and g.is_finite is True:
                delta = abs(N(p-g, 50)); scale = max(abs(N(p, 50)), abs(N(g, 50)))
                if delta == 0:
                    return True, 'exact_number'
                approximate = any(re.search(r'\d\.\d|\d[eE][+-]?\d', x) for x in (prediction, gold))
                if not approximate:
                    return False, 'different_exact_numbers'
                # Relative tolerance only for decimal/scientific answers.
                return bool(scale != 0 and delta <= Rational(1, 10**6)*scale), 'relative_numeric_1e-6'
        return bool(verify(g, p, strict=True, float_rounding=50, numeric_precision=50,
                           timeout_seconds=2, raise_on_error=True)), 'symbolic_strict'
    except (Exception, TimeoutException) as error:
        return False, 'parse_or_verify_error:' + type(error).__name__


def rescore(text, gold, answer_type='math'):
    if answer_type == 'choice':
        from moe_exp.correlation_pipeline.scoring import extract_choice
        answer = extract_choice(text)
        return answer, bool(answer and answer == gold.strip().upper()), 'exact_choice'
    answer = extract_model_answer(text)
    correct, method = equivalent(answer, gold)
    return answer, correct, method
