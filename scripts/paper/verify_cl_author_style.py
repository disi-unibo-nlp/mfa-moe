"""Check the author's explicit abstract and heading exclusions before delivery.

These lexical checks supplement native paper-writer semantic/page review. They
do not judge scientific claims, figure legibility or reference-style cadence.
"""
from pathlib import Path
import json
import re

PAPER = Path(__file__).resolve().parents[2] / "paper/6a510c6db0d9b08aa438c142"
MODEL_NAMES = r"\b(?:GPT|Qwen\d*(?:\.\d+)?|Gemma|GLM|Nemotron|LFM[\d.]*)\b"


def inspect_abstract(text):
    errors = []
    checks = {
        "colon": r":",
        "indirect author phrasing": r"\bour approach\b",
        "stock wrap-up": r"\btogether[,]?\s+these findings\b",
        "model name": MODEL_NAMES,
        "domain-limited reasoning framing": r"\bmathematical\s+reasoning\b",
    }
    for label, pattern in checks.items():
        if re.search(pattern, text, re.I):
            errors.append("abstract: " + label)
    if not re.search(r"large\s+language\s+models\s*\(LLMs\)", text, re.I):
        errors.append("abstract: missing large language models (LLMs) introduction")
    return errors


def inspect_headings(text):
    errors = []
    for title in re.findall(r"\\(?:subsection|paragraph)\*?\{([^{}]+)\}", text):
        if re.search(MODEL_NAMES, title, re.I):
            errors.append("Results heading contains a model name: " + title)
        if re.search(r"\b(?:\d+|one|two|three|four|five|six|seven|eight|every|all)\s+(?:\w+\s+)?models?\b", title, re.I):
            errors.append("Results heading uses model-count framing: " + title)
        if re.search(r"\b(?:completed|provisional|pending|fixtures?|implementation details|software pipeline)\b", title, re.I):
            errors.append("Results heading uses implementation/status framing: " + title)
    return errors


def main():
    source = (PAPER / "main.tex").read_text()
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", source, re.S)
    if not abstract:
        raise ValueError("abstract not found")
    errors = inspect_abstract(abstract.group(1))
    errors += inspect_headings((PAPER / "journal/results.tex").read_text())
    control = "Large language models (LLMs) reason."
    assert not inspect_abstract(control)
    for forbidden in [":", "Our approach", "Together these findings", "Qwen", "Qwen3.6", "LFM2.5", "mathematical reasoning"]:
        assert inspect_abstract(control + " " + forbidden), forbidden
    for title in ["Gain in every model", "Gains in five models", "Two completed analyses", "Qwen gains"]:
        assert inspect_headings("\\subsection{" + title + "}"), title
    assert not inspect_headings("\\subsection{Reasoning concentrates on analysis and implementation}")
    result = {"status": "fail" if errors else "pass", "violations": errors,
              "negative_controls": 11,
              "semantic_and_visual_review_still_required": True}
    print(json.dumps(result, indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
