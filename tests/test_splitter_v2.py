"""Version 2 splitter: targeted fix of the stray single-`$` merge, v1 kept for mapping."""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from moe_exp.correlation_pipeline.spans import (  # noqa: E402
    math_protection_spans,
    map_selection_to_v2,
    sentence_spans,
    sentence_spans_v1,
    splitter_delta,
)
from moe_exp.schemas import TraceRecord  # noqa: E402

LEGACY_PATTERN = re.compile(
    r"\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)"
    r"|(?<!\\)\$(?:(?:\\.|[^$\\])*?\$|(?:\\.|[^$\\])*\\\$)",
    flags=re.DOTALL,
)
LEGACY_BOUNDARY = re.compile(r"(?<=[.!?])[^\S\n]+|\n+")
OPENING_CONTEXT = "([{=,;:"


def legacy_reference_protection(text: str) -> list[tuple[int, int]]:
    """The verified prototype: frozen v1 matches minus the suspicious single-`$` spans."""
    spans = []
    for match in LEGACY_PATTERN.finditer(text):
        span_text = match.group(0)
        if span_text.startswith("$") and not span_text.startswith("$$"):
            previous = text[match.start() - 1] if match.start() > 0 else ""
            inner = span_text[1:-1]
            if (
                previous
                and not previous.isspace()
                and previous not in OPENING_CONTEXT
                and (len(inner) > 1000 or len(LEGACY_BOUNDARY.findall(inner)) >= 2)
            ):
                continue
        spans.append(match.span())
    return spans


def trace(text: str, *, dataset: str = "math500", problem_id: str = "p0", sample_id: int = 0, metadata=None):
    return TraceRecord(
        dataset=dataset,
        problem_id=problem_id,
        source_problem_id=problem_id,
        sample_id=sample_id,
        prompt="Question?",
        gold_answer="1",
        model_id="test",
        model_answer="1",
        is_correct=True,
        cot_text=text,
        metadata=metadata or {},
    )


def unit_texts(text: str) -> list[str]:
    return [unit["text"] for unit in sentence_spans(trace(text))]


class ScannerEquivalenceTests(unittest.TestCase):
    def test_scanner_matches_the_verified_prototype_on_edge_cases(self):
        samples = [
            "",
            "Plain prose. Another sentence.",
            "The value $1. 5$ equals. Next sentence.",
            r"Cost $a. b\$. After.",
            "Value is 5$ First sentence here. Second sentence here. And $x^{2}$ is the formula.",
            "Money $5 and $10 in prose. Another sentence follows.",
            "$" + r"\alpha " * 200 + "\nNext.",
            "Result 5$" + "Sentence number here. " * 400 + "final$ tail.",
            "Display $$x. y$$ stays. Next.",
            r"Block \[a. b\] stays. Next.",
            r"Inline \(c. d\) stays. Next.",
            "Unclosed $$ display start. Next sentence.",
            "Trailing opener $",
        ]
        for text in samples:
            with self.subTest(text=text[:40]):
                self.assertEqual(sorted(math_protection_spans(text)), sorted(legacy_reference_protection(text)))


class StrayDelimiterTests(unittest.TestCase):
    def test_stray_closing_dollar_no_longer_merges_prose(self):
        text = "Value is 5$ First sentence here. Second sentence here. And $x^{2}$ is the formula."
        self.assertEqual(
            unit_texts(text),
            [
                "Value is 5$ First sentence here.",
                "Second sentence here.",
                "And $x^{2}$ is the formula.",
            ],
        )
        # The frozen splitter hid both inner sentence boundaries, which is exactly the bug.
        self.assertEqual(len(sentence_spans_v1(trace(text))), 1)

    def test_real_inline_math_stays_protected(self):
        self.assertEqual(
            unit_texts("The value $1. 5$ equals. Next sentence."),
            ["The value $1. 5$ equals.", "Next sentence."],
        )

    def test_escaped_dollar_still_closes_a_span(self):
        self.assertEqual(
            unit_texts(r"Cost $a. b\$. After."),
            [r"Cost $a. b\$.", "After."],
        )

    def test_unclosed_latex_does_not_stall_sentence_splitting(self):
        text = "$" + r"\alpha " * 2000 + "\nNext."
        self.assertEqual(unit_texts(text), [text.split("\n")[0].strip(), "Next."])

    def test_oversized_single_dollar_span_splits(self):
        inner = "Sentence number here. " * 900
        text = "Result 5$" + inner + "final$ tail."
        legacy = sentence_spans_v1(trace(text))
        current = sentence_spans(trace(text))
        self.assertEqual(len(legacy), 1)
        self.assertGreater(len(current), 100)
        self.assertEqual(current[0]["text"], "Result 5$Sentence number here.")
        self.assertTrue(current[-1]["text"].endswith("tail."))


class MappingTests(unittest.TestCase):
    def test_map_selection_to_v2_expands_merged_units(self):
        text = "Intro. Value is 5$ First sentence here. Second sentence here. And $x^{2}$ is the formula. Done."
        item = trace(text)
        legacy = sentence_spans_v1(item)
        current = sentence_spans(item)
        merged = [index for index, unit in enumerate(legacy) if unit["text"].startswith("Value is 5$")]
        self.assertEqual(len(merged), 1)
        mapped = map_selection_to_v2(item, merged)
        self.assertEqual([unit["text"] for unit in legacy], ["Intro.", "Value is 5$ First sentence here. Second sentence here. And $x^{2}$ is the formula.", "Done."])
        self.assertEqual(mapped, [1, 2, 3])
        self.assertEqual(
            [current[index]["text"] for index in mapped],
            [
                "Value is 5$ First sentence here.",
                "Second sentence here.",
                "And $x^{2}$ is the formula.",
            ],
        )
        # Every v1 index that is not inside the merged unit maps onto itself.
        self.assertEqual(map_selection_to_v2(item, [0]), [0])
        self.assertEqual(map_selection_to_v2(item, [len(legacy) - 1]), [len(current) - 1])

    def test_map_selection_rejects_out_of_range_indices(self):
        with self.assertRaises(ValueError):
            map_selection_to_v2(trace("One. Two."), [5])

    def test_splitter_delta_reports_superseded_and_added_units(self):
        text = "Value is 5$ First sentence here. Second sentence here. And $x^{2}$ is the formula."
        delta = splitter_delta(trace(text))
        self.assertTrue(delta["affected"])
        self.assertEqual(delta["v1_units"], 1)
        self.assertEqual(delta["v2_units"], 3)
        self.assertEqual(delta["superseded_units"], 1)
        self.assertEqual(delta["added_units"], 3)
        self.assertEqual(delta["mapped_units"], 3)

    def test_plain_trace_is_unchanged(self):
        delta = splitter_delta(trace("First sentence. Second sentence. Third sentence."))
        self.assertFalse(delta["affected"])
        self.assertEqual(delta["v1_units"], delta["v2_units"])
        self.assertEqual(delta["superseded_units"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
