"""Imported v2 annotations select their splitter without changing legacy traces."""
import unittest
from types import SimpleNamespace

from moe_exp.correlation_pipeline.spans import (
    sentence_spans,
    trace_digest,
    validate_available_annotation,
    validate_annotation,
)


class Trace(SimpleNamespace):
    def model_copy(self, update):
        return Trace(**{**vars(self), **update})


class ImportedLabelSplitterTests(unittest.TestCase):
    def setUp(self):
        self.trace = Trace(
            dataset="example", problem_id="one", prompt="Solve.",
            generation_messages=None, system_prompt=None,
            cot_text="Price$ closes.\nCheck this.\nThen $x$ ends.", metadata={},
        )

    def annotation(self):
        trace = self.trace.model_copy(update={
            "metadata": {"reasoning_annotation": {"splitter_version": 2}},
        })
        units = sentence_spans(trace)
        return {
            "schema_version": 1, "splitter_version": 2,
            "trace_sha256": trace_digest(trace), "dataset": trace.dataset,
            "problem_id": trace.problem_id, "status": "complete",
            "sentence_selection": {"schema_version": 1, "indices": [u["index"] for u in units]},
            "units": [{**u, "label": "Implement"} for u in units],
        }

    def test_versioned_splitter_preserves_legacy(self):
        legacy = sentence_spans(self.trace)
        annotation = self.annotation()
        self.assertGreater(len(annotation["units"]), len(legacy))
        validate_available_annotation(self.trace, annotation)
        self.assertEqual(sentence_spans(self.trace), legacy)
        tagged = self.trace.model_copy(update={"metadata": {"reasoning_annotation": annotation}})
        self.assertEqual(sentence_spans(tagged), [
            {k: v for k, v in unit.items() if k != "label"} for unit in annotation["units"]
        ])

    def test_strict_validation_uses_imported_version(self):
        annotation = self.annotation()
        selected = self.trace.model_copy(update={"metadata": {
            "sentence_selection": annotation["sentence_selection"],
        }})
        validate_annotation(selected, annotation)

    def test_incorrect_offsets_rejected(self):
        annotation = self.annotation()
        annotation["units"][0]["end"] += 1
        with self.assertRaisesRegex(ValueError, "offsets/text"):
            validate_available_annotation(self.trace, annotation)

    def test_stale_trace_rejected(self):
        annotation = self.annotation()
        annotation["trace_sha256"] = "stale"
        with self.assertRaisesRegex(ValueError, "Stale"):
            validate_available_annotation(self.trace, annotation)

    def test_unsupported_version_rejected(self):
        annotation = self.annotation()
        annotation["splitter_version"] = 99
        with self.assertRaisesRegex(ValueError, "splitter version"):
            validate_available_annotation(self.trace, annotation)
