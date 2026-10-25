"""The prefix-probe rows: copied from the saved summary, written into the appendix table, and checked against the summary."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import store
from paper import export_cl_exhibits as export
from paper import verify_cl_tables as verify

TABLE = r"""\label{tab:prefix-probes}
\begin{tabular}{@{}lrrr@{}}
\toprule
Encoder & Probe AUROC & Position AUROC & Paired gain \\
\midrule
\modelname{openai}{GPT-OSS-20B} & old \\
\modelname{gemma}{Gemma-4-26B-A4B} & old \\
\modelname{qwen}{Qwen3.5-35B-A3B} & old \\
\bottomrule
\end{tabular}
"""


def summary():
    """A grouped-probe summary: probe AUROC rises with the encoder index, interval limits are asymmetric."""
    models = {}
    for i, name in enumerate(("GPT-OSS", "Gemma", "Qwen")):
        def auc(estimate):
            return {"estimate": estimate, "ci95": [estimate - 0.0141, estimate + 0.0286]}
        models[name] = {"macro": {"probe": {"auc": auc(0.70 + 0.05 * i), "f1": {"estimate": 0.40 + 0.01 * i}},
                                  "position": {"auc": auc(0.57)}, "probe_minus_position": {"auc": auc(0.13 + 0.05 * i)}}}
    return {"models": models}


class ProbeRows(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(dir=store.WORK / "tools/tmp")
        self.addCleanup(self.folder.cleanup)
        self.paper = Path(self.folder.name)
        self.data = self.paper / "figures/journal/data"
        self.data.mkdir(parents=True)
        (self.paper / "journal").mkdir()
        self.source = self.paper / "grouped_probe_sources.json"
        self.source.write_text(json.dumps(summary()))
        (self.data / "SOURCES.json").write_text(json.dumps({"sources": {}}))
        sentence = "At the selected decision rule, macro binary F1 is 0.420 for Qwen3.5, 0.400 for GPT-OSS, and 0.410 for Gemma."
        (self.paper / "journal/appendices.tex").write_text(TABLE + sentence + "\n")
        for target, value in (("PAPER", self.paper), ("DATA", self.data), ("PROBE_SOURCE", self.source), ("REPO", self.paper)):
            patcher = patch.object(export, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(verify, "PROBE_SOURCE", self.source)
        patcher.start()
        self.addCleanup(patcher.stop)
        export.capture_probes()
        export.probe_table()

    def appendix(self):
        return (self.paper / "journal/appendices.tex").read_text()

    def test_the_capture_keeps_the_nine_estimates_with_both_limits(self):
        rows = (self.data / "probes.csv").read_text().splitlines()
        self.assertEqual(len(rows), 1 + 9)
        self.assertIn("qwen35,paired_gain,0.23,", rows[-1])

    def test_a_row_prints_both_deviations_and_the_best_and_second_best_marks(self):
        text = self.appendix()
        self.assertIn(r"\bestcid{0.800}{0.014}{0.029}", text)
        self.assertIn(r"\secondcid{0.750}{0.014}{0.029}", text)
        self.assertEqual(text.count(r"\cid{0.570}{0.014}{0.029}"), 3)  # position AUROC is not ranked

    def test_the_table_the_exporter_writes_passes_the_check_and_a_changed_cell_does_not(self):
        self.assertEqual(verify.verify_probes(self.paper)[0][1], 9)
        changed = self.appendix().replace(r"\bestcid{0.800}", r"\bestcid{0.900}")
        with self.assertRaises(verify.TableMismatch):
            verify.verify_probes(self.paper, changed)

    def test_a_deviation_taken_from_rounded_endpoints_is_refused(self):
        changed = self.appendix().replace(r"\bestcid{0.800}{0.014}{0.029}", r"\bestcid{0.800}{0.014}{0.030}")
        with self.assertRaises(verify.TableMismatch):
            verify.verify_probes(self.paper, changed)

    def test_a_probes_copy_that_differs_from_the_saved_summary_is_refused(self):
        path = self.data / "probes.csv"
        path.write_text(path.read_text().replace("0.7,", "0.71,", 1))
        with self.assertRaises(verify.TableMismatch):
            verify.verify_probes(self.paper)

    def test_a_second_run_leaves_the_appendix_as_it_is(self):
        first = self.appendix()
        export.probe_table()
        self.assertEqual(first, self.appendix())


if __name__ == "__main__":
    unittest.main()
