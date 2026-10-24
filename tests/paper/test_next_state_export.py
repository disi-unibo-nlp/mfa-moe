"""The next-state and transition-summary exports: sign convention, labels, flags, and the checks that read them back."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import store
from paper import export_cl_exhibits as export
from paper import verify_cl_tables as verify
from paper.cl_models import MODELS


def receipt(mean, low, high, simultaneous, text=1.5):
    """A next-state receipt of one model: routing minus text is `mean`, so a negative value favours routing."""
    gain = {"P": mean, "ci95": [low, high], "simultaneous_ci7": list(simultaneous), "p": 0.01}
    return {"status": "CLEAN", "definition": export.NEXT_STATE_DEFINITION, "pairs": 1234, "labelled_sentences": 5678,
            "source_tokens": 90123, "cross_entropy": {"markov": 1.6, "text": text, "routing": text + mean},
            "readout_analysis": {"status": "CLEAN", "n_questions": 500, "n_families": 480},
            "routing_gain": gain, "routing_excess_over_null": dict(gain)}


def receipts():
    """Eight models; the last one's interval contains zero."""
    out = {}
    for i, model in enumerate(MODELS):
        mean = -0.02 * (i + 1)
        interval = (mean - 0.01, mean + 0.01)
        out[model] = receipt(mean, *interval, (mean - 0.02, mean + 0.02 if i < 7 else 0.1))
    return out


class NextStateRow(unittest.TestCase):
    def test_a_row_keeps_the_sign_of_the_receipt_and_orders_the_limits(self):
        row = export.next_state_row("gpt", receipt(-0.05, -0.06, -0.04, (-0.07, -0.03)))
        self.assertEqual((row["delta_logloss"], row["ci95_low"], row["simultaneous_ci7_high"]), (-0.05, -0.06, -0.03))
        self.assertEqual((row["questions"], row["families"], row["pairs"]), (500, 480, 1234))

    def test_a_gain_that_is_not_routing_minus_text_is_refused(self):
        wrong = receipt(-0.05, -0.06, -0.04, (-0.07, -0.03))
        wrong["cross_entropy"]["routing"] = 1.55
        with self.assertRaises(ValueError):
            export.next_state_row("gpt", wrong)

    def test_a_simultaneous_interval_inside_the_plain_one_is_refused(self):
        with self.assertRaises(ValueError):
            export.next_state_row("gpt", receipt(-0.05, -0.06, -0.04, (-0.055, -0.045)))

    def test_another_definition_of_the_gain_is_refused(self):
        other = receipt(-0.05, -0.06, -0.04, (-0.07, -0.03))
        other["definition"] = "gain = text minus routing"
        with self.assertRaises(ValueError):
            export.next_state_row("gpt", other)


class ExportedFiles(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(dir=store.WORK / "tools/tmp")
        self.data = Path(self.folder.name) / "figures/journal/data"
        self.data.mkdir(parents=True)
        patcher = patch.object(export, "DATA", self.data)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.folder.cleanup)
        rows = [export.next_state_row(model, receipts()[model]) for model in MODELS]
        with (self.data / "next_state.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def table(self, name):
        lines = (self.data / name).read_text().splitlines()
        return lines[0].split(), [line.split() for line in lines[1:]]

    def test_the_plot_data_has_one_row_per_model_with_a_label_above_each_whisker(self):
        export.next_state_data()
        heading, body = self.table("next_state.dat")
        self.assertEqual(len(body), len(MODELS))
        for row in body:
            values = dict(zip(heading, row))
            gain, margin = -float(values["gain"]), max(float(values["gainminus"]), float(values["gainplus"]))
            self.assertEqual(values["gainlabel"], f"{gain:.3f}")
            self.assertAlmostEqual(float(values["gainlabely"]), gain + margin + export.NEXT_STATE_LABEL_GAP)

    def test_a_missing_interval_gives_a_bar_without_whisker_and_a_dagger(self):
        results = {}
        for model in MODELS:
            cell = {"estimate": 0.25, "interval": [0.2, 0.3]}
            results[model] = {"results": {key: cell for _, key, _, _ in export.SUMMARY_PANELS}}
        results[MODELS[1]]["results"]["Verify:next:Verify"] = {"estimate": 0.3, "interval": None}
        export.summary_data(results)
        heading, body = self.table("state_transitions_summary.dat")
        flags = [dict(zip(heading, row))["vvmissing"] for row in body]
        self.assertEqual(flags, ["0", "1"] + ["0"] * (len(MODELS) - 2))
        missing = dict(zip(heading, body[1]))
        self.assertEqual(missing["vvlabel"], r"30.0$^{\dagger}$")
        self.assertEqual(missing["vvminus"], "nan")
        self.assertAlmostEqual(float(missing["vvlabely"]), 30 + export.SUMMARY_LABEL_GAP * 52)


class CompletedPanels(unittest.TestCase):
    VARIANT = "read-verify"

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(dir=store.WORK / "tools/tmp")
        self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name)
        self.study, self.data = root / "study", root / "data"
        self.data.mkdir()
        (self.data / "native_dynamics.json").write_text(json.dumps({m: {"old": True} for m in MODELS}))
        (self.data / "SOURCES.json").write_text(json.dumps({"sources": {}}))
        self.panels = {}
        for model in MODELS:
            panel = store.seal({"panel": "native-dynamics:" + model, "results": {"label_coverage": {"family_support": 1486}},
                                "pair_completion": {"labelled_successors": 3}})
            folder = self.study / "transition-pairs" / model / self.VARIANT
            folder.mkdir(parents=True)
            (folder / "NATIVE_DYNAMICS_SUMMARY.json").write_text(json.dumps(panel))
            self.panels[model] = panel
        for name, value in (("STUDY", self.study), ("DATA", self.data)):
            patcher = patch.object(export, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def join(self, panels):
        (self.study / "driver").mkdir(exist_ok=True)
        joined = store.seal({"models": {m: {"panel_sha256": p["sha256"]} for m, p in panels.items()}})
        (self.study / f"driver/TRANSITION_PAIRS.{self.VARIANT}.json").write_text(json.dumps(joined))

    def test_every_panel_is_replaced_by_the_completed_one(self):
        self.join(self.panels)
        export.capture_pairs(self.VARIANT)
        saved = json.loads((self.data / "native_dynamics.json").read_text())
        self.assertTrue(all(saved[m]["pair_completion"] == {"labelled_successors": 3} for m in MODELS))
        sources = json.loads((self.data / "SOURCES.json").read_text())["sources"]
        self.assertIn("transition_pairs", sources)
        self.assertTrue(all(sources[m + ":dynamics"]["path"].endswith(self.VARIANT + "/NATIVE_DYNAMICS_SUMMARY.json") for m in MODELS))

    def test_a_panel_that_the_joined_receipt_does_not_name_is_refused(self):
        self.join({**self.panels, MODELS[0]: {"sha256": "0" * 64}})
        with self.assertRaises(ValueError):
            export.capture_pairs(self.VARIANT)


class OccupancyLabels(unittest.TestCase):
    ORDER = ("Read", "Analyze", "Explore", "Plan", "Implement", "Verify", "Monitor")

    def test_a_label_clears_the_label_of_a_neighbouring_column_only(self):
        tops = {state: (50., column, 40.) for column, state in enumerate(self.ORDER, 1)}
        tops["Plan"], tops["Explore"], tops["Read"] = (6., 4, 5.), (8., 3, 7.), (6.5, 1, 5.5)
        ends = dict(zip(self.ORDER, export.occupancy_labels(tops, self.ORDER)))
        self.assertAlmostEqual(ends["Plan"][0], 6 + export.OCCUPANCY_LEAD)
        self.assertAlmostEqual(ends["Explore"][0], ends["Plan"][0] + export.OCCUPANCY_CLEAR)
        self.assertAlmostEqual(ends["Read"][0], 6.5 + export.OCCUPANCY_LEAD)
        self.assertEqual(ends["Explore"][1], "7.0")


class VerifierReadsTheTable(unittest.TestCase):
    NAMES = {model: label for label, model in verify.LABELS.items()}

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(dir=store.WORK / "tools/tmp")
        self.addCleanup(self.folder.cleanup)
        self.paper = Path(self.folder.name)
        self.data = self.paper / "figures/journal/data"
        self.data.mkdir(parents=True)
        (self.paper / "journal/tables").mkdir(parents=True)
        with patch.object(export, "DATA", self.data):
            rows = [export.next_state_row(model, receipts()[model]) for model in MODELS]
            with (self.data / "next_state.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            export.next_state_data()
        self.rows = {row["model"]: row for row in rows}

    def table_text(self):
        gains = [verify.next_state_gains(self.rows, prefix) for prefix, _ in verify.NEXT_STATE_COLUMNS]
        lines = []
        for model in MODELS:
            cells = []
            for values in gains:
                mark = verify.rank_mark(values[model][0], [v[0] for v in values.values()])
                mean, minus, plus = verify.signed_deviations(values[model], 5)
                cells.append("\\%sd{%s}{%s}{%s}" % (mark, mean, minus, plus))
            row = [r"\modelname{x}{%s}" % self.NAMES[model], verify.thousands(self.rows[model]["questions"]),
                   verify.thousands(self.rows[model]["pairs"]), *cells]
            lines.append(" & ".join(row) + r" \\")
        return "\n".join(lines)

    def test_a_table_written_from_the_export_passes_and_a_changed_cell_does_not(self):
        text = self.table_text()
        self.assertEqual(verify.verify_next_state(self.paper, {"next_state.tex": text})[0][1], 4 * len(MODELS))
        changed = text.replace("cid{0.02", "cid{0.03", 1)
        self.assertNotEqual(changed, text)
        with self.assertRaises(verify.TableMismatch):
            verify.verify_next_state(self.paper, {"next_state.tex": changed})

    def test_a_swapped_rank_mark_is_refused(self):
        text = self.table_text()
        with self.assertRaises(verify.TableMismatch):
            verify.verify_next_state(self.paper, {"next_state.tex": text.replace(r"\bestcid", r"\cid", 1)})


if __name__ == "__main__":
    unittest.main()
