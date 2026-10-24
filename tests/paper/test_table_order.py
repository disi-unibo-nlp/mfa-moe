"""Check the model display order and that the table source comparison accepts a row reorder but not changed numbers.

Run with PYTHONPATH=src python -m unittest discover -s tests/paper -p test_table_order.py.
Needs no manuscript, receipt or PDF.
"""
import copy
import unittest

from paper.cl_models import MODELS, ORIGINAL_MODELS, PLOT_LABELS, SOURCE_MODELS, TICKS
from paper.verify_cl_printed_tables import compare_source_tables, parse_table
from store import MODELS as STORE_MODELS

TABLE = r"""\begin{table}
\caption{Accuracy.}\label{tab:benchmarks}
\begin{tabular}{lcc}
\toprule
Model & A & B \\
\midrule
\modelname{openai}{GPT-OSS-20B} & \bestcid{50.0}{1.0}{1.0} & \cid{40.0}{1.0}{1.0} \\
\modelname{qwen}{Qwen3-30B-A3B} & \cid{30.0}{1.0}{1.0} & \secondcid{45.0}{1.0}{1.0} \\
\bottomrule
\end{tabular}
\end{table}
"""


def tables():
    """The small table, parsed twice: as the accepted revision and as the current manuscript."""
    parsed = parse_table(TABLE, "tables/benchmarks.tex")
    return {"tab:benchmarks": copy.deepcopy(parsed)}, {"tab:benchmarks": copy.deepcopy(parsed)}


class ModelOrderTest(unittest.TestCase):
    def test_display_order_is_ascending_total_parameters(self):
        self.assertEqual(MODELS, ("gpt", "lfm25", "gemma", "nemotron", "glm", "qwen330b", "qwen35", "qwen36"))

    def test_display_order_keeps_every_source_model(self):
        self.assertEqual(SOURCE_MODELS, STORE_MODELS)
        self.assertEqual(sorted(MODELS), sorted(SOURCE_MODELS))
        self.assertEqual(ORIGINAL_MODELS, SOURCE_MODELS[:7])
        self.assertNotIn("lfm25", ORIGINAL_MODELS)

    def test_ticks_follow_the_display_order(self):
        self.assertEqual(TICKS, ",".join(PLOT_LABELS[model] for model in MODELS))
        self.assertTrue(TICKS.startswith(PLOT_LABELS["gpt"] + "," + PLOT_LABELS["lfm25"]))


class SourceComparisonTest(unittest.TestCase):
    def test_a_row_reorder_is_accepted(self):
        current, accepted = tables()
        current["tab:benchmarks"]["rows"].reverse()
        self.assertEqual(compare_source_tables(current, accepted)["status"], "pass")

    def test_swapped_model_values_are_rejected(self):
        current, accepted = tables()
        first, second = current["tab:benchmarks"]["rows"]
        first["cells"][1:], second["cells"][1:] = second["cells"][1:], first["cells"][1:]
        with self.assertRaises(ValueError):
            compare_source_tables(current, accepted)

    def test_a_changed_rank_mark_is_rejected(self):
        current, accepted = tables()
        row = current["tab:benchmarks"]["rows"][0]
        row["cells"][1] = row["cells"][1].replace(r"\bestcid", r"\cid")
        with self.assertRaises(ValueError):
            compare_source_tables(current, accepted)


if __name__ == "__main__":
    unittest.main()
