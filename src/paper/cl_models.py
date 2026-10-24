"""Display order of the eight evaluated models in the manuscript's tables and figures.

The order is ascending total parameters (journal/setup.tex); equal sizes keep the order of `store.MODELS`.
Exporters import MODELS and TICKS from here so labels, rows and categorical figure coordinates agree.
`SOURCE_MODELS` is the order of the sealed receipts and `ORIGINAL_MODELS` the seven models of the first
capture. The display order never decides which models a receipt must cover.
"""
from store import MODELS as SOURCE_MODELS

ORIGINAL_MODELS = SOURCE_MODELS[:7]  # native-memory fixtures and the first dynamics capture exist for these only
TOTAL_PARAMETERS_B = {"gpt": 20, "nemotron": 30, "qwen35": 35, "qwen36": 35, "gemma": 26, "glm": 30, "qwen330b": 30,
                      "lfm25": 24}
MODELS = tuple(sorted(SOURCE_MODELS, key=TOTAL_PARAMETERS_B.__getitem__))
PLOT_LABELS = {
    "gpt": r"{\plotmodel{openai}{}{20B}}",
    "nemotron": r"{\plotmodel{nvidia}{3.5}{30B}}",
    "qwen35": r"{\plotmodel{qwen}{3.5}{35B}}",
    "qwen36": r"{\plotmodel{qwen}{3.6}{35B}}",
    "gemma": r"{\plotmodel{gemma}{4}{26B}}",
    "glm": r"{\plotmodel{zhipu}{4.7}{30B}}",
    "qwen330b": r"{\plotmodel{qwen}{3}{30B}}",
    "lfm25": r"{\plotmodel{liquid}{2.5}{24B}}",
}
TICKS = ",".join(PLOT_LABELS[model] for model in MODELS)  # x tick labels of the per-model figures
