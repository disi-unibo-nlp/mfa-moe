"""Where the CL manuscript, its figures and its build tools live.

The manuscript folder is an Overleaf checkout: it owns the editable text, the figure sources and the
data behind them. Build output, review notes and old report sources stay in `paper/manuscript-support`.
"""
from pathlib import Path

from store import HOME, WORK

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper/6a510c6db0d9b08aa438c142"
SUPPORT = REPO / "paper/manuscript-support"
FIGURES = PAPER / "figures/journal"   # figure sources, including Overleaf author edits
DATA = FIGURES / "data"               # coordinates and records behind the figures and tables
BUILD = SUPPORT / "build"             # manuscript and figure builds, page renders, review records
NOTES = SUPPORT / "notes"             # writing brief, paper rules, audits
LEGACY = SUPPORT / "legacy"           # earlier report sources, scripts and data

# Tools outside the repository.
PAPER_WRITER = HOME / ".paper-writer-skill/skills/paper-writer"       # rules.py (source files, bibliography)
TECTONIC = WORK / "tools/tmp/verification-paper-tectonic-0.17.0"      # the TeX compiler, its cache and fonts
TECTONIC_SHA256 = "a98aa59ad5c1df39a6c9e56cbfc5088f2b11d6c179c0130b97998e4bd46a46da"  # of TECTONIC/tectonic
