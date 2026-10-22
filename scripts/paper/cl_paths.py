"""Locations of the CL manuscript and of its tooling.

The manuscript folder (also the Overleaf checkout) holds only what main.tex compiles from: the
sections, the tables they input, the figure PDFs and icons, the class, the bibliography style and
references.bib. Everything the tooling reads or writes lives in the support folder beside it.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper/6a510c6db0d9b08aa438c142"
SUPPORT = REPO / "paper/manuscript-support"
FIGURES = SUPPORT / "figures"   # figure sources, build stamps and the exhibit manifest
DATA = FIGURES / "data"         # coordinates and records behind the figures and tables
BUILD = SUPPORT / "build"       # manuscript and figure builds, page renders, review records
NOTES = SUPPORT / "notes"       # writing brief, paper rules, audits
LEGACY = SUPPORT / "legacy"     # earlier report sources, scripts and data
