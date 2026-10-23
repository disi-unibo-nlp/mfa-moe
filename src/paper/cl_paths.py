"""Locations of the CL manuscript and of its tooling.

The Overleaf checkout owns the editable manuscript, native figure sources and plotting data.
Original figure PDFs are retained separately as backups. Build outputs, review records and
historical evidence stay in the support folder beside the manuscript.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper/6a510c6db0d9b08aa438c142"
SUPPORT = REPO / "paper/manuscript-support"
FIGURES = PAPER / "figures/journal"  # canonical editable figures, including Overleaf author edits
DATA = FIGURES / "data"         # coordinates and records behind the figures and tables
BUILD = SUPPORT / "build"       # manuscript and figure builds, page renders, review records
NOTES = SUPPORT / "notes"       # writing brief, paper rules, audits
LEGACY = SUPPORT / "legacy"     # earlier report sources, scripts and data
