"""Outcome-selected gentle routing steering (ledger F-0004, D-0003); see README.md."""
import os
from pathlib import Path

# Reports and receipts live in the working repository, also when the code runs from a frozen release
# snapshot (which holds only code). Release jobs export WHOLE_ROOT=<repo>/report/<release>.
REPORTS = (Path(os.environ['WHOLE_ROOT']).parent if os.environ.get('WHOLE_ROOT') else
           Path(__file__).resolve().parents[4]/'report')
