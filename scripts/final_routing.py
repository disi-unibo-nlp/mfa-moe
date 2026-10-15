#!/usr/bin/env python3
"""Eight-model final-routing CLI; all heavy work executes in Slurm workers."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

if __name__=='__main__':
    from moe_exp.final_routing.cli import main
    raise SystemExit(main())
