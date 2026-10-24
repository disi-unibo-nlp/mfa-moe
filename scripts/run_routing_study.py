#!/usr/bin/env python3
"""Real guarded entry file: vLLM subprocess imports must not start another engine."""
from moe_exp.routing_study.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
