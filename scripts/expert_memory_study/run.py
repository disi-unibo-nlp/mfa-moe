"""Real guarded entry file: safe for vLLM's subprocess re-execution."""
from moe_exp.expert_memory_study.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
