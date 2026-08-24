"""Archived Experiment 2 command-line wrapper."""

from moe_exp.models.routing_extraction import (
    build_parser,
    compute_selected_experts,
    main,
    process_file,
)

__all__ = ["build_parser", "compute_selected_experts", "main", "process_file"]


if __name__ == "__main__":
    main()
