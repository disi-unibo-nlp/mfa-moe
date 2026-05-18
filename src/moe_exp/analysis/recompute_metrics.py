import argparse
import glob
import json
import logging
from pathlib import Path
from pydantic import RootModel

from moe_exp.schemas import TraceRecord
from moe_exp.analysis.classifier import classify_trace
from moe_exp.experiment1.taxonomy import build_row, build_summary

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Recompute taxonomy metrics over existing traces")
    parser.add_argument("--results-dir", type=str, default="results/exp1", help="Base results directory (e.g. results/exp1)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logging.error(f"Results directory {results_dir} does not exist.")
        return

    summary_file = results_dir / "summary.json"
    rows = []

    # Expected folder structure: results/exp1/<model_id...>/<dataset>/traces.jsonl
    # Note: model_id might contain slashes (e.g. allenai/OLMoE...)
    # So we search for any traces.jsonl files
    trace_files = list(results_dir.rglob("traces.jsonl"))
    
    if not trace_files:
        logging.warning(f"No traces.jsonl found under {results_dir}")
        return

    # Process each traces.jsonl
    for trace_file in trace_files:
        logging.info(f"Processing {trace_file}...")
        
        # Read all traces
        records = []
        with open(trace_file, "r", encoding="utf-8") as f:
            for line in f:
                records.append(TraceRecord.model_validate_json(line))

        if not records:
            continue
            
        model_id = records[0].model_id
        dataset_name = records[0].dataset
        
        # Update step_labels
        for record in records:
            # Preserve old first_error_step in case it was a gold label from the dataset
            old_first_error_step = record.step_labels.first_error_step
            
            # Reclassify
            new_labels = classify_trace(record.steps, record.cot_text)
            
            # If the original trace had a non-None first_error_step (meaning it was possibly
            # set by processBench directly, or by prior heuristic), preserve it.
            if old_first_error_step is not None:
                new_labels.first_error_step = old_first_error_step
                
            record.step_labels = new_labels

        # Rewrite updated traces
        with open(trace_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(record.model_dump_json() + "\n")
        
        # Compute taxonomy metrics for this chunk
        rows.append(build_row(model_id, dataset_name, records))

    if rows:
        summary = build_summary(rows)
        # Create summary json
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logging.info(f"Recomputed summary written to {summary_file}")
    else:
        logging.info("No rows generated, summary.json not updated.")

if __name__ == "__main__":
    main()
