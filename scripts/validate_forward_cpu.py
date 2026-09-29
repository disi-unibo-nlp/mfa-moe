"""Run tensor CPU gates once inside Slurm; never submit a job."""
import ast
import json
import os
from pathlib import Path
import sys
import time

if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
    raise SystemExit("CPU tensor validation requires an allocated srun step")
started = time.monotonic()
import torch
import scipy
import transformers
print(f"T0_IMPORT_OK elapsed_seconds={time.monotonic()-started:.3f}", flush=True)
assert not torch.cuda.is_available()
import pytest
from moe_exp.correlation_pipeline.provenance import code_provenance

selection = {
    "tests/test_correlation_model_support.py": None,
    "tests/test_qwen_quantized.py": ("split", "reject", "loader"),
    "tests/test_labels_export.py": None,
    "tests/test_native_source_generate.py": ("forward",),
    "tests/test_streaming_routing_extraction.py": None,
    "tests/test_expert_identity_analysis.py": None,
    "tests/test_reasoning_views.py": (
        "stratified", "unknown", "boundary", "transition", "views_pool",
        "all_stages_roundtrip", "forward_views",
    ),
    "tests/test_correlation_pipeline.py": (
        "hand_computed", "reduced_on_the_fly", "forward_extraction",
        "wrong_top_k", "avg32", "feature_nan", "cross_feature",
    ),
}
nodes = []
for filename, keywords in selection.items():
    if keywords is None:
        nodes.append(filename)
    else:
        for node in ast.parse(Path(filename).read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                if not node.name.startswith("test_cuda_") and any(word in node.name for word in keywords):
                    nodes.append(f"{filename}::{node.name}")
flags = ["-q", "-p", "no:cacheprovider", *nodes]
collected = int(pytest.main(["--collect-only", *flags]))
class Coverage:
    def __init__(self):
        self.passed = 0
        self.skipped = []

    def pytest_collectreport(self, report):
        if report.skipped:
            self.skipped.append(report.nodeid)

    def pytest_runtest_logreport(self, report):
        if report.skipped or getattr(report, "wasxfail", False):
            self.skipped.append(report.nodeid)
        elif report.when == "call" and report.passed:
            self.passed += 1

coverage = Coverage()
result = int(pytest.main(flags, plugins=[coverage])) if collected == 0 else collected
if result == 0 and (coverage.skipped or not coverage.passed):
    result = 1
preparation = None
if result == 0 and "--prepare-qwen36" in sys.argv:
    from moe_exp.correlation_pipeline.forward_debug import REPO, prepare_qwen36
    print("PREPARE_EXISTING_QWEN36_LABELS", flush=True)
    try:
        preparation = prepare_qwen36(REPO / "results/correlation_pipeline/forward-validation-v2")
    except Exception as error:
        import traceback
        traceback.print_exc()
        preparation = {"error": str(error)}
        result = 1
report = {"preparation": preparation, "status": "passed" if result == 0 else "failed", "exit_code": result,
          "code": code_provenance(), "nodes": nodes,
          "passed_tests": coverage.passed, "skipped_tests": coverage.skipped,
          "elapsed_seconds": time.monotonic()-started,
          "torch": torch.__version__, "transformers": transformers.__version__,
          "scipy": scipy.__version__, "job_id": os.environ["SLURM_JOB_ID"]}
path = Path(sys.argv[1]).resolve()
repo = Path(__file__).resolve().parents[1]
if not path.is_relative_to(repo):
    raise SystemExit("Gate output must be inside the repository")
path.write_text(json.dumps(report, indent=2)+"\n")
assert json.loads(path.read_text()) == report
raise SystemExit(result)
