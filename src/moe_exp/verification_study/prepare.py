"""Reuse the existing disjoint family seals and pinned local checkpoints."""
from __future__ import annotations

from pathlib import Path
import importlib.metadata
import sys

from .contracts import file_sha, load, require, save, seal
from .protocol import assignments, family_boundary, protocol
from .registry import MODEL_KEYS, REPO, model_contract

STAGING = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp")
READER = REPO.parent / "cache/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def runtime_binding():
    packages={name:importlib.metadata.version(name) for name in ("vllm","torch","transformers","numpy")}
    root=Path(importlib.metadata.distribution("vllm").locate_file("vllm"))
    relative=("v1/worker/gpu_model_runner.py","v1/core/sched/scheduler.py","model_executor/layers/fused_moe/router/base_router.py",
        "model_executor/models/qwen3_next.py","model_executor/models/qwen3_5.py","model_executor/models/gpt_oss.py",
        "model_executor/models/nemotron_h.py","model_executor/models/gemma4.py","model_executor/models/glm4_moe_lite.py",
        "model_executor/models/glm4_moe.py","model_executor/models/qwen3_moe.py","_custom_ops.py")
    paths={root/name for name in relative if (root/name).is_file()}
    # Bind selector subclasses, runners and their native expert kernels, including
    # the actual Marlin path. Imported inheritance must not escape the source seal.
    paths.update((root/"model_executor/layers/fused_moe").rglob("*.py"))
    paths.update((root/"model_executor/layers/quantization").rglob("*marlin*.py"))
    files={str(path):file_sha(path) for path in sorted(paths)}
    require(files,"installed backend source binding unavailable")
    return seal({"schema":"verification-runtime-v3","python":str(Path(sys.executable).resolve()),
        "packages":packages,"backend_source_files":files})


def code_binding():
    files = list(Path(__file__).parent.glob("*.py"))
    files += [REPO / p for p in (
        "src/moe_exp/routing_control/transitions_v24.py",
        "src/moe_exp/correlation_pipeline/model_profiles.py",
        "src/moe_exp/utils.py", "src/moe_exp/correlation_pipeline/scoring.py",
        "src/moe_exp/correlation_pipeline/outcomes.py", "src/moe_exp/correlation_pipeline/continuation.py",
        "src/moe_exp/correlation_pipeline/answer_equivalence.py", "scripts/campaign_v3/j1_gate.py",
        "scripts/experimental_resume/routing_local_logits_v2.py",
        "scripts/experimental_resume/verification_study_v3.py",
        "scripts/experimental_resume/verification_study_v3_task.py",
        "scripts/experimental_resume/verification_study_v3_gpu.sbatch",
        "scripts/experimental_resume/verification_study_v3_cpu.sbatch")]
    require(all(p.is_file() for p in files), "implementation closure incomplete")
    return {str(p): file_sha(p) for p in sorted(files)}


def prepare(output):
    discovery = load(REPO / "report/experimental-resume-v1/family-freeze.json")
    independent = load(REPO / "report/routing-study-v1/INDEPENDENT_FAMILY_MANIFEST.json")
    full_math = load(REPO / "report/routing-study-v1/FULL_MATH_QUESTION_TABLE.json")
    questions = load(STAGING / "claude-analysis-2026-09-24/steering-v1/manifests/question-table-v1.json")
    boundary = family_boundary(discovery, independent)
    representatives = discovery["new_parent_pools"]["representative_questions"]
    sources = {r["id"]: r for r in full_math["records"]}
    public, grading = {}, {}
    for family in boundary["discovery"]:
        question_id = representatives[family]
        row = questions["questions"][question_id]
        public[question_id] = {"problem": row["problem"], "family": family, "population": "discovery"}
        grading[question_id] = {"gold": row["gold"], "family": family}
    for population in ("validation", "final"):
        for record in independent[population]:
            row = sources[record["question_id"]]
            public[record["question_id"]] = {"problem": row["problem"], "family": record["family"], "population": population}
            grading[record["question_id"]] = {"solution": row["solution"], "family": record["family"]}
    require(len(public) == 176 and all(set(r) == {"problem", "family", "population"} for r in public.values()),
            "public catalog contains nonallowlisted content")
    save(output / "PROTOCOL.json", protocol())
    save(output / "FAMILIES.json", boundary)
    catalog = save(output / "PUBLIC_QUESTIONS.json", {"schema": "verification-public-catalog-v3", "questions": public,
        "boundary_sha256": boundary["sha256"], "no_gold_or_future_text": True})
    save(output / "GRADING_ONLY.json", {"schema": "verification-grading-catalog-v3", "questions": grading,
        "generator_reader_access": "FORBIDDEN", "public_catalog_sha256": catalog["sha256"]})
    reader = save(output / "READER.json", {"schema": "verification-reader-contract-v3", "checkpoint": str(READER),
        "config_files": {str(READER/n): file_sha(READER/n) for n in ("config.json", "tokenizer_config.json")},
        "draws": [0, 1], "output_cap": 1024, "truth_scope": "two fixed draws from one reader; not independent human raters"})
    code = code_binding()
    runtime=save(output/"RUNTIME.json",runtime_binding())
    model_paths = {}
    for key in MODEL_KEYS:
        model = save(output / key / "MODEL.json", model_contract(key))
        model_paths[key] = str(output / key / "MODEL.json")
        for stage, population in (("discovery", "discovery"), ("validation", "validation"), ("final_native", "final")):
            records = [{"family": f, "question_id": representatives[f]} for f in boundary["discovery"]] if population == "discovery" else independent[population]
            save(output / key / (stage.upper() + "_ASSIGNMENTS.json"), assignments(model, stage, records))
    return save(output / "CAMPAIGN.json", {"schema": "verification-campaign-v3", "models": model_paths,
        "protocol_sha256": protocol()["sha256"], "family_sha256": boundary["sha256"],
        "public_catalog_sha256": catalog["sha256"], "reader_sha256": reader["sha256"], "code_files": code,
        "runtime_sha256":runtime["sha256"],
        "historical_recovery": "COMPLETE; DO_NOT_RESTART", "global_campaign_cap": None,
        "user_authorization": "Implement the approved all-seven plan; finite claim-linked jobs, no arbitrary global cap",
        "future_scientific_dispatch": "qualification receipts plus measured role prices; scheduler completion insufficient",
        "independence_scope": "existing full-MATH duplicate graph and previously-used source seal; retain per-model exposure audit",
        "sources": {str(REPO/"report/routing-study-v1/FULL_MATH_DUPLICATE_GRAPH.json"): file_sha(REPO/"report/routing-study-v1/FULL_MATH_DUPLICATE_GRAPH.json"),
                    str(REPO/"report/routing-study-v1/INDEPENDENT_FAMILY_MANIFEST.json"): file_sha(REPO/"report/routing-study-v1/INDEPENDENT_FAMILY_MANIFEST.json")}})
