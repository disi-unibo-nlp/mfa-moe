"""Preparation seals existing sources. It does not freeze data fits or authorize jobs."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

from .contracts import REPO, check_files, file_sha, load, require, save, seal, source_binding
from .protocol import MODELS, ORDER, protocol
from .saved_data import inventory


def runtime_binding():
    package = importlib.metadata.distribution("vllm")
    require(package.version.split("+")[0] == "0.29.0", "loader requires inspected vLLM 0.29.0")
    root = Path(package.locate_file("vllm"))
    paths = []
    for relative in ("model_executor/layers/fused_moe", "model_executor/layers/quantization",
                     "model_executor/model_loader", "v1/worker", "model_executor/models"):
        paths += list((root / relative).rglob("*.py"))
    paths += [root / "config/load.py", root / "config/cache.py", root / "config/parallel.py"]
    return {str(p): file_sha(p) for p in sorted(set(paths))}


def prepare(output):
    from moe_exp.verification_study.registry import model_contract
    root = Path(output)
    require(not (root / "CAMPAIGN.json").exists(), "campaign already sealed; use a new version for repairs")
    scientific = load(REPO / "report/verification-study-v3/CAMPAIGN.json")["code_files"]
    readiness = load(REPO / "report/verification-study-v3/AUTOMATION_READINESS-r5.json")["code_files"]
    check_files(scientific)
    check_files(readiness)
    inputs, models, files = inventory(), {}, {}
    for model in MODELS:
        original = model_contract(model)
        config = json.loads((Path(original["local_path"]) / "config.json").read_text())
        text = config.get("text_config", config)
        spec = inputs[model]["spec"]
        require(spec["num_experts"] == original["experts"] and spec["top_k"] == original["native_k"] and
                len(spec["layers"]) == original["expected_routed_layers"], "historical/native geometry differs")
        models[model] = seal({**original, "groups": text.get("n_group", 1),
                              "native_layer_limit": text["num_hidden_layers"],
                              "kv_cache_memory_bytes": 8 * 1024**3})
        for field in ("count_archive", "attempts", "provenance", "sentences", "histogram"):
            path = inputs[model][field]
            files[path] = file_sha(path)
    family_path = REPO / "report/experimental-resume-v1/family-freeze.json"
    files[str(family_path)] = file_sha(family_path)
    population_sources = {name: REPO / "report/routing-study-v1" / name for name in (
        "FULL_MATH_QUESTION_TABLE.json", "FULL_MATH_DUPLICATE_GRAPH.json", "INDEPENDENT_FAMILY_MANIFEST.json")}
    population_seals = {name: load(p)["sha256"] for name, p in population_sources.items()}
    files.update({str(p): file_sha(p) for p in population_sources.values()})
    observed = REPO / "report/model-analysis-review-20261008/CHECKPOINT_HEADER_INVENTORY.json"
    files[str(observed)] = file_sha(observed)
    byte_inventory = json.loads(observed.read_text())
    for row in byte_inventory["models"]:
        require(row["minimum_bytes_per_expert"] == row["maximum_bytes_per_expert"],
                "equal count retention does not match expert bytes; repair quotas before fitting")
        models[row["model"]] = seal({**models[row["model"]],
            "serialized_native_expert_bytes": row["native_routed_expert_tensor_bytes"],
            "serialized_retained_expert_bytes_target": row["native_routed_expert_tensor_bytes"] * 3 // 4,
            "serialized_bytes_scope": "header-derived only; actual GPU storage separately qualified"})
    record = save(root / "PROTOCOL.json", protocol())
    return save(root / "CAMPAIGN.json", {"schema": "expert-memory-campaign-v1", "models": models,
                 "engineering_order": list(ORDER), "protocol_sha256": record["sha256"],
                 "code_files": source_binding(), "runtime_files": runtime_binding(),
                 "packages": {n: importlib.metadata.version(n) for n in ("vllm", "torch", "numpy", "scipy", "transformers")},
                 "saved_inputs": inputs, "saved_source_files": files,
                 "population_source_seals": population_seals,
                 "parent_family_path": str(family_path), "scientific_files_preserved": scientific,
                 "readiness_files_preserved": readiness, "submission_authorized": False,
                 "allocation_state": "CPU measurement and qualifications require exact workload/resource approval",
                 "scientific_state": "selection, transitions, loader qualification, memory and prospective accuracy UNMEASURED"})


def freeze_selections(campaign, paths, output):
    require(set(paths) == set(MODELS), "all seven completed selection artifacts required")
    selected = {m: load(p) for m, p in paths.items()}
    require(not list((Path(output).parent / "results").glob("*.start-*.json")),
            "pilot already started before final mask freeze")
    require(all(s["campaign_sha256"] == campaign["sha256"] and s["outcomes_used"] is False
                for s in selected.values()), "foreign or outcome-ranked selection")
    require(len({s["folds_sha256"] for s in selected.values()}) == 1, "cross-model folds differ")
    return save(output, {"schema": "expert-memory-final-masks-v1", "campaign_sha256": campaign["sha256"],
                         "models": selected, "source_files": {str(p): file_sha(p) for p in paths.values()},
                         "pilot_outcomes_exist": False, "accuracy_ranking": False})
