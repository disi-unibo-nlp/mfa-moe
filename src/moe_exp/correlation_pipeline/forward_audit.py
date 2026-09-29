"""Audit saved forwards, including an independent NumPy metric oracle."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from moe_exp.correlation_pipeline.provenance import file_sha256
from moe_exp.correlation_pipeline.spans import digest, validate_annotation
from moe_exp.jsonl import iter_jsonl
from moe_exp.schemas import TraceRecord


def numerical_oracle(router, hidden, experts, layers, max_geometry_tokens):
    """Recompute scalar definitions from saved arrays without the feature reducer."""
    result = {}
    for i, layer in enumerate(layers):
        logits = np.asarray(router[i], dtype=np.float64)
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        ids = np.asarray(experts[i], dtype=np.int64)
        n, width = p.shape
        k = ids.shape[1]
        chosen = np.take_along_axis(p, ids, axis=1)
        mass = chosen.sum(axis=1)
        other = p.copy()
        np.put_along_axis(other, ids, -1, axis=1)
        top = np.sort(p, axis=1)[:, ::-1]
        primary = ids[np.arange(n), chosen.argmax(axis=1)]
        entropy = -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1)
        values = {
            "router_entropy": entropy.mean(),
            "router_confidence": (1-entropy/math.log(width)).mean() if width > 1 else 1.,
            "router_margin": (top[:, 0]-top[:, 1]).mean() if width > 1 else top[:, 0].mean(),
            "router_selected_mass": mass.mean(),
            "router_boundary_margin": (chosen.min(axis=1)-other.max(axis=1)).mean()
                                      if k < width else chosen.min(axis=1).mean(),
            "router_topk_non_topk_gap": (chosen.mean(axis=1)-(1-mass)/(width-k)).mean()
                                       if k < width else chosen.mean(axis=1).mean(),
            "router_switch_rate": np.mean(primary[1:] != primary[:-1]) if n > 1 else np.nan,
            "router_topk_overlap": np.mean([
                len(set(a) & set(b))/k for a, b in zip(ids[:-1], ids[1:])
            ]) if n > 1 else np.nan,
        }
        if hidden is not None:
            h = np.asarray(hidden[i], dtype=np.float64)
            norms = np.linalg.norm(h, axis=1)
            normalized = h / np.maximum(norms[:, None], 1e-8)
            values["hidden_norm"] = norms.mean()
            values["hidden_step_distance"] = (
                np.mean(1-(normalized[1:]*normalized[:-1]).sum(axis=1)) if n > 1 else np.nan
            )
            sample = np.unique(np.rint(np.linspace(0, n-1, min(n, max_geometry_tokens))).astype(int))
            hn = h[sample] / np.maximum(np.linalg.norm(h[sample], axis=1, keepdims=True), 1e-12)
            pn = p[sample] / np.maximum(np.linalg.norm(p[sample], axis=1, keepdims=True), 1e-12)
            upper = np.triu_indices(len(sample), 1)
            a, b = (hn @ hn.T)[upper], (pn @ pn.T)[upper]
            values["hidden_router_geometry"] = (
                np.corrcoef(a, b)[0, 1] if len(sample) >= 3 and a.std() > 0 and b.std() > 0 else np.nan
            )
        for name, value in values.items():
            result[f"{name}_l{layer:02d}"] = float(value)
    for name in values:
        values_by_layer = np.array([result[f"{name}_l{layer:02d}"] for layer in layers])
        finite = values_by_layer[np.isfinite(values_by_layer)]
        result[f"{name}_mean_layers"] = float(finite.mean()) if len(finite) else math.nan
        result[f"{name}_std_layers"] = float(finite.std()) if len(finite) else math.nan
    return result


def audit_bundle(input_path: Path, *, require_raw=False, require_replay=False):
    import json
    import torch
    from moe_exp.correlation_pipeline.analyze import _resolve_tensor
    from moe_exp.correlation_pipeline.features import FEATURE_SCHEMA_VERSION

    rows = []
    errors = []
    for record in iter_jsonl(input_path):
        trace = TraceRecord(**record)
        key = [trace.model_id, trace.dataset, trace.problem_id, trace.sample_id]
        compact = trace.metadata.get("correlation_features", {})
        row = {"identity": key, "compact_features_available": False,
               "raw_tensors_available": False, "expert_tensor_available": False,
               "label_coverage": {"selected": 0, "labeled": 0, "unknown": 0},
               "expert_by_class_ready": False, "errors": []}
        try:
            if compact.get("schema_version") != FEATURE_SCHEMA_VERSION or not compact.get("values"):
                raise ValueError("Missing or incompatible compact feature schema")
            row["compact_features_available"] = True
            paths = {name: _resolve_tensor(getattr(trace.model_logs, name), input_path)
                     for name in ("selected_experts", "router_logits", "hidden_states")}
            if paths["selected_experts"] is None:
                raise ValueError("Missing expert tensor")
            expert_path = paths["selected_experts"]
            checkpoint_path = expert_path.with_name(expert_path.name.replace("_experts.pt", "_extraction.json"))
            checkpoint = json.loads(checkpoint_path.read_text())
            if checkpoint.get("payload_sha256") != digest(
                {k: v for k, v in checkpoint.items() if k != "payload_sha256"}
            ):
                raise ValueError("Corrupt extraction checkpoint")
            if checkpoint.get("config", {}).get("forward_provenance") != trace.metadata.get("forward_provenance"):
                raise ValueError("Trace/checkpoint revision provenance differs")
            if checkpoint.get("correlation_features") != compact:
                raise ValueError("Trace/checkpoint compact features differ")
            for name, path in paths.items():
                if path is not None and checkpoint["tensor_sha256"].get(path.name) != file_sha256(path):
                    raise ValueError(f"Corrupt tensor: {name}")
            experts = torch.load(expert_path, map_location="cpu", weights_only=True)
            expected = (compact["num_layers"], compact["token_count"], compact["top_k"])
            if (tuple(experts.shape) != expected or experts.dtype not in
                    (torch.int16, torch.int32, torch.int64) or int(experts.min()) < 0
                    or int(experts.max()) >= compact["num_experts"]):
                raise ValueError("Invalid expert tensor shape, dtype, or IDs")
            layers = trace.model_logs.layer_indices
            if layers is None or len(layers) != expected[0] or len(set(layers)) != len(layers):
                raise ValueError("Missing/invalid absolute layer indices")
            row["expert_tensor_available"] = True
            row["shape"] = list(experts.shape)
            row["tensor_bytes"] = {name: path.stat().st_size for name, path in paths.items() if path}
            row["expert_frequencies_by_layer"] = {
                str(layer): {str(int(expert)): int(count) / (expected[1]*expected[2])
                             for expert, count in zip(*torch.unique(experts[i], return_counts=True))}
                for i, layer in enumerate(layers)
            }
            replay = trace.metadata.get("token_replay")
            if require_replay and (not replay or len(replay["completion_token_ids"]) != expected[1]):
                raise ValueError("Missing or misaligned exact token replay")
            row["exact_token_replay_available"] = replay is not None
            annotation = trace.metadata.get("reasoning_annotation")
            views = trace.metadata.get("correlation_views", {})
            if views != checkpoint.get("correlation_views", {}):
                raise ValueError("Trace/checkpoint reasoning views differ")
            if annotation:
                validate_annotation(trace, annotation)
                if views.get("annotation_sha256") != digest(annotation):
                    raise ValueError("Annotation does not match saved views")
                units = annotation["units"]
                row["label_coverage"] = {
                    "selected": len(units), "labeled": sum("label" in u for u in units),
                    "unknown": sum(u.get("status") == "unknown" for u in units),
                }
                spans = {u["index"]: u for u in views.get("sentence_spans", [])}
                row["expert_by_class_ready"] = bool(units) and all(
                    u["index"] in spans and "token_ranges" in spans[u["index"]] for u in units
                ) and row["label_coverage"]["labeled"] > 0
            if paths["router_logits"] is not None and paths["hidden_states"] is not None:
                router = torch.load(paths["router_logits"], map_location="cpu", weights_only=True)
                hidden = torch.load(paths["hidden_states"], map_location="cpu", weights_only=True)
                if tuple(router.shape) != (*expected[:2], compact["num_experts"]) or hidden.shape[:2] != router.shape[:2]:
                    raise ValueError("Raw tensor dimensions do not align")
                recomputed = numerical_oracle(router.numpy(), hidden.float().numpy(),
                                              experts.numpy(), layers,
                                              compact["config"]["max_geometry_tokens"])
                discrepancies = {}
                for name, actual in recomputed.items():
                    saved = compact["values"].get(name)
                    if saved is None and math.isnan(actual):
                        continue
                    if saved is None or not math.isfinite(actual) or not math.isclose(
                        saved, actual, rel_tol=2e-2, abs_tol=2e-3,
                    ):
                        raise ValueError(f"Raw/compact metric mismatch: {name}: {saved} vs {actual}")
                    discrepancies[name] = abs(saved-actual)
                row["raw_tensors_available"] = True
                row["max_absolute_metric_discrepancy"] = max(discrepancies.values(), default=0.)
                row["metric_discrepancies"] = discrepancies
            elif require_raw:
                raise ValueError("Raw router and decoder-input tensors are required")
        except (ValueError, KeyError, OSError, RuntimeError, TypeError, EOFError) as error:
            row["errors"].append(str(error))
            errors.append({"identity": key, "error": str(error)})
        rows.append(row)
    if not rows:
        errors.append({"error": "Empty forward bundle"})
    return {
        "status": "failed" if errors else "complete", "traces": rows, "errors": errors,
        "oracle_tolerance": {"relative": 2e-2, "absolute": 2e-3,
                             "reason": "saved hidden states are bfloat16"},
        "expert_by_class_estimators": "deferred; readiness describes retained inputs only",
        "signals": "router affinities/log probabilities and decoder inputs; not expert-output activations",
    }
