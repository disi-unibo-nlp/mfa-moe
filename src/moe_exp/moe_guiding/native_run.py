"""Execute a previously frozen and qualified branch manifest inside Slurm only.

Example: python -m moe_exp.moe_guiding.native_run --branches branches.json
         --engine engine.json --output /private/project/results/pilot
The engine JSON contains local checkpoint/settings, never credentials. This
command neither submits jobs nor performs qualification implicitly.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import socket

from moe_exp.correlation_pipeline.dynamics.common import digest
from moe_exp.correlation_pipeline.dynamics.contracts import load


class NativeBackend:
    def __init__(self, engine):
        from vllm import LLM
        required = dict(enforce_eager=True, max_num_seqs=1, enable_prefix_caching=False)
        for key, value in required.items():
            if engine.get(key) != value:
                raise ValueError(f"Exact request timing requires {key}={value}")
        if engine.get("speculative_config"):
            raise ValueError("Speculative execution is unsupported")
        if engine.get("worker_extension_cls") not in (None, "moe_exp.moe_guiding.native_integration.NativeWorkerExtension"):
            raise ValueError("Conflicting worker extension")
        self.engine = LLM(**{**engine, "worker_extension_cls":
            "moe_exp.moe_guiding.native_integration.NativeWorkerExtension"})

    def __call__(self, branch):
        from vllm import SamplingParams
        prompt = branch["prompt_token_ids"] + branch["prefix_token_ids"]
        decoding = dict(branch["decoding"])
        if decoding.get("top_k") == 0:
            decoding["top_k"] = -1  # historical HTTP 0 and native -1 both disable top-k
        params = SamplingParams(**{**decoding, "seed": branch["seed"],
                                   "max_tokens": branch["max_new_tokens"], "n": 1})
        self.engine.collective_rpc("native_begin", kwargs=dict(policy=branch["policy"],
            request_id=branch["branch_id"], prompt_ids=branch["prompt_token_ids"],
            prefix_ids=branch["prefix_token_ids"], closing_sequences=branch["closing_sequences"]))
        try:
            outputs = self.engine.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)
        finally:
            reports = self.engine.collective_rpc("native_end")
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise ValueError("Unexpected engine request/sample count")
        response = outputs[0]
        sample = response.outputs[0]
        if response.prompt_token_ids != prompt:
            raise ValueError("Engine changed exact prompt tokens")
        events = reports[0]["events"]
        for report in reports[1:]:
            if report["events"] != events:
                raise ValueError("Tensor-parallel workers disagree on native route telemetry")
        endpoint = len(branch["prefix_token_ids"])
        completion = [e for e in events if e["output_token"] >= endpoint]
        if [e["output_token"] for e in completion] != list(range(endpoint, endpoint + len(sample.token_ids))):
            raise ValueError("Intervention telemetry does not cover exact generated token positions")
        active = [e for e in completion if e["reason"] == "applied"]
        return dict(suffix_token_ids=list(sample.token_ids), suffix_text=sample.text,
            prompt_echo_verified=True, telemetry_verified=True, finish_reason=sample.finish_reason,
            telemetry=events, phase_transitions=reports[0]["phase_transitions"],
            membership_change=sum(e["membership_changed"] for e in active) / len(active) if active else 0.,
            weight_l1=sum(e["weight_l1"] for e in active) / len(active) if active else 0.,
            selected_layer_expert_executions=sum(len(e["selected_ids"]) for e in events),
            expert_executions=None,  # do not extrapolate unobserved layers/shared experts
            controller_seconds=sum(r.get("controller_seconds", 0.) for r in reports),
            reasoning_tokens=None, is_correct=None)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--branches", required=True, type=Path)
    p.add_argument("--engine", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    if (not os.environ.get("SLURM_JOB_ID") or not socket.gethostname().startswith("lrdn")
            or getpass.getuser() != "lmolfett"
            or os.environ.get("SLURM_JOB_ACCOUNT") != "iscrc_miosr"
            or os.environ.get("SLURM_JOB_PARTITION") != "boost_usr_prod"):
        raise RuntimeError("Native continuation inference requires an authorized LEONARDO GPU Slurm job")
    root = args.output.resolve()
    if not any(root.is_relative_to(Path(p)) for p in (
            "/leonardo_work/IscrC_MIOSR/lmolfett", "/leonardo/home/userexternal/lmolfett")):
        raise ValueError("Output is outside approved private storage")
    manifest = load(args.branches, "branches")
    engine = json.loads(args.engine.read_text())
    qualification = manifest["payload"]["qualification"]
    if qualification.get("engine_binding") != digest(engine) or qualification.get("status") != "passed":
        raise ValueError("Engine differs from the qualified configuration")
    from moe_exp.correlation_pipeline.continuation import run_branches
    backend = None
    def generate(branch):
        nonlocal backend
        if backend is None:
            backend = NativeBackend(engine)
        return backend(branch)
    # Check code/output bindings and completed branches BEFORE an expensive model load.
    results = run_branches(manifest, root, generate)
    print(json.dumps(dict(completed=len(results), manifest_binding=manifest["binding"])))


if __name__ == "__main__":
    main()
