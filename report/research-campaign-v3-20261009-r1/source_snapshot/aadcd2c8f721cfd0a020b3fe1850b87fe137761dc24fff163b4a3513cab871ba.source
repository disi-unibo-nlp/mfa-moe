"""Actual all-architecture engineering gates; fixtures are never paper outcomes."""
from __future__ import annotations

import os
import time

from .contracts import check_files, digest, identity, load, require, save, seal
from .backend import Backend
from .semantics import candidate_proposals, eligible_proposal
from .storage import commit_generation

PROBLEM = "Find all real values of x satisfying x^2=4 and x>0."


def qualify(campaign_path, key, output, deadline):
    campaign = load(campaign_path, "verification-campaign-v3")
    check_files(campaign["code_files"])
    model = load(campaign["models"][key])
    check_files(model["config_files"])
    output.mkdir(parents=True, exist_ok=True)
    cases, failures, backend = [], [], None
    started = time.monotonic()
    def test(name, condition, detail=None):
        cases.append({"name": name, "pass": bool(condition), "detail": detail})
        require(condition, "qualification failed: " + name)
    def assignment(name, cap, *, prefix=None, arm="native"):
        return {"uid": identity(key, "engineering", name), "model_sha256": model["sha256"],
                "model": key, "stage": "conditional", "seed": 7721, "arm": arm, "cap": cap, "prefix": prefix}
    def run(name, cap, action=None, prefix=None, overrides=None, all_layers=False, forced=True):
        repeat = {"token_id": ordinary, "cap": cap, "overrides": overrides or {}} if forced else None
        receipt = backend.generate(assignment(name, cap, prefix=prefix, arm="target" if action and action["bias"] else "native"),
                                   PROBLEM, action=action, capture_all=all_layers, repeat=repeat)
        saved = commit_generation(output / "fixtures", receipt)
        return receipt, saved
    try:
        backend = Backend(model, deadline=deadline, engineering=True)
        require(backend.runtime["sha256"]==campaign["runtime_sha256"],"qualified runtime differs from campaign")
        require(backend.executor_sha256==digest(campaign["code_files"]),"qualified executor differs from campaign")
        ordinary_ids = backend.tokenizer.encode(" a", add_special_tokens=False)
        require(ordinary_ids, "ordinary fixture token unavailable")
        ordinary = ordinary_ids[-1]
        receipt, first = run("all-layer-noop", 16, all_layers=True)
        geometry = receipt["capture"]["geometry"]
        layer = geometry[0]["layer"]
        native_ids = receipt["capture"]["capture"][str(layer)]["native_ids"]
        expert = min(range(model["experts"]), key=lambda e: (sum(e in row for row in native_ids), e))
        inactive = {"layer": layer, "expert": expert, "bias": 0.}
        active = {**inactive, "bias": 100.}
        test("actual-all-layer-geometry", len(geometry) == model["expected_routed_layers"], geometry)
        parity = receipt["capture"]["same_state_noop"]
        test("same-hidden-state-native-noop", all(v["evaluated_rows"] > 0 and v["ids_equal"] and v["weights_equal"] for v in parity.values()))
        test("all-layer-token-attribution", all(v["positions"] == list(range(16)) for v in receipt["capture"]["capture"].values()))
        natural, _ = run("natural-noop", 16, all_layers=True, forced=False)
        from vllm import SamplingParams
        sampler = dict(model["sampler"])
        if sampler.get("top_k") == 0:
            sampler["top_k"] = -1
        plain = backend.llm.generate([{"prompt_token_ids": backend.prompt_ids(PROBLEM)}],
            SamplingParams(**sampler, seed=7721, max_tokens=16), use_tqdm=False)[0].outputs[0]
        plain_again = backend.llm.generate([{"prompt_token_ids": backend.prompt_ids(PROBLEM)}],
            SamplingParams(**sampler, seed=7721, max_tokens=16), use_tqdm=False)[0].outputs[0]
        native_equal = list(plain.token_ids) == list(plain_again.token_ids)
        hook_equal = list(plain.token_ids) == natural["emitted_token_ids"]
        test("native-output-comparison-diagnostic", bool(plain.token_ids) and bool(plain_again.token_ids) and bool(natural["emitted_token_ids"]),
             {"native_native_equal":native_equal,"hook_native_equal":hook_equal,
              "status":"EXACT_OBSERVED" if native_equal and hook_equal else "CROSS_RUN_FIDELITY_INCONCLUSIVE",
              "scope":"different sampled executions; same-hidden-state routing tensor parity is the mandatory fidelity gate"})
        pulse, _ = run("pulse-256", 300, action=active)
        cap = pulse["capture"]["capture"][str(layer)]
        test("native-k-and-membership-engagement", sum(expert in ids for ids, a in zip(cap["executed_ids"], cap["active"]) if a) >
             sum(expert in ids for ids, a in zip(cap["native_ids"], cap["active"]) if a) and
             any(a and n != e for n, e, a in zip(cap["native_ids"], cap["executed_ids"], cap["active"])))
        test("exact-pulse-boundary", cap["active"] == [i < 256 for i in range(300)])
        marker = backend.closure[0]
        overrides = {str(8+i): token for i, token in enumerate(marker)}
        closure, _ = run("native-closure", 40, action=active, overrides=overrides)
        cc = closure["capture"]["capture"][str(layer)]
        test("stop-after-native-closure", closure["closed"] and cc["active"] == [i < 8+len(marker) for i in range(40)])
        prefix_ids = backend.tokenizer.encode("We obtain $x=2$.\n", add_special_tokens=False)
        # Probe the maximum qualified context, rather than pricing only a tiny
        # prefill and extrapolating it to long native prefixes.
        max_prefix = 32768-20-len(backend.prompt_ids(PROBLEM))
        require(max_prefix >= len(prefix_ids), "qualification prompt exceeds native context")
        prefix_ids += [ordinary]*(max_prefix-len(prefix_ids))
        prefix = {"prefix_token_ids": prefix_ids, "prefix_ids_sha256": digest(prefix_ids)}
        replay, _ = run("exact-prefix-replay", 20, action=active, prefix=prefix)
        test("conditional-prefix-prediction-origin", replay["capture"]["capture"][str(layer)]["positions"] == list(range(len(prefix_ids), len(prefix_ids)+20)),
             {"input_tokens":replay["prompt_input_tokens"]+replay["prefix_input_tokens"],"total_context_tokens":32768})
        restored, _ = run("restored-noop", 20, action=inactive)
        rc = restored["capture"]["capture"][str(layer)]
        test("inactive-and-neighbor-isolation", not any(rc["active"]) and rc["native_ids"] == rc["executed_ids"] and rc["native_weights"] == rc["executed_weights"])
        try:
            backend.generate(assignment("invalid-setup", 2), PROBLEM,
                action={"layer": layer, "expert": model["experts"], "bias": 1.}, repeat={"token_id": ordinary, "cap": 2, "overrides": {}})
        except ValueError:
            pass
        else:
            raise ValueError("invalid expert was accepted")
        after_error, _ = run("native-after-local-rejection", 8, action=inactive)
        test("local-rejection-leaves-engine-usable", after_error["execution_complete"] is True,
             {"scope":"invalid client action rejected before RPC; worker patch rollback tested separately on CPU"})
        candidate_tokens = backend.tokenizer.encode("We obtain $x=2$.\n", add_special_tokens=False)
        online_assignment = assignment("online-trigger", 300, arm="target")
        online_assignment["stage"] = "utility"
        online = backend.generate(online_assignment, PROBLEM, action=active,
            recognition=seal({"schema": "engineering-recognizer-fixture", "status": "PASS", "method": "cheap",
                              "model_sha256": model["sha256"], "engineering_only": True}),
            repeat={"token_id": ordinary, "cap": 300, "overrides": {str(i): t for i,t in enumerate(candidate_tokens)}})
        commit_generation(output / "fixtures", online)
        fired = online["decisions"]
        test("prospective-online-trigger", len(fired) == 1 and
             online["capture"]["capture"][str(layer)]["active"] == [fired[0]["position"] <= i < fired[0]["position"]+256 for i in range(300)])
        test("candidate-specific-intermediate-and-symbolic", bool(eligible_proposal(PROBLEM, "We obtain $x=2$.\n")) and
             bool(eligible_proposal(PROBLEM, "We obtain $x=\\sqrt{4}$.\n")) and
             bool(eligible_proposal(PROBLEM, "The answer is 2.\n")))
        candidates = candidate_proposals(PROBLEM, "We obtain $x=1$.\nWe obtain $y=10$.\nChecking $y=10$, we find $y-10=0$.\n")
        test("checked-other-candidate-and-intent", any(r["lhs"] == "x" and not r["checked"] for r in candidates) and
             bool(eligible_proposal(PROBLEM, "We obtain $x=2$.\nWe should check it.\n")) is False)
        full, full_saved = run("full-cap-16384-price", 16384, action=inactive)
        test("full-cap-execution-and-cost", full["generated_tokens"] == 16384 and full_saved["capture"]["array_bytes"] > 0,
             {"seconds": full["generation_seconds"], "capture_bytes": full_saved["capture"]["array_bytes"]})
    except Exception as exc:
        failures.append({"type": type(exc).__name__, "message": str(exc)[:2000]})
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:
                failures.append({"type": type(exc).__name__, "message": "shutdown: " + str(exc)[:500]})
    return save(output / "QUALIFICATION.json", {"schema": "verification-backend-qualification-v3",
        "status": "PASS" if len(cases) == 14 and all(c["pass"] for c in cases) and not failures else "FAIL",
        "campaign_sha256": campaign["sha256"], "model_sha256": model["sha256"], "code_files": campaign["code_files"],
        "job_id": os.environ.get("SLURM_JOB_ID"), "cases": cases, "failures": failures,
        "load_seconds": backend.load_seconds if backend else None, "elapsed_seconds": time.monotonic()-started,
        "runtime":backend.runtime if backend else None,"engine_kwargs":backend.kwargs if backend else None,
        "engineering_only": True, "scientific_effect": "NOT_MEASURED",
        "scientific_dispatch": "requires discovery nomination, semantic qualification, matched dose/pool, validation and measured prices"})
