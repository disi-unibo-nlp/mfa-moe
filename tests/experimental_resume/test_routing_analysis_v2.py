from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from moe_exp.routing_control.anticipation_v2 import build_prediction_frame, prefix_baseline
from moe_exp.routing_control.generation_v2 import execute_request
from moe_exp.routing_control.policy_stages_v2 import PrefixPolicyController, THINK_END_ID
from moe_exp.routing_control.prefix import PrefixInput
from moe_exp.routing_control.study_v1 import digest
from moe_exp.routing_control.study_v2 import CAP, EARLY_WINDOW, seal
from moe_exp.routing_control.utility_analysis_v2 import utility_frontier
from test_routing_study_v2 import action, boundary, execution, recognition, response


def test_anticipation_hash_baseline_and_shuffled_control_preserve_composition():
    audit = recognition()
    aligned = seal({"status": "PASS", "prediction_row_alignment": True, "closure_row_included": True})
    prefix = PrefixInput("q", (1,) * 768, "x" * 768)
    recognized = {"audit_sha256": audit["sha256"], "status": "DECIDED", "prefix_sha256": digest(asdict(prefix)),
        "source_state": "Analyze", "candidate_status": "complete_unchecked", "state_history": ["Read", "Analyze"], "run_age_tokens": 128}
    ids = np.tile(np.arange(8), (769, 40, 1))
    ids[::2] += 8
    trace = {"uid": "trace", "population": "discovery", "arm": "native", "seed": 0, "family": "family", "problem": "q",
        "alignment_audit_sha256": aligned["sha256"], "prediction_positions": list(range(769)), "emitted_token_ids": [1] * 769,
        "executed_ids": ids, "selected_weights": np.full(ids.shape, .125), "prediction_rows": [{"uid": "row", "position": 768,
        "outcome": True, "transition": "candidate_to_verify", "recognition": recognized}]}
    frame = build_prediction_frame([trace], audit, aligned, lambda t: "x" * len(t), family_boundary=boundary())
    assert np.array(frame["baseline"]).shape == (1, 144)
    assert np.array(frame["selection"]).shape == (1, 40, 256)
    # After turnover/drift, velocity and acceleration are compositions and
    # therefore exactly preserved by within-block order shuffling.
    temporal, shuffled = np.array(frame["temporal"])[0], np.array(frame["order_shuffled"])[0]
    block = 2 + 2 * 40 * 256
    assert np.array_equal(temporal[2:block], shuffled[2:block])
    assert np.array_equal(temporal[block + 2:], shuffled[block + 2:])
    changed = deepcopy(trace)
    changed["emitted_token_ids"][-1] = THINK_END_ID  # unseen future closure cannot change this prefix
    changed["executed_ids"][-1] = np.tile(np.arange(16, 24), (40, 1))
    other = build_prediction_frame([changed], audit, aligned, lambda t: "x" * len(t), family_boundary=boundary())
    for name in ("baseline", "static", "temporal", "selection"):
        assert frame[name] == other[name]
    assert prefix_baseline(prefix, recognized).tolist() == frame["baseline"][0]


def test_utility_joint_contrasts_seed_averaging_and_missing_sensitivity():
    families = ["f0", "f1", "f2", "f3"]
    arms = ["native", "selected", "matched_random", "incumbent"]
    rows = [{"family": f, "seed": s, "arm": a, "continuation_cap": CAP, "correct": (a == "selected" or f == "f0"),
        "generated_tokens": 100 if a == "selected" else 120, "controller_tokens": 5 if a == "selected" else 0,
        "intervened_positions": 256 if a == "selected" else 0} for f in families for s in (0, 1) for a in arms]
    pairs = [["selected", "native"], ["selected", "matched_random"], ["selected", "incumbent"]]
    result = utility_frontier(rows, families, arms, pairs, draws=500)
    assert len(result["contrasts"]) == 6 and result["arm_means"]["selected"]["total_tokens"] == 105
    assert result["contrasts"][0]["estimate"] == 75 and result["contrasts"][1]["estimate"] == -15
    rows[0]["correct"] = None
    rows[0]["generated_tokens"] = None
    missing = utility_frontier(rows, families, arms, pairs, draws=500)
    assert len(missing["complete_families"]) == 3
    assert missing["contrasts"][0]["missing_accuracy_bounds"] == [75., 87.5]
    assert missing["contrasts"][1]["missing_cost_upper_bound"] == "unbounded"
    rows[1]["continuation_cap"] = 1024
    with pytest.raises(ValueError, match="historical"):
        utility_frontier(rows, families, arms, pairs, draws=500)


class Backend:
    def __init__(self, ex, *, failure=None, cap=False):
        self.qualification = seal({"status": "PASS", "execution_contract_sha256": ex["sha256"],
            "synchronous_one_token_steps": True, "preserves_live_sampler": True})
        self.calls, self.closed, self.failure, self.cap = [], False, failure, cap

    def start(self, assignment, prefix_ids, *, max_new_tokens):
        assert max_new_tokens == CAP and prefix_ids == []

    def step(self, *, policy_name, worker_action, prediction_position):
        self.calls.append((policy_name, prediction_position))
        native = np.tile(np.arange(8), (40, 1))
        actual = native.copy()
        if policy_name != "native": actual[20, 7] = 12
        if self.failure == "inactive": actual[0, 7] = 9
        return {"token_id": THINK_END_ID if prediction_position == 1 else 1, "row_position": prediction_position + (self.failure == "position"),
            "policy_name": policy_name, "executed_ids": actual, "selected_weights": np.full((40, 8), .125, np.float32),
            "native_ids": native, "native_weights": np.full((40, 8), .125, np.float32),
            "finished": prediction_position == (CAP - 1 if self.cap else 2),
            "finish_reason": "length" if self.cap else "stop", "processed_input_tokens": 1}

    def finish(self): self.closed = True


def runtime_fixture():
    from dataclasses import asdict
    ex = execution()
    assignment = {"uid": "request", "problem": "q", "continuation_cap": CAP,
                  "early_endpoint_window": EARLY_WINDOW, "execution_contract_sha256": ex["sha256"],
                  'arm': 'positive', 'action': asdict(action())}
    manifest = seal({"assignments": [assignment], "execution": ex})
    dispatch = seal({"status": "AUTHORIZED", "manifest_sha256": manifest["sha256"]})
    return ex, assignment, manifest, dispatch


def test_request_loop_captures_closure_and_native_answer_without_restarting_sampler():
    ex, assignment, manifest, dispatch = runtime_fixture()
    c = PrefixPolicyController([action()], response, recognition(), uid="request", seed=0)
    backend = Backend(ex)
    receipt, ids, weights = execute_request(assignment, ex, backend, lambda t: "x" * len(t),
        dispatch=dispatch, manifest=manifest, controller=c)
    assert backend.calls == [("verify", 0), ("verify", 1), ("native", 2)] and backend.closed
    assert ids.shape == weights.shape == (3, 40, 8)
    assert receipt["total_policy_tokens"] == 3 + 2 * 3
    from moe_exp.routing_control.capture_v2 import array_binding
    dose = np.zeros((3, 40, 2), np.float32)
    dose[:2, 20, 0], dose[:2, 20, 1] = 1, .125
    assert receipt['inactive_rows_verified'] == 1 and receipt['dose'] == array_binding(dose)
    assert receipt["captures_sha256"] == digest({"ids": array_binding(ids), "selected_weights": array_binding(weights)})


@pytest.mark.parametrize("failure, message", [("position", "prediction row"), ("inactive", "fidelity")])
def test_request_loop_keeps_explicit_failure_and_shuts_down(failure, message):
    ex, assignment, manifest, dispatch = runtime_fixture()
    assignment = {**assignment, 'arm': 'native', 'action': None}
    manifest = seal({'assignments': [assignment], 'execution': ex})
    dispatch = seal({'status': 'AUTHORIZED', 'manifest_sha256': manifest['sha256']})
    backend = Backend(ex, failure=failure)
    with pytest.raises(ValueError, match=message) as error:
        execute_request(assignment, ex, backend, lambda t: "x" * len(t), dispatch=dispatch, manifest=manifest)
    assert error.value.routing_partial["status"] == "EXPLICIT_EXECUTION_FAILURE" and backend.closed


def test_terminal_receipt_accounts_cap_without_firing_after_final_token():
    c = PrefixPolicyController([action()], response, recognition(), uid="request", seed=0, origin_position=CAP - 1)
    prefix = [1] * (CAP - 1)
    c.observe({"problem": "q", "emitted_token_ids": prefix}, lambda t: "x" * len(t))
    # Check the terminal interface on one final token with a one-token shifted
    # initial prefix, avoiding a 16k-long artificial inference loop in tests.
    c.observe({"problem": "q", "emitted_token_ids": prefix + [1]}, lambda t: "x" * len(t), terminal=True)
    assert len(c.ledger()["controller_calls"]) == 1
    assert c.ledger()["decisions"]["verify"]["position"] == CAP - 1


def test_early_substantive_endpoint_separates_cap_closure_and_missing_labels():
    from moe_exp.routing_control.mechanism_v2 import early_endpoint
    def sentence(index, start, end, state, substantive=False):
        return {"sentence_index": index, "token_start": start, "token_end": end, "label": state,
            "label_finish_reason": "stop", "complete": True, "behavior_votes": [
                {"reader": r, "finish_reason": "stop", "prefix_only": True, "visible_token_end": end,
                 "substantive": substantive, "transition": "candidate_to_verify" if substantive else None} for r in (0, 1)]}
    labels = [sentence(0, 0, 3, "Analyze"), sentence(1, 3, 5, "Verify", True)]
    result = early_endpoint([1] * 5 + [THINK_END_ID] + [2] * 10, labels, "candidate_to_verify", finish_reason="stop")
    assert result["substantive_success"] is True and result["observed_tokens"] == 5 and result["closure_competes"]
    assert result["state_trajectory"]["termination"] == "closure"
    labels[1]["label"] = None
    missing = early_endpoint([1] * 5 + [THINK_END_ID], labels, "candidate_to_verify", finish_reason="stop")
    assert missing["substantive_success"] is None
    assert sum(sum(row) for row in missing["state_trajectory"]["transition_counts"]) == 0
    capped = early_endpoint([1] * CAP, [sentence(0, 0, EARLY_WINDOW, "Analyze")], "candidate_to_verify", finish_reason="length")
    assert capped["substantive_success"] is False and capped["window_censored"] and capped["answer_utility"].startswith("SEPARATE")
