import unittest

from moe_exp.routing_control.trajectory_analysis_v1 import analyze, digest


def _vote(event="candidate_to_verify", substantive=True):
    return [{"reader": f"reader{i}", "finish_reason": "stop", "transition": event,
             "substantive": substantive} for i in (0, 1)]


def _sentence(index=6, label="Verify", start=0, end=2, event="candidate_to_verify",
              complete=True):
    return {"sentence_index": index, "segment": 0, "token_start": start,
            "token_end": end, "complete": complete, "label": label,
            "label_finish_reason": "stop", "behavior_votes": _vote(event)}


def _input():
    assignments, observations = [], []
    for family in ("f1", "f2"):
        for arm in ("native", "policy"):
            uid = family + "-" + arm
            base = {"uid": uid, "question": family + "-q", "family": family,
                    "arm": arm, "seed": 0}
            assignments.append({**base, "trigger_sentence_index": 5})
            observations.append({**base, "status": "complete", "correct": 1,
                                 "injected_token_count": 0, "emitted_token_ids": [11, 12],
                                 "reasoning_closed": False, "reasoning_token_end": 2,
                                 "token_owner": [6, 6], "routed_positions": [0, 1],
                                 "sentences": [_sentence(event="candidate_to_verify" if arm == "policy" else None)]})
    value = {"schema": "dense-trajectory-analysis-input-v1", "horizon": 1024,
             "population": "fixture", "ordered_transitions": ["candidate_to_verify"],
             "bindings": {name: "0" * 64 for name in (
                 "generation_manifest_sha256", "generation_summary_sha256",
                 "tokenizer_config_sha256", "sentence_label_binding_sha256",
                 "semantic_rating_binding_sha256", "blind_map_sha256", "rubric_sha256")},
             "comparison_pairs": [["policy", "native"]],
             "assignments": assignments, "observations": observations}
    value["sha256"] = digest(value)
    return value


def _seal(value):
    value["sha256"] = digest({k: v for k, v in value.items() if k != "sha256"})
    return value


class DenseTrajectoryAnalysisTests(unittest.TestCase):
    def test_all_assignment_itt_and_no_cross_request_transition(self):
        result = analyze(_input(), n_boot=100)
        self.assertEqual(result["assignments"], 4)
        self.assertEqual(len(result["receipts"]), 4)
        self.assertEqual([r["semantic_success"] for r in result["receipts"]], [0, 1, 0, 1])
        self.assertEqual(sum(sum(row) for row in result["class_summary"]["transition_counts"]), 0)
        contrasts = result["paired_itt"]["contrasts"]
        self.assertEqual(len(contrasts), 3)
        self.assertEqual({row["metric"]: row["estimate"] for row in contrasts}, {
            "semantic_success": 1.0, "correct": 0.0, "tokens": 0.0})

    def test_missing_receipt_and_identity_mismatch_are_rejected(self):
        value = _input()
        value["observations"].pop()
        with self.assertRaisesRegex(ValueError, "missing ITT"):
            analyze(_seal(value), n_boot=100)
        value = _input()
        value["observations"][0]["family"] = "wrong"
        with self.assertRaisesRegex(ValueError, "mismatched observation"):
            analyze(_seal(value), n_boot=100)

    def test_missing_blinded_measurement_binding_is_rejected(self):
        value = _input()
        del value["bindings"]["blind_map_sha256"]
        with self.assertRaisesRegex(ValueError, "bindings required"):
            analyze(_seal(value), n_boot=100)

    def test_dense_indices_token_offsets_routing_and_closure_must_agree(self):
        mutations = [
            lambda row: row["sentences"][0].update(sentence_index=7),
            lambda row: row["sentences"][0].update(token_end=3),
            lambda row: row.update(token_owner=[6, None]),
            lambda row: row.update(routed_positions=[0]),
            lambda row: row.update(reasoning_token_end=1),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                value = _input()
                mutation(value["observations"][0])
                with self.assertRaises(ValueError):
                    analyze(_seal(value), n_boot=100)

    def test_closure_excludes_later_tokens_and_late_sentences(self):
        value = _input()
        row = value["observations"][1]
        row.update(emitted_token_ids=[11, 12, 248069, 99], reasoning_closed=True,
                   reasoning_token_end=2)
        self.assertEqual(analyze(_seal(value), n_boot=100)["receipts"][1]["semantic_success"], 1)
        row["sentences"][0]["token_end"] = 3
        with self.assertRaisesRegex(ValueError, "exceed emitted reasoning"):
            analyze(_seal(value), n_boot=100)

    def test_incomplete_final_sentence_and_reader_disagreement_cannot_succeed(self):
        value = _input()
        policy = value["observations"][1]
        policy["sentences"][0]["complete"] = False
        self.assertEqual(analyze(_seal(value), n_boot=100)["receipts"][1]["semantic_success"], 0)
        value = _input()
        policy = value["observations"][1]
        policy["sentences"][0]["behavior_votes"][1]["substantive"] = False
        receipt = analyze(_seal(value), n_boot=100)["receipts"][1]
        self.assertEqual((receipt["semantic_success"], receipt["semantic_measurement"]),
                         (0, "reader_disagreement"))

    def test_two_action_order_and_trigger_exclusion(self):
        value = _input()
        value["ordered_transitions"] = ["candidate_to_verify", "approach_to_commit"]
        for row in value["observations"]:
            row.update(emitted_token_ids=[11, 12, 13, 14], reasoning_token_end=4,
                       token_owner=[6, 6, 7, 7], routed_positions=[0, 1, 2, 3],
                       sentences=[_sentence(), _sentence(index=7, label="Plan", start=2, end=4,
                                                          event="approach_to_commit")])
        result = analyze(_seal(value), n_boot=100)
        self.assertEqual([r["semantic_success"] for r in result["receipts"]], [1, 1, 1, 1])
        self.assertEqual(sum(sum(row) for row in result["class_summary"]["transition_counts"]), 4)
        for row in value["observations"]:
            row["sentences"][0]["behavior_votes"] = _vote("approach_to_commit")
            row["sentences"][0]["label"] = "Plan"
            row["sentences"][1]["behavior_votes"] = _vote("candidate_to_verify")
            row["sentences"][1]["label"] = "Verify"
        self.assertTrue(all(r["semantic_success"] == 0 for r in analyze(_seal(value), n_boot=100)["receipts"]))

    def test_nonfire_failure_and_injected_tokens_remain_in_itt(self):
        value = _input()
        nonfire = value["observations"][0]
        nonfire["status"] = "nonfire"
        nonfire["injected_token_count"] = 3
        natural_nonfire = value["observations"][1]
        natural_nonfire["status"] = "nonfire"
        failed = value["observations"][2]
        failed["status"] = "failure"
        failed["correct"] = 0
        failed["injected_token_count"] = 2
        receipts = analyze(_seal(value), n_boot=100)["receipts"]
        self.assertEqual((receipts[0]["tokens"], receipts[0]["semantic_success"]), (5, 0))
        self.assertEqual((receipts[1]["status"], receipts[1]["semantic_success"]), ("nonfire", 1))
        self.assertEqual((receipts[2]["tokens"], receipts[2]["correct"]), (4, 0))


if __name__ == "__main__":
    unittest.main()
