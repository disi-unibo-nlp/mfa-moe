import numpy as np
import pytest

from moe_exp.routing_control.historical_analysis_v1 import describe_windows, dose_descriptions, window_contrasts


def fixture(arm="once", seed=0, family="family", prefix="prefix"):
    ids = np.tile(np.arange(8), (300, 40, 1))
    ids[:, 28, 7] = 189
    sentence = {"sentence_index": 0, "segment": 0, "token_start": 0, "token_end": 299,
                "token_indices": list(range(299)), "measurement_status": "parsed_stop", "label": "Verify"}
    record = {"uid": arm + str(seed) + prefix, "family": family, "prefix_uid": prefix,
        "transition": "candidate_to_verify", "arm": arm, "seed": seed, "reasoning_tokens": 299,
        "emitted_tokens": 300, "sentences": [sentence], "reasoning_closed": True,
        "hit_cap": True, "generation_error": False}
    mapped = {"uid": record["uid"], "reasoning_end": 299, "emitted_token_ids": list(range(300)),
              "token_owner": [0] * 299}
    return record, mapped, ids


def test_closed_reasoning_and_total_cap_remain_distinct():
    record, mapped, ids = fixture()
    result = describe_windows(record, mapped, ids, horizon=1024)
    assert result["windows"][1]["observed_reasoning_tokens"] == 43
    assert result["windows"][1]["closure_in_window"]
    assert not result["windows"][1]["cap_censored"]
    assert result["windows"][2]["observed_reasoning_tokens"] == 0
    assert result["trajectory"]["termination"] == "closure"
    # A missing closure-producing prediction row must fail full alignment.
    with pytest.raises(ValueError, match="align"):
        describe_windows(record, mapped, ids[:-1], horizon=1024)


def test_unknown_label_is_missing_occupancy_and_breaks_adjacency():
    record, mapped, ids = fixture()
    record["sentences"][0]["measurement_status"] = "unparsed_or_capped_label"
    result = describe_windows(record, mapped, ids, horizon=1024)
    assert result["windows"][0]["unknown_reasoning_tokens"] == 256
    assert result["windows"][0]["class_occupancy_among_measured_tokens"]["Verify"] is None
    assert not result["trajectory"]["destination_specific_exit_counts"]


def test_tp_dose_is_separate_and_inactive_padded_layer_removed():
    totals = {"active_rows": 2, "target_mass": .5, "weight_l1": .4,
              "actual_target_hits": 1, "native_target_hits": 0}
    rows = [{"action": "x", "layer": layer, "rank": rank, "totals": totals}
            for rank in ("0", "1") for layer in (24, 28)]
    result = dose_descriptions({"records": rows}, [{"name": "x", "kind": "reweight", "experts": [[28, [189]]]}])
    assert len(result) == 2
    assert {r["rank"] for r in result} == {"0", "1"}
    assert all(r["selected_weight_TV_per_active_row"] == .1 for r in result)


def test_pairing_is_equal_family_and_seeds_averaged_first():
    records = []
    for family, prefixes in (("a", ["a1", "a2"]), ("b", ["b1"])):
        for prefix in prefixes:
            for seed in (0, 1):
                for arm in ("once", "native"):
                    r, m, ids = fixture(arm, seed, family, prefix)
                    if arm == "native" or family == "b":
                        ids[:, 28, 7] = 10
                    records.append(describe_windows(r, m, ids, horizon=1024))
    contrasts = window_contrasts(records, [("once", "native")])
    x = next(r for r in contrasts if r["start"] == 0 and r["metric"] == "selection:L28E189")
    assert x["estimate"] == .5  # Family a contributes 1, family b 0.
    assert x["families"] == 2 and x["paired_seed_cells"] == 6
