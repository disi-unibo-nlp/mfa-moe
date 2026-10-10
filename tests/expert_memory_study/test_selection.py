import numpy as np
import pytest

from moe_exp.expert_memory_study.selection import analyze, family_folds, keep_set, scores, validate_counts
from moe_exp.expert_memory_study.protocol import KEEP, MODELS
from moe_exp.expert_memory_study.transitions import analyze_transitions


def test_family_normalization_and_pooled_token_control():
    # Family a contributes one total weight despite its two duplicate questions.
    counts = np.array([[[[9, 1]], [[0, 1]]], [[[9, 1]], [[0, 1]]], [[[0, 1]], [[9, 1]]]])
    balanced, frequency, support = scores(counts, ["a", "a", "b"], [True] * 3)
    assert np.allclose(balanced, [[.45, .55]])
    assert np.allclose(frequency, [[18/22, 4/22]])
    assert support.tolist() == [[2], [2]]
    assert np.allclose(scores(counts[[0, 2]], ["a", "b"], [True, True])[0], balanced)


def test_test_fold_cannot_change_training_ranking():
    rng = np.random.default_rng(1)
    counts = rng.integers(1, 10, size=(15, 7, 2, 8))
    families = [str(i) for i in range(15)]
    folds = family_folds(families)
    train = np.array([folds[f] != 0 for f in families])
    first = scores(counts, families, train)
    counts[~train] = 100000
    second = scores(counts, families, train)
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])


@pytest.mark.parametrize("model", MODELS)
def test_native_group_quotas_and_deterministic_random(model):
    E = KEEP[model] * 4 // 3
    score = np.zeros((2, E))
    mask = keep_set(score, KEEP[model], 4)
    assert len(mask[0]) == KEEP[model]
    assert all(sum(g * E // 4 <= e < (g + 1) * E // 4 for e in mask[0]) == KEEP[model] // 4 for g in range(4))
    assert np.array_equal(keep_set(score, KEEP[model], 4, model=model, random=True),
                          keep_set(score + 99, KEEP[model], 4, model=model, random=True))
    assert mask[0, 0] == 0


def test_selected_slots_validate_per_layer_and_reject_corruption():
    counts = np.ones((3, 7, 2, 4), dtype=np.uint32)
    tokens = np.ones((3, 7), dtype=np.int64) * 2
    validate_counts(counts, tokens, 2)
    counts[0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="selected slots"):
        validate_counts(counts, tokens, 2)


def test_random_mask_does_not_vary_between_folds():
    counts = np.ones((10, 7, 2, 8), dtype=np.uint32)
    result = analyze(counts, [str(i) for i in range(10)], keep=6, groups=2, layers=[1, 3], model="gpt",
                     classes=[str(i) for i in range(7)])
    assert all(r["masks"]["random"] == result["final_masks"]["random"] for r in result["folds"])
    assert result["outcomes_used"] is False


def test_gaps_missing_labels_and_segments_break_adjacency():
    labels = [(0, 0, "a"), (1, 0, "a"), (3, 0, "b"), (4, 0, None), (5, 0, "b"), (6, 1, "a"), (7, 1, "b")]
    rows = [dict(attempt_id="x", family="f", sentence_index=i, segment=s, label=c, n_tokens=3) for i, s, c in labels]
    result = analyze_transitions(rows, ["a", "b"], total_sentences=8)
    assert result["counts"] == [[1, 1], [0, 0]]
    assert result["label_coverage"] == 6/8
    assert result["source_family_support"] == [1, 0]
    assert any(r["left_censored"] for r in result["dwell"])
    assert any(r["right_censored"] for r in result["dwell"])


def test_family_transition_rates_do_not_pool_sentence_mass():
    rows = []
    for family, labels in (("short", ["a", "b"]), ("long", ["a"] * 11)):
        rows += [dict(attempt_id=family, family=family, sentence_index=i, segment=0, label=c, n_tokens=1)
                 for i, c in enumerate(labels)]
    result = analyze_transitions(rows, ["a", "b"], total_sentences=13)
    assert result["family_balanced_probability"][0] == [.5, .5]
