"""The user's prospective defaults, not an allocation authorization."""
from .contracts import seal

MODELS = ("gpt", "nemotron", "qwen35", "qwen36", "gemma", "glm", "qwen330b")
ORDER = ("qwen330b", "glm", "qwen36", "qwen35", "gpt", "gemma", "nemotron")
ARMS = ("native", "state_balanced", "labelled_frequency", "random")
KEEP = dict(gpt=24, nemotron=96, qwen35=192, qwen36=192, gemma=96, glm=48, qwen330b=96)
SEED = 20261008
OUTPUT_CAP = 32768
CONTEXT = 40960


def protocol():
    return seal({
        "schema": "expert-memory-protocol-v1", "models": list(MODELS),
        "question": "At common routed-expert byte retention, does state-balanced selection preserve accuracy better than pooled labelled-token frequency using the same observations?",
        "arms": list(ARMS), "keep_per_layer": KEEP, "removed_routed_bytes_target": .25,
        "output_cap": OUTPUT_CAP, "context": CONTEXT, "seed": SEED,
        "sampling": "model-native settings; one deterministic family seed shared across arms/models",
        "selection": {
            "folds": 5, "unit": "duplicate family", "ties": "original expert ID ascending",
            "balanced": "mean across classes of mean across supported families of within-family class selected-slot shares",
            "frequency": "sum of family-weighted counts divided by family-weighted labelled slot mass; equal total weight per family, pooled by observed token mass within that weighting",
            "family_weight": "each family has weight one; each member question has weight 1/family_size",
            "quotas": "equal retention per routed layer and per native contiguous expert group",
            "random": "SHA256(seed,model,layer,group,original expert); same set in every fold",
            "all_reasoning": "offline coverage/overlap sensitivity only",
            "final_fit": "all historical training families before any pilot outcome",
            "outcomes_used": False,
        },
        "transitions": {"common_questions": 522, "common_families": 509,
                        "adjacency": "consecutive original sentence indices in one reasoning segment; gaps break adjacency",
                        "dwell": "observed sentence count/token sum; gaps and segment endpoints censored"},
        "qualification": {"within_execution_only": True, "routing_atol": 1e-6,
                          "routing_rtol": 1e-5, "bf16_atol": 1e-2, "bf16_rtol": 1e-2,
                          "all_layers_prefill_and_decode": True, "cache_equal_across_arms": True},
        "evaluation": {"pilot_families": 32, "contrasts": 21,
                       "pilot_excluded_from_final": True, "alpha": .05,
                       "discordance_upper": "one-sided exact binomial alpha=0.05/21, maximum across contrasts",
                       "sample_rule": "32*ceil(3.038**2*q_upper/(0.05**2*32))",
                       "planning_half_width": .05, "sample_range": [640, 3712],
                       "accuracy_retention_margin": None, "bootstrap_draws": 50000,
                       "unknown_accuracy": "worst-case bounds; all assigned families retained",
                       "random_inference": "conditional on frozen sets; no population of random masks"},
        "deadlines": {"compute_cutoff_utc": "2026-10-13T23:59:59Z", "results": "2026-10-15"},
        "reservation_margin": .25, "submission_authorized": False,
        "historical_native_reuse": "not compatible with new runtime/cap without exact evidence",
        "verification_campaign": "preserve sources, accepted jobs and protected ownership",
    })
