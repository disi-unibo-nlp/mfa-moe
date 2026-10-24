# Native all-layer measurement results

These are observational associations on the available labeled completions. They do not estimate intervention effects. Label cohorts remain separate.

## gpt

Capture: complete; primary analysis: complete; class/temporal supplement: complete.

- `legacy-8158e333fd6a`: six-benchmark aggregate unavailable (unavailable_incomplete_benchmarks).
- `qwen38-mtp3-production-gpt`: six-benchmark aggregate unavailable (unavailable_incomplete_benchmarks).
- `qwen38-mtp3-stratified-gpt`: equally weighted accuracy 60.59% (question-bootstrap 95% interval 56.56–64.75%), mean generated tokens 4486.7, truncation rate 0.03%.

Temporal coverage: 24 actual routed layers; 0 layer/window/metric cells without observations.

Length/cap-adjusted associations passing their prespecified family BH correction: 3196.

- qwen38-mtp3-production-gpt / math500: `saved_class/Monitor/router_topk_non_topk_gap_l22`, r=-0.460, q=3.23e-24, questions=496.
- qwen38-mtp3-production-gpt / math500: `saved_class/Monitor/router_selected_mass_l22`, r=-0.460, q=3.23e-24, questions=496.
- qwen38-mtp3-production-gpt / math500: `saved_class/Monitor/router_confidence_l22`, r=-0.414, q=3.63e-19, questions=496.
- qwen38-mtp3-production-gpt / math500: `saved_class/Monitor/router_entropy_l22`, r=+0.414, q=3.63e-19, questions=496.
- qwen38-mtp3-production-gpt / math500: `saved_class/Monitor/router_confidence_l21`, r=-0.355, q=1.33e-13, questions=496.
- qwen38-mtp3-production-gpt / math500: `saved_class/Monitor/router_entropy_l21`, r=+0.355, q=1.33e-13, questions=496.
- qwen38-mtp3-production-gpt / math500: `saved_class/Monitor/router_topk_non_topk_gap_l23`, r=-0.345, q=7.17e-13, questions=496.
- qwen38-mtp3-production-gpt / math500: `saved_class/Monitor/router_selected_mass_l23`, r=-0.345, q=7.17e-13, questions=496.
- qwen38-mtp3-production-gpt / math500: `saved_class/Monitor/router_selected_mass_l21`, r=-0.339, q=1.9e-12, questions=496.
- qwen38-mtp3-production-gpt / math500: `saved_class/Monitor/router_topk_non_topk_gap_l21`, r=-0.339, q=1.9e-12, questions=496.
- qwen38-mtp3-production-gpt / math500: `saved_class/Monitor/router_entropy_l20`, r=+0.332, q=6.21e-12, questions=496.
- qwen38-mtp3-production-gpt / math500: `saved_class/Monitor/router_confidence_l20`, r=-0.332, q=6.21e-12, questions=496.

## glm

Capture: complete; primary analysis: complete; class/temporal supplement: complete.

- `qwen38-mtp3-stratified-glm`: equally weighted accuracy 73.52% (question-bootstrap 95% interval 70.86–75.88%), mean generated tokens 8565.4, truncation rate 0.19%.

Temporal coverage: 46 actual routed layers; 0 layer/window/metric cells without observations.

Length/cap-adjusted associations passing their prespecified family BH correction: 4475.

- qwen38-mtp3-stratified-glm / olympiad: `class_token_fraction/Monitor`, r=-0.337, q=6.49e-15, questions=539.
- qwen38-mtp3-stratified-glm / olympiad: `class_fraction/Monitor`, r=-0.329, q=3.51e-14, questions=539.
- qwen38-mtp3-stratified-glm / olympiad: `new_global/W256/mixture_shuffled_jsd`, r=-0.296, q=1.83e-10, questions=530.
- qwen38-mtp3-stratified-glm / olympiad: `temporal_complete_global/W256/mixture_shuffled_jsd`, r=-0.296, q=2.41e-10, questions=530.
- qwen38-mtp3-stratified-glm / olympiad: `saved_full/hidden_norm_l04`, r=-0.308, q=2.87e-10, questions=539.
- qwen38-mtp3-stratified-glm / olympiad: `saved_full/router_entropy_l43`, r=+0.302, q=3.34e-10, questions=539.
- qwen38-mtp3-stratified-glm / olympiad: `saved_full/router_confidence_l43`, r=-0.302, q=3.34e-10, questions=539.
- qwen38-mtp3-stratified-glm / olympiad: `new_layer/L18/W256/full_shuffled_jsd`, r=-0.314, q=3.76e-10, questions=530.
- qwen38-mtp3-stratified-glm / olympiad: `temporal_complete/L18/W256/full_shuffled_jsd`, r=-0.314, q=4.97e-10, questions=530.
- qwen38-mtp3-stratified-glm / olympiad: `new_layer/L19/W64/full_local_entropy`, r=+0.305, q=5.68e-10, questions=539.
- qwen38-mtp3-stratified-glm / olympiad: `saved_full/router_confidence_l19`, r=-0.295, q=6.83e-10, questions=539.
- qwen38-mtp3-stratified-glm / olympiad: `saved_full/router_entropy_l19`, r=+0.295, q=6.83e-10, questions=539.

## qwen330b

Capture: not_complete; primary analysis: not_complete; class/temporal supplement: not_complete.

## qwen36

Capture: not_complete; primary analysis: not_complete; class/temporal supplement: not_complete.

## nemotron

Capture: not_complete; primary analysis: not_complete; class/temporal supplement: not_complete.

## gemma

Capture: not_complete; primary analysis: not_complete; class/temporal supplement: not_complete.

## gemma-bf16-kv-sensitivity

Capture: not_complete; primary analysis: not_complete; class/temporal supplement: not_complete.

NVFP4 checkpoint with BF16 KV cache sensitivity; historical generation checkpoint revision and cache dtype remain unverified.

## Qwen/Qwen3.5-35B-A3B-GPTQ-Int4

Required campaign model. Saved generations: 4647; labeling: partial.
Preparation: requires_native_checkpoint_backend_validation_and_measured_sizing.
Capture: not_complete; primary analysis: not_complete; class/temporal supplement: not_complete.
Existing Unsloth 4-bit replay is preserved separately and does not establish native GPTQ/GGUF equivalence.

## qwen3.5-35b-a3b-mtp-ud-q4-k-xl

Required campaign model. Saved generations: 6627; labeling: partial.
Preparation: requires_native_checkpoint_backend_validation_and_measured_sizing.
Capture: not_complete; primary analysis: not_complete; class/temporal supplement: not_complete.
Existing Unsloth 4-bit replay is preserved separately and does not establish native GPTQ/GGUF equivalence.

## Remaining coverage

- qwen330b: Incomplete observed stages: capture, primary, supplement
- qwen36: Incomplete observed stages: capture, primary, supplement
- nemotron: Incomplete observed stages: capture, primary, supplement
- gemma: Incomplete observed stages: capture, primary, supplement
- gemma-bf16-kv-sensitivity: Incomplete observed stages: capture, primary, supplement
- qwen35_gptq: Incomplete observed stages: capture, primary, supplement
- qwen35_gguf: Incomplete observed stages: capture, primary, supplement

Bootstrap replicates resample questions within each benchmark. Matched repeated attempts stay together. Reported extremes are selected exploratory findings; independent replication is required. Pooled motif/hazard tables are descriptive.
