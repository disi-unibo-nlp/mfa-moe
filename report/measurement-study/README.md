# Complete cross-model measurement campaign

Status: validation_submitted. Existing CPU results are preliminary. Submission IDs and resource reservations are in results/measurement-study/gpu-budget.json.

| Model | Routed layers | Native replay readiness |
|---|---:|---|
| openai/gpt-oss-20b | 24 | debug_validation_ready |
| nvidia/Gemma-4-26B-A4B-NVFP4 | 30 | native_checkpoint_not_resolved |
| Qwen/Qwen3.6-35B-A3B-FP8 | 40 | requires_native_vllm_replay_capture_validation |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | 23 | requires_native_vllm_replay_capture_validation |
| zai-org/GLM-4.7-Flash | 46 | debug_validation_ready |
| Qwen/Qwen3-30B-A3B | 48 | debug_validation_ready |

Every source requires the same metric inventory in campaign.json. Counts refer to routed layers, not total decoder blocks.
Native routing distributions differ (softmax versus normalized sigmoid affinities). Report semantics and normalized supplements; do not pretend these are identical selection rules.
Native selection boundary diagnostics must account for bias/group restrictions; missing adapters are engineering work, not permission to substitute logit top-k.
Unknown or sparse labels break class histories. Completing routing does not fill unlabeled sentences.
Qwen3.5 GPTQ and GGUF are required separate recovery tracks. Their native captures remain unfinished; see EXPERIMENT_ORDER.md and the required recovery manifest.

Validation envelope: at most eight two-GPU, 30-minute debug jobs (8 GPU-hours total), at most two running or pending debug jobs per user. The user authorized this envelope; backend readiness is checked before each submission.
First concrete job: native GPT-OSS all-layer replay of two saved labeled completions, including hidden-state summaries and the new temporal reducers.
Production replay will use normal QoS and a separate measured capped proposal. Validation submissions are recorded in the budget ledger.
