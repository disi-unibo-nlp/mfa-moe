# Gemma cache compatibility decision

Validation 58521678 failed during engine startup after 5m10s (0.1722 GPU-hours). The verified native NVFP4 checkpoint requests an FP8 KV cache; the installed Triton attention backend rejects that cache format on SM80 A100 GPUs. No accepted capture was produced. All ten verified checkpoint files, saved labels and 1,642-attempt preparation remain intact. Dependent sizing job 58521729 was canceled before allocation.

Saved generation records identify the served model, sampling settings and exact token IDs, but do not establish the historical KV-cache precision. A BF16-cache replay therefore cannot currently be represented as a verified match to the original cache configuration.

A separate sensitivity validation is staged and has not been submitted. It preserves the native NVFP4 weights, tokenizer, 30 routed layers and learned expert scaling, while explicitly selecting BF16 KV cache. It uses an isolated output root and cannot overwrite primary results. Two CPU tests verify labeling and output-isolation safeguards; frozen code hashes and rebound input manifests also pass validation.

Resources: 2 A100 GPUs, 8 CPUs, 96 GiB, 30 minutes, debug QoS; maximum 1 GPU-hour within the existing campaign ceiling. Scope: short generation, short saved completion, 33,295-token longest saved input, exact token/layer coverage, native dispatch, and unchanged resume. The proposed job shares an existing dependency lane to preserve the four-GPU-job concurrency cap, subject to live queue recheck before submission.

Frozen bundle: results/measurement-study/production-code-gemma-bf16-kv-v1 (SHA256 manifest 81f84161b3ea25e7b712ed4768cd9f85da82cdd376ba9c78082069e083dec60c). Exact command, resources, output root and parent manifest hashes: results/measurement-study/gemma-bf16-kv-proposal.json.

Approval is needed for the alternate cache-precision sensitivity run. The campaign precision policy states: “Any alternate precision is a separately named sensitivity analysis, never a replacement.” Primary native-cache collection remains unaccepted.
