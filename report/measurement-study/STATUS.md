# Measurement campaign recovery status

Verified 2026-09-24T10:17:43.772390+00:00. Recovery is in progress; GPU acceptance and remaining full collections are unfinished.

| Source/stage | Observed result | Active followups |
|---|---|---|
| GPT | All captures and primary analysis complete; repaired supplement 58516688 accepted: 1,580 attempts, 9 groups, 50,812 tests | Final report |
| GLM | All captures and primary analysis complete; repaired supplement 58516753 completed 0:0, 1,390 attempts | Final report |
| Qwen3.6 | Four healthy captures preserved | 58492402, 58492407, 58492432, 58492434; primary 58492436; supplement 58516703 |
| Qwen3 labels | Unfinished part validated: 15,616 saved records retained; 9,384 remain | 58517706: two GPUs, six hours, normal QoS; analysis 58518846; preparation 58519367; sizing 58520134; source join 58519372 |
| Nemotron | All four corrected captures passed the selection boundary check, but the historical mismatch reproduced on only one input; strict acceptance failed | Validation 58518118 failed 1:0 after 19m15s; sizing 58520130 canceled by dependency; production remains gated |
| Gemma | All ten checkpoint files verified; source repair 58521473 accepted all 1,642 attempts, no unmatched/rejected/conflicting records | Preparation 58521526 accepted; validation 58521678 failed at startup (FP8 KV cache unsupported on A100); sizing 58521729 canceled; isolated BF16-cache sensitivity proposal staged, not submitted |
| Reporting | Importable multiprocessing launcher regression 58521850 completed 0:0 | Final report 58521917 uses reporting-code-recovery-v4 and observed artifacts |

Gemma uses native NVFP4 checkpoint revision a19cfe00be84568a6867111c9a68c9c44fdcffe6 in private WORK, 18,825,668,663 transferred bytes, 30 routed layers, 128 experts, top-k 8. Historical generation revision remains unverified. Original and repaired sentence annotations are now separate provenance cohorts; old labels, generation files and frozen bundles are preserved. The accepted recovered source collection SHA-256 is 0bf64c0a5790586f9db0eb824fae3651a8d1dc0fc4210c21e052be923220efb2. Production recovery bundle v10 is bound by 379a8b63a98a63f7141c4c454abe43bba41226c2f744cb9c177f024c550b4b65.

Queue lanes retain at most four concurrent GPU jobs: Qwen3.6 shards 0 and 3; Qwen3 label recovery then Qwen3.6 shard 2; Nemotron validation then Gemma validation then Qwen3.6 shard 1. All healthy capture IDs remain intact. Gemma additionally requires successful preparation. Obsolete reporting jobs 58492955/58492958/58492964 and superseded preparation/validation attempts are terminal and have inspected receipts. Gemma's canceled validation 58518836 had no allocation; its original one-GPU-hour reservation was reused.

Current accounting: **155.214/360 GPU-hours spent or committed**, **204.786 remaining**. Finished allocations use observed elapsed GPU time; unfinished jobs retain their full requested maxima. Initial recovery envelope remains 14 GPU-hours (Gemma 1, Nemotron 1, Qwen labels 12). The earlier two label-resume jobs remain in their explicitly separate budget. Historical requested maxima in production-submissions.json are retained for provenance; recovery-accounting.json is the reconciled snapshot.

New full Gemma/Nemotron/Qwen3 capture requests remain staged until native acceptance, measured throughput/output sizing, budget reconciliation and storage checks pass. Requests use 1.5 times measured startup plus collection runtime, rounded up to 30 minutes. Final-report dependencies must incorporate the eventual accepted collection/analysis chains; the current report job will identify absent stages accurately if those remain unfinished. Earlier Qwen GPTQ/GGUF checkpoint/backend and exact-token gaps remain explicit.

Monitoring is attached to this session with bounded foreground watchers. The visible queue was verified empty after privately archiving 12 queued entries containing 14 stale/inspected job notices. Watchers now serialize delivery, defer while the agent is working or inputs are queued, and honor acknowledgments; all 11 watcher regression tests pass. Old login02 watchers cannot be identity-verified from here because strict SSH authentication was denied, so those remote processes were not killed. Private receipts remain in HOME.

Recovery tests include verified/resumable downloads, transport retry limits, label checkpoint preservation and unknown labels, native routing dtype/bias/tie rejection, exact historical sentence spans and split-cohort provenance. A real multiprocessing worker completed through the frozen reporting launcher. GPU validation results remain outstanding.

Detailed receipts: results/measurement-study/recovery-20260924.json, recovery-accounting.json, queue-reorder-20260924.json, gemma-lane-replacement-20260924.json, gpt-supplement-recovery-acceptance.json, recovery-storage/source-joins/gemma-segmentation-v1/acceptance.json. Previous narrative is preserved in STATUS-before-recovery-20260924.md.

Gemma failure diagnosis and concrete cache-sensitivity proposal: gemma-kv-cache-recovery.md. Native preparation and checkpoint remain accepted; primary capture remains incomplete.
