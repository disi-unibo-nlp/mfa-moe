# Generation-settings audit — 2026-09-24

Read-only audit for the joint replanning request. No generation, download, Slurm submission, environment change, or source edit was performed. Read Claude's REPORT.md first. This note does not independently validate re-scoring or estimate regeneration costs.

## Scope and strength of evidence

Examined the current routing-study generation/client/profile code, historical native launchers in the sibling `repo` checkout, installed vLLM 0.29 code, local HF snapshot `generation_config.json`/README/chat templates, small saved server manifests, and exactly the first saved trace of each benchmark for the eight generation tracks below (49 records total). Every benchmark sample in a track agrees on the listed explicit sampling settings. This is a bounded metadata sample, not an exhaustive all-record validation. Existing generation IDs/revision provenance remains as recorded; recommendations from the currently cached revision do not prove that an unrecorded historical revision had the same defaults.

Abbreviations: T = temperature; p = top_p; k = top_k; PP = presence_penalty; RP = repetition_penalty. All sampled completions request **32,768 output tokens**. All request p=.95. k=0 in trace metadata means **k=-1 in the actual API payload**, disabling top-k filtering (`src/moe_exp/correlation_pipeline/client.py:70`). It does not mean “inherit model defaults.”

## Per-model comparison

| Track | Saved explicit sampling | Context and thinking | Local recommendation/default | Finding |
|---|---|---|---|---|
| GPT-OSS-20B | T=.6, p=.95, k=disabled | Generic launcher defaults to 49,152 context; actual historical server context not established by sampled trace. No explicit reasoning_effort; cached template defaults to medium. | Cached generation_config has token IDs/do_sample only, no sampling recommendation; README absent. | **Unknown recommendation**, not a documented sampling violation. Do not infer recommended output cap from its 131,072 architectural context. |
| GLM-4.7-Flash | T=1, p=.95, k=disabled | Saved server manifests: 49,152 context. No disable-thinking override; cached template defaults to thinking. | Own README lines 45–50: most tasks T=1, p=.95, **max new tokens=131,072**. | Sampling matches; **32,768 output cap is one quarter of local card's evaluation budget**. Context also cannot accommodate that larger output budget. |
| Qwen3-30B-A3B | T=.6, p=.95, k=20 | Saved server context 40,960; thinking default enabled. | Own README lines 131/329: thinking T=.6, p=.95, k=20, min_p=0; example line 79 uses 32,768 new tokens. | **Matches documented thinking sampling**; no verified below-recommended-output-budget claim. |
| Qwen3.6-35B-A3B-FP8 | T=.6, p=.95, k=disabled | Saved server context 49,152; MTP; thinking default enabled. Native launcher selects neutral vLLM defaults, so omitted min_p=0, PP=0, RP=1. | Own README lines 670 and 1012–1015: general-thinking T=1, p=.95, k=20, min_p=0, **PP=1.5**, RP=1. T=.6/PP=0 is precise-coding guidance, not general math. Line 564 advises context **at least 128K**; thinking examples lines 695/735/780 use **81,920 output tokens**. | **T, k, PP differ from general-task recommendation**; context below explicit advice. Output cap below thinking examples (these are examples, not a universal mandated minimum). |
| Nemotron-3.5-Lightning NVFP4 + DSpark | T=1, p=.95, k=disabled | Saved server context 49,152; DSpark draft. Thinking enabled by default per card. | Own NVFP4 card lines 54/606: T=1, p=.95; lines 616–630 thinking default and a 16,000-token example. | **Matches specified sampling**. No documented output-cap violation: 32,768 exceeds local example. Its large advertised context alone is not an output recommendation. |
| Gemma-4-26B-A4B NVFP4 | T=.6, p=.95, k=disabled | Explicit enable_thinking=true. Generic launcher default context 49,152, historical server context not independently established here. | Native restored snapshot generation_config lines 11–13: T=1, k=64, p=.95. Same defaults in local Google BF16 metadata. No model card/output recommendation locally. | **T and k override cached native model defaults**. No demonstrated larger-output-budget recommendation; historical generation revision remains unverified. |
| Qwen3.5-35B-A3B GPTQ-Int4 | T=.6, p=.95, k=disabled | Recorded self-MTP draft ID; generic launcher defaults 49,152 context. Historical resolved server/thinking defaults not independently established. | Own checkpoint README/generation_config absent in inspected cache roots. | **Unknown recommendation**; do not transfer Qwen3.6 advice to Qwen3.5. Native checkpoint recovery remains necessary for backend-specific validation. |
| Qwen3.5-35B-A3B GGUF UD-Q4_K_XL | T=.6, p=.95, k=disabled in stored client config | Stored server URL is llama.cpp port 8080; launcher defaults context 49,152. min_p/RP/effective thinking defaults are not recorded in traces and installed historical server runtime was not reconstructed. | Own GGUF card/config absent in inspected cache roots. | **Unknown recommendation and incomplete effective-sampler provenance**. vLLM defaults cannot be asserted for llama.cpp. |

The Qwen3.8 judge is distinct from these generation tracks: its local card recommends thinking T=1,p=.95,k=20,min_p=0,PP=0,RP=1 (README lines 252/502). The preserved label recovery launcher passes those values (`sbatch/label_recovery.sbatch:199–201`), with 16,384 output tokens, explicit reasoning_effort=low (`:202`), and preserved 57,344 context capacity. No change is proposed to its settings from this audit.

## Exact local evidence

Paths below are relative to `/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/` unless absolute.

### Code and historical launchers

- `routing-study/src/moe_exp/correlation_pipeline/model_profiles.py:28–41` defines architectural context by family. `:88` caps the emitted value with `min(49152, generation.context)` for the generic server; model context and effective serving context differ.
- `routing-study/src/moe_exp/correlation_pipeline/generate.py:125–145` records a limited request configuration and hashes it; `:138–140` explicitly enables thinking only for Gemma; `:340–343` defaults to output 32768/T.6/p.95/k0.
- `routing-study/src/moe_exp/correlation_pipeline/client.py:64–78` builds explicit API sampling fields and maps k0→-1. It does not set min_p, PP, RP, or reasoning_effort.
- `routing-study/src/moe_exp/correlation_pipeline/run_all.sh:175` takes profile context; `:294–296` passes max-model-len and `--generation-config vllm`.
- `repo/sbatch/native_qwen36_generate.sbatch:59–62,70,162–182`: 32768/T.6/p.95/k0/context49152, `--generation-config vllm`, explicit request flags.
- `repo/sbatch/native_source_generate.sbatch:56–59,125–142,229–233`: output32768; GLM context49152/T1/k0, Qwen3 context40960/T.6/k20.
- `repo/sbatch/native_nemotron_generate.sbatch:50–52,167–169,261`: output32768/T1/p.95/context49152; default generate.py k0.
- `routing-study/src/common/llamacpp/serve_qwen3_5_35b_a3b_mtp.sh:19,105`: default context49152. No explicit min_p or repetition-penalty in this launcher.

### Sampling precedence, verified in installed vLLM source

Local runtime: `envs/vllm-cu129/lib/python3.11/site-packages/vllm/`.

- `config/model.py:1718–1735`: generation-config=vllm chooses neutral defaults; auto uses checkpoint generation_config; override_generation_config is merged afterwards.
- `entrypoints/openai/chat_completion/protocol.py:628–634`: neutral RP1/T1/p1/k0/min_p0. `:228`: PP defaults0.
- Same file `:667–686`: request T/p/k take precedence whenever non-None; server defaults apply only to omitted values. Changing only a server override cannot fix an explicit client T=.6/k=-1. Fix and record the request contract as well.

### Cached checkpoint evidence

All under `cache/hf/hub/`:

- `models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/README.md:79,131,329`; generation_config T.6/k20/p.95.
- `models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d08a9490559dae23d0cff1d9466213d989/README.md:564,670–678,695,1012–1017`; generation_config T1/k20/p.95.
- `models--zai-org--GLM-4.7-Flash/snapshots/7dd20894a642a0aa287e9827cb1a1f7f91386b67/README.md:45–50`; chat_template.jinja:85 default thinking.
- `models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/snapshots/bee7596271d1495f6992ae224aefde4410e816b8/README.md:54,606,616–630`.
- `models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/generation_config.json` has no sampling values; `chat_template.jinja:203–206` defaults medium reasoning effort.
- Gemma native: `cache/native-replay/Gemma-4-26B-A4B-NVFP4/a19cfe00be84568a6867111c9a68c9c44fdcffe6/generation_config.json:11–13`. Same values in Google metadata revision4d7ae4984b7db7de8f8457170b3f1a419ee76d52.
- Qwen3.5 metadata/cards not present in the project HF cache, HOME HF cache (only version.txt), or the separate user WORK HF cache (only unrelated Qwen2.5VL/MedGemma). No downloads were attempted.

### Saved trace and server evidence

Trace samples are `repo/results/correlation_pipeline/<leaf>/generation/<model-slug>/<dataset>/traces.jsonl:1` for leaves gpt-oss-20b, gemma-nvfp4-nf4, glm-4.7-flash, nemotron-nvfp4-dspark, qwen3-30b-a3b, qwen36-35b-a3b-fp8; datasets math500,aime24,aime25,amc23,minerva,olympiad. Qwen3.5 tracks instead live directly under `repo/results/correlation_pipeline/generation/{Qwen--Qwen3.5-35B-A3B-GPTQ-Int4,qwen3.5-35b-a3b-mtp-ud-q4-k-xl}/<dataset>/traces.jsonl:1`; GGUF also has gpqa_diamond.

Saved server contracts sampled: `repo/results/correlation_pipeline/{qwen36-35b-a3b-fp8,glm-4.7-flash,qwen3-30b-a3b,nemotron-nvfp4-dspark}/parallel/aime24/server_manifest.json`. These directly establish serving context for the four native Slurm generation tracks; generic launcher defaults are weaker evidence for older tracks.

## Implications for the joint plan

1. Preserve existing generations as a **32K-output, historical-sampler experiment**. A non-recommended sampler does not make its routed tensors invalid. It limits model-card comparisons and requires explicit reporting.
2. For Qwen3.6, a fresh recommended-sampler experiment must record T1/k20/PP1.5 explicitly, plus min_p/RP/thinking/context. Keep the existing T.6/k-disabled/PP0 runs as a separate condition. Changing sampler and budget simultaneously cannot identify which caused an accuracy/length change; use matched question/seed subsets and staged contrasts if regeneration is chosen.
3. GLM and Qwen3.6 provide local evidence motivating a targeted larger-budget sensitivity condition. This audit alone does not require full-corpus regeneration; Claude is separately evaluating which truncated inputs and budget increments matter.
4. No local evidence demands larger-output regeneration for GPT, Gemma, Nemotron, Qwen3 or either Qwen3.5 track merely to match a recommendation. Their observed truncation may independently justify targeted sensitivity tests.
5. Extend provenance in any future generation contract: checkpoint revision, server/runtime version, effective min_p/PP/RP, thinking/reasoning_effort, effective context and generation-config mode. Current trace hash (`generate.py:125–145`) omits these, so an unchanged hash is insufficient to establish equality of effective server sampling after such a change. This is a reproducibility/resume-guard gap, not evidence that an already saved trace was modified.
