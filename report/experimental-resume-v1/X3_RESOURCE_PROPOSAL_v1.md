# Legacy X3 resource and launch audit v1 (2026-10-02)

The registered two-sign X3 workload is CPU-prepared, but its 12 GPU-hour launch
ceiling does not pass complete-stage pricing. This is a resource gate, not a
negative steering result. No X3 GPU generation or scoring has been launched.

Job `59177995` completed on LEONARDO with exit `0:0` in 40 seconds using four
CPUs and zero GPUs. Its immutable source is
`scripts/experimental_resume/prepare_x3_cpu.py` (SHA256
`1d95a973387d63631ab5595efe2c85476a528e8f52168be3388c35250d3aa823`).
The three output seals verified after the job:

- `X3_G3_SELECTION_v2.json`: binds the earlier X2 G3 estimates and native NLL
  parity to the registered random-dose support rule, each ratio in [0.90, 1.10].
  Exactly two cells pass: `reweight_m1_L1_landmark` and
  `reweight_p2_BAND_landmark`. The prior selection was unchanged. This is
  dose selection only; it establishes no semantic steering.
- `X3_ELIGIBILITY_v1.json`: all 96 frozen dev-disc questions in the pinned
  `forum-v1|` order, both native X1 seed UIDs verified against eight complete
  X1 result shards. There are 77 firing questions and 19 retained nonfires
  (reasoning closure before first qualifying onset), with no replacement.
  Fifty of the first 64 long-endpoint questions and 27 of the other 32 fire.
  The median firing prefix is 4,782 tokens for the long subset and 5,011 for
  the short subset. Two confirm-connected duplicate families (eight questions)
  are flagged for sensitivity; original X3 enrollment remains intact.
- `X3_COST_INVENTORY_v1.json`: frozen 96-row, 77-firing workload and historical
  throughput scenarios, including all 19 nonfires as assigned zero-pulse
  native comparisons.

| Scenario | Firing requests | Maximum new tokens | Maximum prefill tokens | Historical generation + prefill, two A100 GPU-hours |
|---|---:|---:|---:|---:|
| One sign, N/E/M | 462 | 8,429,442 | 2,479,554 | 8.049 |
| Registered two signs, N/E−/M−/E+/M+ | 770 | 14,049,070 | 4,132,590 | 13.104 |

These are maximum-token planning scenarios at the X1 historical steady rate,
derated 25%, plus one historical cold load and Q10 prefill rate. They are not
measured 32k X3 throughput or a lower bound on actual consumption; natural
stops may use fewer tokens, and long contexts may run slower. The two-sign
scenario already exceeds 12 GPU-hours before its missing components are
charged. Dropping one sign to fit the cap would change the registered G3/X3
design. E4 and dev-spare escalation remain separate, unqualified options.

Complete-stage pricing still needs: a context-matched 32k generation/prefill
pilot under the frozen sampler; frozen GEPA sentence-labeler throughput for
the approximately 20,000 registered X3 sentences; correctness grading and
blind-audit cost; any required X3 native NLL measurement; retry and shutdown
reserve. The known X2 native NLL expense (0.963 GPU-hour) is evidence for
its own 3,354 short sequences and cannot be copied to X3 long branches.
The separate dense-discovery labeling run is a different prompt program; its
throughput cannot certify the GEPA rate. After those measurements, calculate
one all-in price over the 770-request maximum, including every model load and
retry. A versioned resource amendment is necessary if that total exceeds
12 GPU-hours. The X3 builder must refuse launch until the amended ceiling and
new-study-priority gate are sealed. The user's broad GPU authorization permits
preparing a larger allocation, but it does not silently change the registered
12-hour protocol ceiling.

The separate `x3_build_v2.py` fixes an engineering error in the original
builder: it had imported the X2 guard, which hard-requires **dev-cal**, while
X3 is frozen to **dev-disc**. The corrected guard enforces all 96 dev-disc
questions without excluding confirm-connected development families. It also
binds the v2 G3 selection, eligibility and price seals. A negative dry call
verified that the sealed CPU artifacts pass those checks and the incomplete
price blocks manifest creation. The original `x3_build.py` and legacy protocol
remain unchanged. A later version must explicitly accept any resource
amendment; v2 still enforces the original 12 GPU-hour ceiling.
