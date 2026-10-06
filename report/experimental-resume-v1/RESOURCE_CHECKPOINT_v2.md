# Resource checkpoint, resume v1

The user approved **+11.25 GPU-hours** on 2026-10-01: .75 additional H1–H4
recovery and 10.50 for the separate routing-action study. The later explicit
reply **“Authorize +0.15 GPU-hour for this one recovery”** raised that study's
qualification ceiling from .75 to .90, and its total to **10.65**. Cumulative
additional authorization is **11.40 GPU-hours**. Original and later receipts
remain separate under `steering-v1/runs/resume-v1/`; no stage allocation was
transferred. See `RUNBOOK_v1.md` and exact final accounting in `JOB_AUDIT.json`.

| Line | Ceiling | Verified expenditure / remaining work |
|---|---:|---|
| H1–H4 original + recovery | 1.25 GPU-h | .94611; all four checks passed in recovery 59092583 |
| Legacy X2, generation + native NLL + failures | 8.00 GPU-h | **5.12722**; generation and all 3,354 native NLL measurements complete |
| New-study qualification | .90 GPU-h | **.27222**; three attempts, none reached worker batches; **no further retry authorized** |
| New-study discovery | 2.00 GPU-h | zero; detector, eligibility, worker and full price holds |
| New-study mechanism validation | 2.25 GPU-h | zero; frozen discovery policy and full price required |
| New-study original-prompt 16k utility scout | 4.75 GPU-h | zero; unqualified controller and incomplete price |
| New-study reserve | .75 GPU-h | zero; reserved, no transfer |
| R3-D clean readout / anticipation / B1 | 16 CPU core-h | **15.90111**; registered family and clean B1 incomplete |
| R3-E clean B4 / correctness calibration | 16 CPU core-h | held by clean R3-D inputs; no real-data result |
| M9 closure generation/replay | 2 GPU-h | manifest prepared; qualification and complete grading price held |
| M10 including M8 repeats/hooks | 3 GPU-h | prefixes/folds prepared; PAC labels, capture qualification and full price held |
| Shared forum J1 grading reserve | 2 GPU-h | .86222 spent previously; 1.13778 remains for all shared charges |
| Legacy X3 | 12 GPU-h | unlaunched; G3, builder/full cost and new-study priority gates |

X2 spent 3.68278 GPU-hours on generation (59093506), .48111 on a post-parity
driver failure (59095712), and .96333 on its successful recovery (59101870).
The cancelled incompatible measurement job allocated zero time. The recovery
was priced at 1.5 complete GPU-hours plus .5 contingency after releasing unused
generation reservation: 4.16389 + 1.5 + .5 = 6.16389 within the existing eight.
All cold loads, identical-fixture parity, prefill and measurements are charged.
No additional X2 GPU budget was requested. Saved-results CPU analysis is separately
receipted and included in actual accounting.

Qualification jobs 59096925, 59098926 and 59103206 spent .00722, .13500 and
.13000 GPU-hours. The first failed on a missing module. Both later attempts
stopped at the unchanged 950-second startup/test allowance before model loading.
CPU preparation validated the fixed fixtures, but the latest attempt still used
219.35 setup seconds and had only 880 seconds left at that guard. The recovery
proposal missed remaining runtime setup cost. Its exact failed proposal and
authorization are preserved; they do not authorize another submission.

Repository code now applies the same 950-second allowance **before** runtime
fingerprinting, whose module import belongs to that cold phase, and records phase
timings. This correction is not a GPU qualification. A future recovery needs a
new immutable driver, a complete price including setup and shutdown, and explicit
authorization. .62778 remains on the current stage, but a new 19-minute TP2
allocation with the observed 196-second shutdown allowance would reserve .74222.
There is no automatic retry or borrowing from the reserve, H1–H4, or discovery.

The utility scout's historical pessimistic decode/load scenario is **6.07304
GPU-hours**, already above 4.75 before prefill, detector execution, blind ratings
and retries. It is not a qualified-controller price or physical lower bound.
Missing measurements must be added before any complete revised stage proposal.
Do not silently change its enrolled families, seeds, horizon or primary outcomes.

R3-D's actual combined use stays below 16 despite an internal readout reservation
overrun caused by scheduler shutdown time. Full anticipation cannot fit its
remaining .09889 core-hour. Cached fold predictions and three completed S3 cells
are preserved. A full continuation price must inventory all remaining folds,
noise fits, models and the missing B1 work; preliminary task times are not a
complete price. No reductions or additional fits are authorized here.

M9/M10 retain the existing 2/3-hour GPU lines. Their complete generation/replay/
load and shared grading costs must be reconciled before submission. Neither an
unused legacy line nor the new study's authorization creates a grading allocation.
No further resource increase is requested at this checkpoint.
