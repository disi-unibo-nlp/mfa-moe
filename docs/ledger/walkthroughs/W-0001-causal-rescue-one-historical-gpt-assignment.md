# W-0001: Causal rescue, one historical GPT assignment
- date: 2026-10-20
- commit: 5614db4455f3a970e9b8e5bece8f4bf5d42673ac
- source: report/final-routing-rescue-20261020-r2/SAMPLE_CAPTURE.json
- paths: src/moe_exp/study, tests/study
- keywords: rescue, capture, provenance, historical, qualification

**Input:**
> GPT assignment 3f7281a7f1364ad6222e6f1a5f59010ea169386e2e7e4deeb95250eba17ab0ae, target activation of Monitor, triple-layer regime, olympiad, fold 0.

## 1. Original assignment (`driver.DRIVER`)

The manifest supplies the scientific identity. No outcome changes selection.

| Field | Value |
|---|---|
| family | 0e87fa42b04c1ebbf6a92832cfbcc059ac85caaeecdb83490e45581a47ef3594 |
| seed | 3890345348 |
| horizon | 512 |
| source job | 61497098 |

## 2. Saved capture joins (`cohort.qualify_capture`)

The function reads the original FAST folder in place. It checks the assignment,
profile, source segments, seed, token count, capture positions, expert geometry,
finite weights, membership and pulse boundaries. The returned logical hashes are:

| Receipt | SHA-256 |
|---|---|
| OUTPUT | 4b5ee3428e87080d4911d95f1cdd60f296c58b4013ce95a3af040e8ec56deeab |
| RESULT | 89612db40569fc776e965cd7ea4872dcdffbf8467c21ecffa03991427b6c4e63 |
| CAPTURE | c555db0ff651958814c751f5082a302dbfc0a4748f502c71ed2ec57da9649c64 |

## 3. Allocated replay gate (`cohort.qualify_saved`)

This stage was skipped in the bounded login-side sample. The released CPU stage
must validate saved-token replay and rebuild extraction for every candidate.
This sample does not establish full qualification or a steering effect.

## Worth noticing

- Earlier empty admissions may omit `generated`; history lengths establish that they contributed no tokens.
- Passing current capture checks does not retroactively repair the historical source closure.
- No model was loaded for this trace; inference and full replay remain allocated work.
