# W-0001: Causal rescue, one historical GPT assignment
- date: 2026-10-20
- commit: 00842564a11f9c67106882af4c271a11d7ac7a2f
- source: report/final-routing-rescue-20261020-r3/SAMPLE_CAPTURE.json and WALKTHROUGH_REPLAY.json
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

CPU job 61554184 completed 0:0 after 637 seconds and qualified all 5380 saved
candidates. The sample's replay row is joined back to the qualification index
by UID and its exact canonical hash.

| Field | Actual value |
|---|---|
| qualification | eligible |
| replay SHA-256 | 8444e213f96653f329c656ac5d42f4bf8f7dd655e0b43747c3bcee3267246eb7 |
| generated tokens | 512 |
| extracted sentences | 22 |
| first complete sentence index | 10 |
| first complete token interval | [0, 13) |
| straddles prefix | false |

## 4. Reader and effect join (`rescue_reader.semantic_labels`)

Semantic annotation has not executed. The sentence qualifies for an exact-context
reader task when terminal cohort extraction runs. Qualification alone supplies
no observed semantic label or steering estimate.

## Worth noticing

- Earlier empty admissions may omit `generated`; history lengths establish that they contributed no tokens.
- Passing current capture checks does not retroactively repair the historical source closure.
- Full saved-token replay is allocated evidence; semantic annotation and engineering remain pending.
