# Paper tooling

The active manuscript is `paper/6a510c6db0d9b08aa438c142`. Its native figures
and plotting data are canonical; local review outputs stay in the ignored
`paper/manuscript-support/build` directory.

`optimize_sections.py` runs a previously approved section-optimization protocol
through the existing ELM batch client. It resumes the saved initial generator
responses, submits two independent counterbalanced GPT-6 Luna High reviews per
proposal, and feeds actual criticism back to GPT-6.1 Sol Medium. All ten rounds
are retained, including unchanged or rejected proposals. Scientific-source
changes and judge-identified regressions veto automatic incumbent selection.
The command does not edit, commit or push the manuscript. Final adoption still
requires an independent comparison with the fixed baseline, source and visual
checks, and the paper-writer workflow.

```sh
python -m paper.optimize_sections \
  --run-dir paper/manuscript-support/build/elm-optimization \
  --client ~/.paper-writer-skill/skills/deep-research/scripts/deep-research
```

The run directory must contain `PROTOCOL.json`, section baselines and complete
author examples, `round-01-generators.jsonl`, its successful responses, and the
judge rubric/schema. Requests and responses are immutable evidence. Completed
selection records resume without more calls; unfinished batches resume through
the existing client's successful-request IDs. Credentials remain with that
client and never enter run artifacts.

Validate selection behavior with:

```sh
python -m unittest discover -s tests -p test_paper_optimization.py
```
