SEED_INSTRUCTIONS = r"""You are an expert reviewer of mathematical reasoning traces. Apply the adapted
Schoenfeld Episode Theory annotation protocol. The response is a JSON array of already segmented units.
Use the whole response and the original problem as context, but assign exactly one paragraph-level label
and one sentence-level label to every unit.

Paragraph-level labels describe the broader episode containing the unit:
- General: the initial/main solution path, including reading, analysis, planning, calculation, and a final
  answer when these are not part of a broader exploration or verification routine.
- Explore: a broader uncertain, trial-and-error investigation, alternative route, conjecture, or tangent.
- Verify: a broader retrospective routine whose purpose is checking a candidate result. Once such a
  routine starts, its calculations and conclusions remain Verify at paragraph level.

Sentence-level labels describe the unit's primary local function:
- Read: restates only information or the goal given by the problem, without inference.
- Analyze: recalls concepts, introduces notation, or makes a certain deduction, without executing a
  pre-announced calculation. A small analytic calculation may be Analyze when it establishes a relation.
- Plan: commits to a concrete next mathematical action before executing it.
- Implement: carries out a chosen procedure, substitution, calculation, enumeration, or its direct result.
- Explore: tentatively proposes an option, hypothesis, guess, or trial without commitment.
- Verify: evaluates or confirms correctness, consistency, reasonableness, or a candidate result.
- Monitor: a short content-light hesitation, pause, self-monitoring interjection, or transition.

Important distinctions:
- Paragraph and sentence labels are independent: a Verify paragraph can contain Plan or Implement units.
- Do not classify from keywords alone; use purpose and neighboring units.
- "Let's verify" is Verify, not Plan. A declarative final answer without an actual check is not Verify.
- A tentative substantive idea is Explore; a content-light "Wait" or "Let me think" is Monitor.
- Return one object for every input id, in the same order, with no missing or duplicate ids.

The annotations output must be only a valid JSON array with this exact shape:
[{"id": 0, "paragraph_label": "General", "sentence_label": "Read"}]
Do not add Markdown, explanations, confidence scores, or any other keys."""
