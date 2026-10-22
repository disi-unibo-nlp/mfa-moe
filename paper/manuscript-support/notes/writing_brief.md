# Journal submission manuscript brief

Updated 2026-10-14 for the author's eight-model visual revision.

- Venue: Computational Linguistics, Long Paper submission. Use the downloaded
  official `clv2025.cls` and `compling.bst` byte-for-byte. The manuscript is
  single-spaced without `manuscript`, with named authors and affiliations.
  No extra Long Paper document head is required. Unassigned volume/issue
  metadata remain the official placeholders; no editor or acceptance dates
  are invented. Main text is typically at most 40 pages, excluding
  acknowledgments, references and appendices. Sources:
  https://submissions.cljournal.org/index.php/cljournal/StyleGuide and
  https://submissions.cljournal.org/index.php/cljournal/OriginalSubmissionChecklist.
- Entry: `main.tex`; bibliography: `references.bib`, `compling.bst`.
  Existing verified Tectonic 0.17.0 and private cache provide builds bounded
  to 45 seconds per TeX invocation. Research and agent environments are untouched.
- Author-directed title: **Guiding Mixture-of-Experts Large Language Models through
  Routing Interventions for Efficient Reasoning**. The exact wording realizes
  the author's explicit guidance and remains open to editorial feedback.
- Latest author-approved order: Lorenzo Molfetta, Gianluca Moro.
  Emails: lorenzo.molfetta@unibo.it, gianluca.moro@unibo.it.
  Affiliation: University of Bologna, Italy. No department, contribution
  marker or funding attribution is invented.
- Core question: can state-conditioned routing interventions guide useful
  reasoning while reducing generation or deployment cost? Mathematics is the
  evaluation setting.
  Completed evidence supports routing-state information, model-dependent
  expert coverage and target engagement. Joint causal quality--cost outcomes
  remain unmeasured; the title states the research objective, not a proven
  efficiency gain.
- Abstract: reference-matched problem/bottleneck/mechanism/supported-outcome
  cadence, no data-instance/family/sentence-count inventory, no invented gains.
- Structure: Introduction; mechanism-based Related Work; Reasoning Guidance
  through Routing Interventions; Experimental Setup; Results; Discussion and
  Limitations; Conclusion; scientific appendices.
- Displays: revised visual overview with clear reasoning-state key and symmetric
  rounded connectors; separate worked conditioning figure; small authentic
  model icons; performance means with available symmetric envelopes of saved 95% intervals in
  plus-minus subscripts; bold/underline first/second distinct descriptive ranks, captioned;
  compact diverse panels rather than oversized sparse bars; one shared SPSD FAF7EF
  beige background; seven saved state-transition matrices with red-to-blue probabilities and duration summaries, plus an explicitly pending eighth LFM2.5 panel.
  Occupancy retains seven-class bars for every model. Figures 3/4/7/8 use actual
  reference-paper bar recipes rather than rejected dot/range encodings.
  Seven-state keys use four entries above three, with aligned first three columns.
  Multi-series legends sit below the plots, following the author's explicit
  placement request; this overrides the older default above-axis legend rule.
  Benchmark headers use AIME24/AIME25. Sparse compatible tables are consolidated.
  Figures 1/2 show normalized probabilities and constructed class-specific
  continuations. RIG, ARG, WSC and coverage gain name existing claim-specific
  estimands; no new metric values or scientific computation are introduced.
  Latest corrections require readable orthogonal actual-value labels, vertical beside bars in Figure 8, compact
  complementary subplots, and no bar chart with just two values. The next-state comparison uses three compact native bar panels for all four
  observed gains and their question/pair support, with no two-point lines or
  two one-bar panels. Tables identify full verified versions
  and parameter counts; plot aliases may omit counts once unambiguous. Ordinary
  tabulars use non-enlarging adjustbox width bounds after manual spacing, while
  multipage longtables retain manual widths to preserve pagination.
  The original single occupancy chart could not accommodate all 49 requested
  horizontal labels cleanly. Under the later compact-subplot instruction it uses
  eight model panels, one explicitly pending, with one shared scale and the same seven-class key.
- Latest prose direction: introduce large language models (LLMs), use LLM
  consistently, frame reasoning generally and introduce mathematics as the
  evaluation setting. Abstracts omit colons, model names, our approach, Together
  these findings and long comma-separated lists. Results headings express
  substantive claims without model names/counts or implementation-status tails.
- Pending routing exhibits: author explicitly requests mock values, plots,
  tables and discussion. Their data and source path are separate and reversible,
  their captions carry visible MOCK/TODO replacement notices, and the discussion
  is conditional. Fabricated intervals have no statistical confidence. Replace
  every mock entry before submission. No mock number enters the abstract or
  supported findings. `journal/tables/routing_pending.tex` retains the blank path.
- Evidence boundary: only existing saved real estimates and native fixtures.
  No new fits, resamples, generation, annotation or Slurm allocations. Unsupported
  intervals remain marked; fold observations are diagnostic, not five independent
  replications. All seven observed models remain in scope. The eighth manuscript model is
  LiquidAI/LFM2.5-24B-A2B, whose results and native fixtures are pending. Its
  official configuration supplies geometry only. This revision does not modify
  any sealed scientific campaign or authorize experiments.
- Related work: `report/iclr2027_conference_anonym_private.pdf` is a local discovery
  seed only. Cite verified public primary records for efficient reasoning,
  trajectories, steering, local geometry, MoE adaptation and expert compression.
- Preservation: original report and source closure stay in the pre-journal archive.
  The previously delivered journal PDF/source are preserved under
  `.paper-review/author-revision-20261014/baseline/`. Unrelated research work,
  frozen sources, controllers, outcomes and Git HEAD/index remain untouched.
- Publication: local edits only; no remote push. This task formats a submission
  draft; author approval, replacement of mock results and completed science are
  separate from successful formatting. Overleaf threads are not accessible
  through Git; no absence of comments is inferred.
- Latest table direction: necessary notes appear only in official captions, with
  no sub-legends or redundant brand-icon explanations. Nemotron's full name in
  Table 2 fits on one line through manual spacing and non-enlarging adjustbox.
  State headings and axes use mnemonic Re, An, Ex, Pl, Im, Ve and Mo, with the
  aligned four-plus-three explanatory plot key. Figure 8 uses narrower bars and
  wider inter-group gaps. Full-roster panels use a balanced eight-model layout.
- Skill synchronization: automatic approval review rejected a skill pull as an
  agent-environment modification. The local source is used read-only; the latest
  author feedback is retained in paper-rules.md and the local feedback record.
  No environment changes, installations, commits or remote writes are authorized
  by this manuscript revision. Overleaf synchronization is deferred to preserve
  concurrent local changes and the raw Git indexes.
- Delivery gates: current offline build, lexical checks, source-cell checks,
  independent semantic/visual judgments and verification, at most three rounds.
  Full formatting checks do not imply scientific or submission readiness.

- Cover letter: an editable local draft is supplied with the two current authors,
  article type and keywords. The private ICLR PDF is another group's reference
  paper, as confirmed by the author; it is not a concurrent manuscript. Prior
  publication/history declarations and replacement of mock results remain
  explicit author/submission TODOs. No submission has occurred.

- 2026-10-15 feedback overrides earlier label/CI instructions: use plus-minus symmetric envelopes with the larger saved margin; retain original scientific limits and identify the enclosing transformation. Avoid diagonal labels; Figure 8 values sit vertically beside bars, Figure 6 has a vertical right-side probability scale, repeated panels are tighter, and shared legends use compact gray backgrounds without outlines, except Figure 1 which has no extra legend background. Feedback is recorded locally in LOC-029--031; installed agent skills and remotes are unchanged.

- 2026-10-15 prose correction: remove repeated editorial caveats about comparisons from captions; state the statistical construction directly in Methods. Separate LaTeX prose sentences with a standalone percent line (newline, %, newline), preserving paragraph boundaries and all mathematical/source literals. These author decisions are LOC-032 and LOC-033.

- 2026-10-15 source formatting: write each complete prose sentence on one physical line, including its references, with the standalone percent line between sentences. Preserve displayed equations, tables and literal blocks. This author decision is LOC-034.

- 2026-10-15 visual correction: categorical x labels are always horizontal. Dense model axes use authentic logos with compact version and total-parameter labels; state abbreviations stay horizontal. Figure 8 keeps all 56 numeric values vertical in clear gaps beside the bars. Legends have no enclosing stroke, and Figure 1 has no extra colored legend background inside its existing carrier. Open the final figure renders and their integrated pages before reporting completion; compilation and declared font sizes are insufficient. These corrections are LOC-035--038 and supersede the earlier vertical model labels and outlined legend containers.

- 2026-10-19 results update: the author asked to schedule all remaining jobs and populate the paper. The mock routing exhibits are replaced by the measured eight-model paired contrasts (Table routing, Figure routing, appendix levels, efficiency link and strong-dose tables), and LFM2.5 cells are filled from their receipts. The intervention result is null: targeted edits change neither the next state nor generated tokens and accuracy beyond matched random edits. The abstract, introduction, discussion and conclusion now report this directly; the title is unchanged pending the author's framing decision. Native-memory residence and latency claims are removed. These decisions are LOC-039--041 and supersede LOC-007 and LOC-026.

