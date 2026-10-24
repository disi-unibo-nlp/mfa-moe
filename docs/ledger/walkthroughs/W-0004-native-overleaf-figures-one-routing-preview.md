# W-0004: Native Overleaf figures, one routing preview
- date: 2026-10-24
- commit: 610ed2a6b1f5089b828de755c0eb951df4df7e08
- source: Overleaf checkout paper/6a510c6db0d9b08aa438c142 at its local commit c98ed00c46604c256056ff72481db97bd49b6fa4 (figures/journal/routing.tex, figures/journal/data/routing.dat, journal/style.tex, journal/results.tex); a fresh run on 2026-10-24 of paper.build_cl_figures routing, paper.verify_cl_figure_labels and paper.build_cl_manuscript on an export of that commit (build output in a scratch folder, not in the repository)
- paths: src/paper/build_cl_figures.py, src/paper/build_cl_manuscript.py, src/paper/verify_cl_figure_labels.py, src/paper/cl_paths.py, scripts/paper/build.sh, tests/paper/test_table_order.py
- keywords: overleaf, tikz, figure preview, routing, backup

**Input:**
> figures/journal/routing.tex and figures/journal/data/routing.dat from the active Overleaf paper.

This is the retrace of the walkthrough first written on 2026-10-22 (Overleaf commit bb9fb04), when the
tooling lived in `scripts/paper` and `paper/manuscript-support`; it is now `src/paper`. `cl_paths.py` finds the
manuscript relative to the repository root, and a fresh git worktree only holds the old PDF-only copy of the
paper folder. The run therefore used `git archive` of the Overleaf commit above, placed beside a copy of
`src/` from this commit. The paper folder itself was only read.

## 1. Resolve canonical sources (`cl_paths.FIGURES`, `cl_paths.DATA`, `build_cl_figures.build`)

`FIGURES` is the paper checkout's `figures/journal`, `DATA` its `data` subfolder, and no older editable copy
is overlaid. The compiler is the verified offline tectonic (sha256 a98aa59a…).

| Source | SHA-256 |
|---|---|
| `figures/journal/routing.tex` | 845ee0bfe798f90b7c31a82e92df52cb38244658db12576472797ea1504903fa |
| `figures/journal/data/routing.dat` | d12cc4aa14a74b32626e7e371a7cf2dcef432ff21bbe624c55cd3753ca99e6e7 |
| `journal/style.tex` | 55e886d701f0c2fcc2cb90aa28626bb507513326dc08e31c27a7e4efa0a79a1c |

Since bb9fb04 (where `routing.tex` was fae4827f…) the commit c98ed00 changed 94 lines of `routing.tex` and 14
lines of `routing.dat`.

## 2. Compile an independent preview (`build_cl_figures.build`)

The build copies the style and the whole `figures/journal` folder into a fresh snapshot, compiles
`\input{figures/journal/routing}` as a standalone page with `--only-cached --untrusted`, and requires one page.

| Result | Value |
|---|---|
| compile time | 8.9 s |
| page size | 379.77 x 327.44 pt (the same size as on 2026-10-22) |
| `routing.pdf` SHA-256 | 5262bc0396e3cea9580d8b29b58365998ac28808ce4b3bedb6ae3f1e55699385 |
| the original PDF in `figures/pdf-backup/routing.pdf` | 4a5515af…, not identical: 97,436 of 1,493,400 pixel bytes differ at 144 dpi |

The backup is the render of the earlier figure; the figure changed afterwards (step 1), so the pixel
identity measured on 2026-10-22 no longer holds, and nothing is overwritten.

## 3. Check values and marks (`verify_cl_figure_labels.main`)

The expected labels are read from the `\node` lines of `routing.tex`, each checked against the `routing.dat`
value it annotates; the PDF is then searched for rotated numeric text, bars and strokes.

| Check | Result |
|---|---|
| expected numeric labels / labels in the PDF / bars | 108 / 108 / 108 |
| short stroke segments checked | 510 |
| number and bar overlaps; number and interval-stroke overlaps | 0; 0 |
| smallest number-to-bar clearance | 25.73 pt |
| rotated x-tick label declarations in any figure source | 0 |

The receipt keeps `visual_inspection_still_required: true`; the script cannot see the picture.

## 4. Render inside the paper (`build_cl_manuscript.main`, `compile_bounded`)

The manuscript inputs `figures/journal/routing` directly (`journal/results.tex`, a `\journalfigure` in a
`figure` environment). Every source file goes into a fresh snapshot, then at most four compiler passes run
until the references stop changing.

| Result | Value |
|---|---|
| passes (seconds) | 14.0, 12.2, 12.1, 12.2; exit code 0 each; a rerun is no longer needed after the fourth |
| pages / SHA-256 of `main.pdf` | 55 / b1b420d599993af2560de0a9812df1e69bc54bbcc2365b6fcaed25d04ceee5e4 |
| the routing figure | Figure 8, page 20 |
| native figure inputs in the compiler log | all 11, once each (`conditioning` to `whole_states`) |
| `pdf-backup` folder in the build snapshot | absent |

The 14 tests of `tests/paper` (`test_table_order.py`, `test_paper_optimization.py`) pass in 0.12 s. They cover
the table order and the section optimizer, not the figure build.

## Worth noticing

- A figure preview takes 9 s and the whole 55-page manuscript 50 s with the cached offline compiler, far
  below the few minutes the build notes assume.
- The PDF backups are history, not a reference for today's figure: the routing figure moved on after them
  and its pixels differ from the backup in 6.5 percent of the render.
- Editable sources have one canonical place, the Overleaf checkout, shared by Overleaf and the local tools; the
  builds only read it and write below `paper/manuscript-support/build`.
- The label check ties each printed number to a row of `routing.dat`: 108 numbers, 108 bars, none touching.
- In a fresh worktree the paper folder is the old PDF-only copy: `build_cl_figures routing` stops there with
  "figure source missing for requested figure(s): routing" instead of rendering old PDFs, which is the useful failure.
