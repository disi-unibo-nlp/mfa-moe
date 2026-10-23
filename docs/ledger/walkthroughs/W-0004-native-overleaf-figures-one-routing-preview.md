# W-0004: Native Overleaf figures, one routing preview
- date: 2026-10-22
- commit: 9961d3337840c6b3966b0901bef0ac80f26be2d9
- source: paper/manuscript-support/build/source-migration/TOOLING_SOURCE.json, figure-builds/routing/BUILD.json, figure-labels-routing/LABEL_GEOMETRY.json and source-migration/NATIVE_BUILD_REVIEW.json
- paths: scripts/paper, paper/manuscript-support
- keywords: overleaf, tikz, figure preview, routing, backup

**Input:**
> figures/journal/routing.tex and figures/journal/data/routing.dat from the active Overleaf paper.

The commit above identifies the research repository baseline, not a clean tooling commit.
Local tooling changes remain uncommitted; TOOLING_SOURCE.json records their exact hashes.
The manuscript itself is committed separately at Overleaf revision bb9fb04e734d50ac7895b76b2d961ea9cc060fb3.

## 1. Resolve canonical sources (cl_paths.FIGURES)

FIGURES is the paper checkout's figures/journal directory, and DATA is its data subdirectory.
The routing.tex SHA-256 is fae4827fd6bbe47433b80b71987d43e80e316afd931ca1db04ea5861064c92e7.
No older editable support copy is overlaid.

## 2. Compile an independent preview (build_cl_figures.build)

The verified compiler reads a self-contained snapshot and writes routing.pdf in the ignored support build folder.
Its SHA-256 is 193188204467c73061e3b68e3a1d3c188dc0265a3586ca7bae44c3999686366e.
Its dimensions are [379.7699890136719, 327.44000244140625] points.
The original PDF backup remains byte-identical to its baseline.

## 3. Check values and marks (verify_cl_figure_labels.main)

The actual preview has 108 expected numeric labels and 108 bars.
The geometry check reports zero number/bar and number/interval-stroke overlaps.
The separately rendered preview is pixel-identical to the original at 144 dpi.

## 4. Render inside the paper (build_cl_manuscript.compile_bounded)

The manuscript inputs routing.tex directly; the native figure is on page 24 of the 57-page offline build.
The runtime log records all 11 native figure inputs, and the build snapshot contains no PDF-backup folder.
The final compiler pass exits 0 with references resolved.

## Worth noticing

- Editable sources have one canonical location shared by Overleaf and local tools.
- Preview builds never overwrite the fixed original PDFs.
- Scientific table cells, data files and manuscript wording are unchanged.
- Existing source-layout and table-checker findings are documented separately from source-packaging verification.
