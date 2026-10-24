# paper: decisions

## figures-live-in-overleaf: Keep native figure sources in Overleaf with original PDF backups
- status: active
- decided by: user (2026-10-22)
- applies to: Editable figures and local build/export tooling for the active CL Overleaf manuscript.
- not to: Scientific result changes, comment-driven redesigns, recomputation or frozen experiment releases.
- chosen: Canonical TikZ sources and plotting data live in the Overleaf checkout under figures/journal; the manuscript renders them directly. Original compiled PDFs stay byte-exact under figures/pdf-backup. Local builds use the canonical sources and write previews outside the checkout.
- rejected: PDF-only Overleaf delivery with editable sources kept solely in the sibling support folder: the author cannot edit the figures directly in the paper project.
- evidence: User: "yes render them directly, but make sure we reain the pdf versions in a folder of the repo for backup"; implementation approved with "Implement the plan." Overleaf commit bb9fb04e734d50ac7895b76b2d961ea9cc060fb3; source-migration/DELIVERY.json and FIGURE_PIXEL_COMPARISON.json verify eleven preserved PDFs and pixel-identical native renders.
- keywords: overleaf, tikz, figures, plots, editable, pdf backup
