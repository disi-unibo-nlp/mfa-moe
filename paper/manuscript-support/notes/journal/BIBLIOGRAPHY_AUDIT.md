# Offline bibliography evidence

Checked 2026-10-14. The revised active manuscript cites 27 keys.
Each has an independently inspected public primary record in
`.paper-review/author-revision-20261014/PRIMARY_RECORDS.json`.
The structured snapshots contain the title, complete ordered credited names,
publication year, source URL, a short source quotation, source location, and
edition-specific notes. They were transcribed from public publisher or author
records, rather than produced by echoing the project's bibliography.

Source-snapshot SHA-256: `d64b87cc7b8727213db3163b1b38418db632f74b74a8f6abf6c1da580c34f70a`.

Automatic approval review rejected the proposed bulk external bibliography
lookup because it would disclose unpublished manuscript bibliography metadata.
That lookup was not retried. The alternative retrieves known public source
pages directly and performs the comparison locally with network lookup disabled.
No project bibliography, citation list, manuscript text, or private-paper
content was uploaded to a metadata API. The private PDF remains only a source
of discovery leads; its claims and metadata are excluded from the records.

This mode verifies citation identity against inspected primary records. It is
an explicitly different mode from an ordinary `pw bib` invocation:

- All 27 active keys have title, ordered-author, and publication-year evidence.
- Published editions are recorded where verified; Eisenstadt's Overclocking,
  THESIS-MoE, and the counterfactual-routing paper retain preprint status.
- GEPA and SteerMoE names follow published PDF title pages where their landing
  pages abbreviate or expand author names differently.
- REAP's publication year is 2026, verified by its published PDF header; its
  arXiv identifier begins with 2025.
- Benjamini--Hochberg's original journal issue is January 1995; the publisher's
  2018 online migration date is not substituted for that original year.
- Venhoff's public v4 verifies workshop publication and its current author
  order. The first two authors are marked equal contributors.
- Shazeer's original public PDF verifies the credited names and title; DBLP's
  conference record independently confirms ICLR Poster 2017. The old public
  PDF retains its historical under-review header.

No unresolved metadata fields remain among these 27 structured records.
The existing comparison function was run offline, with its actual output
recorded separately. This document does not claim that ordinary API-based
`pw bib` completed or that Rebiber normalization was performed. Existing citation keys and unrelated records
remain preserved; claim support is documented in `CITATION_AUDIT.md`.


Local comparison completed: 27 active entries, 27 OK, zero missing or mismatched title/author/year records. The existing skill comparator was run with network access disabled against the independently transcribed primary records; normalization was not run and no external API lookup occurred. The local receipt binds bibliography and primary-record hashes.
