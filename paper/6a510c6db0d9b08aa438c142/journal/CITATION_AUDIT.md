# Citation and literature audit

Checked 2026-10-14 against the active journal sources. This is a mechanism-based
claim audit, not an exhaustive literature review. The pre-rewrite bibliography
and its keys are preserved in `archive/20261014-pre-journal/references.bib`.
The available bibliography, the paper wiki's research inventory, and locally
identified literature were examined. No callable Zotero connector or accessible
project Zotero export was available; collection-wide coverage is unknown.
Generated experiment PDFs are result exhibits, not downloaded research papers.

| Citation key | Supported use and primary source |
|---|---|
| shazeer2017moe | Introduction: a sparsely gated expert layer increases parameter capacity while computing a subset per input. Author paper, introduction and layer definition: https://arxiv.org/pdf/1701.06538; accepted ICLR 2017 edition confirmed by https://dblp.org/rec/conf/iclr/ShazeerMMDLHD17. |
| fedus2022switch | Introduction: sparse activation separates parameter scale from per-input computation. Published abstract and routing description: https://www.jmlr.org/papers/v23/21-0998.html. |
| li2025schoenfeld | Seven episode labels, released sentence/paragraph annotations and SAT responses. Published abstract and annotation framework: https://aclanthology.org/2025.emnlp-main.922/. This supports the vocabulary, not validity on every native model population. |
| hewitt-manning-2019-structural | Retained prior bibliography reference; the revised Related Work uses reasoning-specific studies instead of this generic syntactic-probe comparison. The original supported proposition was recovery of syntactic structure through learned representational transformations. Published abstract: https://aclanthology.org/N19-1419/. No cognitive-module conclusion is attributed to the probe. |
| sun2026trajectories | Related Work: step-specific geometry and late-stage correctness signals. Published abstract: https://aclanthology.org/2026.acl-long.1237/. This representation-space study is distinguished from expert identities and native text. |
| fayyaz2025steermoe | Related Work: contrastive expert activation identifies behavior-associated experts, followed by inference-time activation or suppression. Published abstract: https://proceedings.iclr.cc/paper_files/paper/2026/hash/0d61c5f5ef91e7e8a091b7b8f72b853c-Abstract-Conference.html. |
| yoon2026misrouted | Related Work: fixed-model comparison of native routing with equal-compute alternatives. Author abstract: https://arxiv.org/abs/2605.07260. DBLP lists the preprint at https://dblp.org/rec/journals/corr/abs-2605-07260; no published edition was verified. Its next-token utility endpoint is not our episode-coverage endpoint. |
| chen2025eac | Related Work: expert-selection-aware quantization and pruning by expert-selection frequency. Published abstract explicitly defines QESC and PESF: https://aclanthology.org/2025.acl-long.633/. |
| lasby2026reap | Related Work: gate values and expert activation/output norms enter REAP. Method section 4: https://arxiv.org/html/2510.13999. Published ICLR 2026 citation is supplied by the authors at https://github.com/CerebrasResearch/reap#citation, with https://openreview.net/forum?id=ukGxWd2aDG. The preprint-only record was upgraded; the citation key and complete authors were retained. OpenReview browser verification prevented direct forum inspection. |
| liu2025spiral | Benchmark-suite lineage and repeat counts. Published paper Appendix D, mathematical benchmarks: https://proceedings.iclr.cc/paper_files/paper/2026/file/3ba7c8c36b71b2fc31fae3361aefefbf-Paper-Conference.pdf. It specifies AVG@32 for AIME24/25 and AMC23 and PASS@1 for the other mathematical benchmarks. No self-play effectiveness claim is imported into this paper. |
| lightman2024verify | Setup: the retained MATH-500 subset is associated with process-supervision evaluation. Published paper and evaluation subset: https://proceedings.iclr.cc/paper_files/paper/2024/hash/aca97732e30bcf1303bc22ac3924fd16-Abstract-Conference.html. |
| he2024olympiadbench | Setup: source of retained OlympiadBench mathematical problems. Official record and benchmark description: https://aclanthology.org/2024.acl-long.211/. Our 675-question subset is a study selection, not the full bilingual multimodal benchmark. |
| lewkowycz2022minerva | Setup: quantitative-reasoning problem source. Published paper: https://proceedings.neurips.cc/paper_files/paper/2022/hash/18abbeef8cfe9203fdf9053c9c4fe191-Abstract.html. The 272-question retained set belongs to our evaluation contract. |
| agrawal2026gepa | Setup: reflective prompt optimization during annotation-instrument development. Published abstract: https://proceedings.iclr.cc/paper_files/paper/2026/hash/0e9e708b6f48e14fd0ac29e167413f76-Abstract-Conference.html. Its optimization results do not establish our annotation accuracy. |
| benjamini1995fdr | Appendix F: Benjamini--Hochberg adjustment of exploratory correctness associations. Existing unchanged record: https://doi.org/10.1111/j.2517-6161.1995.tb02031.x. |

All newly added and substantively discussed records were checked for title,
complete ordered author lists, venue, year and persistent identifiers against
their official records. The GEPA and SteerMoE author lists follow the published
PDF title pages: these differ in name expansions from the proceedings landing
pages. No citation key was renamed, no unrelated bibliography record was
deleted, and no research source is quoted in manuscript prose. Acronyms remain
protected in BibTeX. Parenthetical citations use non-breaking separators and
the class's author-year style.

The related-work selection preserves direct counterevidence to simple frequency
arguments: output-aware REAP, counterfactual route utility, and actual expert
steering. Those results motivate the distinctions in the paper; they are not
used as evidence that our observed coverage gains improve answer accuracy.


## Literature reconstruction after author feedback

Related Work was rebuilt into five connected mechanism families: inference
budgets; semantic episodes and representation-based states; activation steering
and local geometry; expert routing and online adaptation; and compression with
measured efficiency endpoints. Its text is approximately 880 words, excluding
citation commands. The existing bibliography and the supplied private paper's
Section 2 and Appendix A were inspected for discovery leads. The private paper
is not cited, quoted, or used as support for a scientific claim. Every adopted
lead below was checked against a public primary record. None of that private
bibliography's inconsistent venue dates were copied.

The rewrite appends 13 keys without changing any existing bibliography bytes.
The pre-append bibliography SHA-256 is
`83a730bf4e3120e92224e6b66af754af5c33c0da7d2ea558389e5d57999ffc04`.
Existing keys remain available even where a short, generic discussion was
replaced by a more directly relevant mechanism comparison. No collection-wide
Zotero review or exhaustive search is claimed.

| New citation key | Supported proposition and inspected primary record |
|---|---|
| chen2025overthinking | Redundant long reasoning is studied through outcome and process efficiency; self-training favors more concise solutions. The official ICML 2025 abstract and BibTeX print the complete 14-author order, PMLR volume 267 and pages 9487–9499: https://proceedings.mlr.press/v267/chen25bx.html. No effect size from that study is imported. |
| muennighoff2025s1 | Budget forcing terminates or extends thinking when the model attempts to finish. The official EMNLP 2025 abstract, metadata and ten-author BibTeX support this mechanism: https://aclanthology.org/2025.emnlp-main.1025/. Its training and inference intervention are distinguished from our expert-identity control. |
| chen2025seal | Execution, reflection and transition thought categories; an offline steering vector followed by inference-time representation calibration. Published COLM 2025 title-page author order is Runjin Chen, Zhenyu Zhang, Junyuan Hong, Souvik Kundu, Zhangyang Wang: https://openreview.net/pdf?id=klPszYDIRT. The author-uploaded public v3 gives accessible method Sections 4.1–4.2: https://arxiv.org/html/2504.07986v3. OpenReview's direct browser challenge prevented a fresh full download; its indexed primary title page and the public author version agree. |
| eisenstadt2025overclocking | The authors manipulate an encoding of progress through the thinking phase during inference. Their public abstract and three-author order are verified at https://arxiv.org/abs/2506.07240; the method is accessible at https://arxiv.org/html/2506.07240v1. The authors' project citation also retains preprint status: https://royeisen.github.io/OverclockingLLMReasoning-paper/. No formal published edition was verified, so the entry remains an arXiv preprint. |
| rimsky2024caa | Contrastive Activation Addition derives a mean activation difference from paired positive and negative examples and adds it during inference. The official ACL 2024 abstract and complete six-author entry are at https://aclanthology.org/2024.acl-long.828/. |
| venhoff2025steering | Reasoning steering can modulate backtracking, uncertainty and example testing. Public v4 PDF pages 1–4 print the workshop publication header and current author order: https://arxiv.org/pdf/2506.18167. Constantin Venhoff precedes Iván Arcuschin, followed by Philip Torr, Arthur Conmy and Neel Nanda; the first two have equal-contribution markers. The citation uses the inspected v4 and the verified ICLR 2025 reasoning/planning workshop status, rather than mixing author orders across versions. |
| huang2025manifold | Projecting an overthinking direction onto a low-dimensional activation subspace addresses interference; this is an activation intervention, not expert-identity selection. The official NeurIPS 2025 abstract and six authors are at https://papers.nips.cc/paper_files/paper/2025/hash/946827aadb9f7dbdb27f4a0c77dd1859-Abstract-Conference.html; PDF Sections 3–4 support the mechanism. Publisher DOI 10.52202/085713-3430 is printed by that proceedings record. |
| minegishi2025topology | Hidden-state clusters define reasoning graphs whose cycles, diameter and connectivity are analyzed. The official NeurIPS 2025 abstract, five-author order, and publisher DOI are at https://proceedings.neurips.cc/paper_files/paper/2025/hash/147521db4d049c75cf18d9ff96bf68f5-Abstract-Conference.html. The published edition is also indexed at https://dblp.org/rec/conf/nips/MinegishiFKIM25. Our semantic episode vocabulary is not equated with those learned graph nodes. |
| yu2025stateaware | Spectral step representations are clustered into latent states and their transitions modeled with a first-order Markov chain. The official Findings EMNLP 2025 abstract, complete nine-author entry and corrected v2 are at https://aclanthology.org/2025.findings-emnlp.904/. The DOI and pages follow that published record. |
| tan2024reliability | Steering effects vary across inputs and some concepts fail under reasonable prompt changes. The official NeurIPS 2024 abstract, seven authors and publisher DOI are at https://proceedings.neurips.cc/paper_files/paper/2024/hash/fb3ad59a84799bfb8d700e56d19c231b-Abstract-Conference.html. The published PDF prints Daniel Tan, followed by Chanin, Lynch, Paige, Kanoulas, Garriga-Alonso and Kirk; its order differs from the earlier preprint. DBLP confirms the published record at https://dblp.org/rec/conf/nips/TanCLPKGK24. |
| shafran2026regions | Mixture-of-factor-analyzer regions combine centroids and local covariance, with evaluation on localization and steering. The official ICML 2026 abstract and complete six-author BibTeX print Or David Shafran first, followed by Ronen, Fahn, Ravfogel, Geiger and Geva: https://proceedings.mlr.press/v306/shafran26a.html. This published record replaces the private bibliography's preprint-only lead; no region-based reasoning result from the private paper is attributed to it. |
| su2026rewiring | Lightweight additive router-logit vectors adapt online using self-supervision during prefill and later intervals, followed by ordinary generation with the modified router. The official ICML 2026 abstract and complete six-author BibTeX are at https://proceedings.mlr.press/v306/su26o.html. This is a nearest public routing mechanism comparison; its online optimization objective is distinguished from a specified next reasoning action. |
| hassani2026thesismoe | Projection and learned per-token gating conditionally edit activations within a frozen-weight MoE model. The author-uploaded abstract and four-author record are verified at https://arxiv.org/abs/2608.15687; methods are accessible at https://arxiv.org/html/2608.15687v1. This controls expert computations rather than merely replacing selected identities. No published edition was verified, so the entry retains preprint status. |

Published editions are preferred wherever verified. arXiv DOI values remain
arXiv identifiers, not publisher DOIs. The added records preserve acronym
capitalization and complete ordered author lists, including accents. No quoted
source-paper wording appears in manuscript prose. The primary-record table
supports the claims without using secondary summaries as evidence.

The synthesis retains counterevidence and the endpoint distinctions that matter
for this paper: prompt-dependent steering, multidimensional/local structure,
behavior inside expert computations, counterfactual routing utility, and
output-aware pruning. An informative routing readout or changed expert set is
not treated as proof of semantic control, improved correctness, shorter traces,
or lower deployed memory. The existing bibliography comparator completed offline against independent
primary records after the citation edits, with all 27 active entries passing.
No bulk API lookup or automatic normalization is claimed.
