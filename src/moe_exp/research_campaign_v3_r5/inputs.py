"""Exact source resolution; corpus identity is checked before any new dispatch."""
from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

from .contracts import (BENCHMARKS, CLASSES, MODELS, REPO, SEED, STAGES, digest,
                        file_sha, load, protocol, publish, require, source_binding)
from .design import cohorts, prefix_roster

BASE = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24")
ANALYSIS = BASE / "v3_analysis/results-r2"
DYNAMICS = BASE / "dynamics-routing/results"
OUTCOMES = BASE / "campaign-v3/outcomes/outcomes_v3-99bf8d59506e2893.parquet"
PROGRAM = REPO / "results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json"
FAMILY = REPO / "report/experimental-resume-v1/family-freeze.json"


def catalogue_refs():
    rows=[]
    def add(artifact,path,scope):
        path=Path(path)
        present=path.is_file()
        rows.append({"artifact":artifact,"path":str(path),"status":"reusable" if present else "compatibility_unresolved",
            "original_scope":scope,"bytes":path.stat().st_size if present else None,
            "sha256_file":file_sha(path) if present and path.stat().st_size<64*1024**2 else None,
            "regeneration_authorized":False})
    for name in ("benchmark_cells.csv","cohort_populations.csv","EXPERT_ASSOCIATION_METHOD-r1.json",
        "CHECKPOINT_HEADER_INVENTORY.json","labelled_expert_coverage.csv"):
        add(name,REPO/"report/model-analysis-review-20261008"/name,"original seven-model corpus/profile scope")
    for name in ("selection_coverage.csv","mask_overlap_stability.csv","class_layer_coverage.csv"):
        add(name,REPO/"report/expert-memory-study-v1/exhibits-r2"/name,"original five-fold 25% study weighting and mask scope")
    for name in ("TRANSITIONS.json","SAVED_DATA_CLOSEOUT-r1.json","FROZEN_MASKS.json"):
        add(name,REPO/"report/expert-memory-study-v1"/name,"original common-population analysis")
    for name in ("NATIVE_BOUNDARY_PROFILES.json","dense-recovery-v3/TRAJECTORY_RESULT.json"):
        add(name,REPO/"report/routing-study-v1"/name,"original Qwen3.6 boundary/dense recovery")
    for model in MODELS:
        for sub in ("results/part1","results/s3/nmi"):
            add("depth-paths:"+model,BASE/"depth-paths"/sub/(model+".json"),"original dominant-expert representation and weighting")
        for name in ("E_expert_enrichment.csv.gz","E_class_experts.csv"):
            add("original-association:"+model,ANALYSIS/model/"A"/name,"original all-layer association weighting")
    add("combined-depth-readout",BASE/"depth-paths/results/part2/results.json","original all-seven depth/readout comparison")
    add("clean-readout",BASE/"forum/tests/r3_context/readout-results.v3.json","original held-out present-class readouts and retained losses")
    add("GPT-B1",BASE/"steering-v1/runs/resume-v1/r3e-b1-clean-v2/result.json","clean GPT coordination; historical fitted model not extended")
    return rows


def read_source(location):
    if isinstance(location, str):
        location = json.loads(location)
    with Path(location["path"]).open("rb") as f:
        f.seek(location["byte_offset"])
        raw = f.read(location["line_bytes"])
    require(hashlib.sha256(raw).hexdigest() == location["line_sha256"], "saved trace line changed")
    return json.loads(raw)


def prepare(root, resume_from=None):
    root = Path(root).resolve()
    require(not (root / "CAMPAIGN.json").exists(), "use a fresh campaign directory")
    if resume_from is not None:
        from .contracts import check_files, private_path
        parent=load(private_path(resume_from)/"CAMPAIGN.json")
        check_files(parent["code_files"])
        check_files(parent["source_files"])
        from .resume import ancestry
        ancestors=ancestry({'resume_from':str(private_path(resume_from))})
        from .storage import initialize
        storage_policy = initialize(root,parent)
        p=publish(root/"PROTOCOL.json",protocol())
        graph=publish(root/"GRAPH.json",dependency_graph(parent))
        from .contracts import digest
        return publish(root/"CAMPAIGN.json",{**{k:v for k,v in parent.items() if k!="sha256"},
            "schema":"research-campaign-binding-v3","campaign_id":root.name,"root":str(root),
            "resume_from":str(private_path(resume_from)),"parent_campaign_sha256":parent["sha256"],
            "ancestor_campaigns":[{'root':a['root'],'sha256':a['sha256']} for a in ancestors],
            "protocol_sha256":p["sha256"],"graph_sha256":graph["sha256"],"code_files":source_binding(),
            "analysis_binding":digest({"numeric_normalization":3,"compact_extraction":3,"full_A_mapping":2,"exact_attempt_accuracy":2,"split_panel_alpha":[.025,.025]}),
            "reap_binding":digest({'native_pre_reduction':4,'within_execution_arrays':True,'unchanged_tolerance':[.02,.02]}),
            "compact_extraction":True,
            "storage_policy":storage_policy,"balanced_sensitivity_overlay":True,"results_first_phases":True,"publish_results":True,
            "instrumentation_binding":digest({"native_signature_mask_wrapper":2,"failure_preservation":2}),
            "continuation_revision":5,"independent_memory_preparation":True,"independent_publication":True,
            "deterministic_recovery_attempts":1,"stochastic_redraws":0,
            "authorization":"User: Implement the final continuation plan; Implement the plan"})
    old = load(REPO / "report/expert-memory-study-v1/CAMPAIGN.json")
    renderer=json.loads((REPO/".research-runtime/RENDERER.json").read_text())
    from moe_exp.correlation_pipeline.model_profiles import CARD_PROFILES
    from moe_exp.verification_study.registry import model_contract
    models, files, saved = {}, {}, {}
    for m in MODELS:
        original = model_contract(m)
        spec = json.loads((ANALYSIS / m / "A/provenance.json").read_text())["spec"]
        card = CARD_PROFILES[m]
        native = old["models"][m]
        generation_file=Path(original["local_path"])/"generation_config.json"
        if generation_file.exists():
            files[str(generation_file)]=file_sha(generation_file)
        require(native["groups"] == 1, "multi-group unequal allocation requires an explicit native contract")
        models[m] = {**original, "card": card, "layers": spec["layers"],
                     "native_layer_limit": native["native_layer_limit"],
                     "expert_bytes": native["serialized_native_expert_bytes"] /
                                     (len(spec["layers"]) * original["experts"]),
                     "kv_cache_memory_bytes": native["kv_cache_memory_bytes"],
                     "runtime_shape": {"gpus": 2, "cpus": 16, "memory_gib": 120}}
        models[m]["profile_sha256"] = digest(models[m])
        saved[m] = {"attempts": str(ANALYSIS / m / "A/attempts.parquet"),
                    "counts": str(ANALYSIS / m / "A/class_counts.npz"),
                    "sentences": str(DYNAMICS / m / "A/sentences.parquet"),
                    "histogram": str(DYNAMICS / m / "A/hist.npz"),
                    "features": str(ANALYSIS / m / "A/features.parquet"),
                    "provenance": str(ANALYSIS / m / "A/provenance.json"),
                    "selection": str(REPO / f"report/expert-memory-study-v1/{m}/SELECTION.json")}
        for name in ("provenance", "selection"):
            files[saved[m][name]] = file_sha(saved[m][name])
    label_binding = BASE / "campaign-v3/labels/gpt/part-00/checkpoints/batch-000000.json"
    config = json.loads(label_binding.read_text())["binding"]["config"]
    require(file_sha(PROGRAM) == config["program_sha256"], "original seven-class program changed")
    hub = Path(models["gpt"]["local_path"]).parents[2]
    judge_candidates = sorted((hub / "models--Qwen--Qwen3.8-27B/snapshots").glob("*/config.json"))
    require(len(judge_candidates) == 1, "original reader revision is ambiguous or unavailable")
    judge = {"path": str(judge_candidates[0].parent), "revision": judge_candidates[0].parent.name,
             "program": str(PROGRAM), "config": config,
             "native_context":49152,"batch_size":64,"parser":"qwen3",
             "original_engine_source":str(BASE/"campaign-v3/code/t1-d2428b83b1b519df/sbatch/label_job.sbatch"),
             "engine_changes":["bounded in-process execution","no speculation","synchronous scheduling"],
             "decoding": {"temperature": config["temperature"], "top_p": config["top_p"],
                          "top_k": config["top_k"], "min_p": config["min_p"],
                          "presence_penalty": config["presence_penalty"], "repetition_penalty": config["repetition_penalty"]}}
    for p in (FAMILY, PROGRAM, label_binding, judge_candidates[0],Path(judge["original_engine_source"])):
        files[str(p)] = file_sha(p)
    p = publish(root / "PROTOCOL.json", protocol())
    reuse=publish(root/"REUSED_CATALOGUE.json",{"records":catalogue_refs(),"original_scopes_preserved":True,
        "missing_reuse_files_do_not_authorize_regeneration":True})
    nodes = [{"stage": s, "model": m, "dependencies": list(deps)} for s, deps in STAGES.items()
             for m in (MODELS if s in ("mapping","derive", "native","supplementary","coordination","measure", "freeze", "qualify", "price", "execute", "contexts","annotate", "score") else ("all",))]
    graph = publish(root / "GRAPH.json", {"nodes": nodes, "finite": True,
                    "production_expansion": "frozen task rows only; no outcome-dependent enrollment"})
    return publish(root / "CAMPAIGN.json", {"schema": "research-campaign-binding-v2",
        "campaign_id": root.name, "root": str(root), "protocol_sha256": p["sha256"],
        "graph_sha256": graph["sha256"], "models": models, "saved_inputs": saved,
        "outcomes": str(OUTCOMES), "family_path": str(FAMILY), "annotator": judge,"renderer":renderer,
        "code_files": source_binding(), "source_files": files, "runtime_files": old["runtime_files"],
        "preserved_scientific_files": old["scientific_files_preserved"],
        "preserved_operational_files": old["readiness_files_preserved"],
        "authorization_local_hours": 42050., "authorization_baseline_consumed": 25950.,
        "reuse_catalogue": str(REPO / "paper/wiki/NEXT_EXPERIMENTS.md"),
        "reused_catalogue_sha256":reuse["sha256"],
        "original_stop_preserved": str(REPO / "report/experiment-stop-20261008-r1/STOP_RECEIPT.json")})


def dependency_graph(parent):
    """Seal the entire expansion rule; future shards only partition these rows."""
    nodes=[{'stage':s,'model':m,'dependencies':list(deps),'kind':'receipt' if s=='local_paper' else 'work'} for s,deps in STAGES.items()
        for m in (MODELS if s not in ('validate','publish','local_paper','resolve','admit','reader_price','snapshot','aggregate','closeout','phase_export') else ('all',))]
    rows={}
    for model in MODELS:
        layers=parent['models'][model]['layers']
        pairs=[{'layer_a':layers[i],'layer_b':layers[i+d],'depth_gap':d,'shard_index':n+1}
            for n,(i,d) in enumerate((i,d) for d in (1,2,4,8) for i in range(len(layers)-d))]
        rows[model]={'coordination_pairs':pairs,'saved_measurement_assignments':1547,
            'calibration_folds':5,'mask_cells':5*5*2*3,
            'causal_configuration_cells':5*7*3*2*9*2,
            'pruning_execution_cells_upper':420*(1+5*2*3),
            'causal_execution_cells_upper':129*(1+7*3*2*2*2),
            'analysis_windows_per_execution':3,'new_natural_draws_for_calibration':0,
            'shards':'partition immutable original rows using retained measured prices'}
    return {'nodes':nodes,'registered_rows':rows,'finite':True,'staged_first_model':'qwen330b',
        'delivery_phases':['memory','central','sensitivity'],
        'phase_dependencies':'memory scoring/export independent of causal; central precedes sensitivity',
        'global_analysis_gate':'required repaired summaries and shared-failure checks; coordination proceeds independently',
        'measurement_components':['router_bank','qualified_native_pre_reduction_reap_norms'],
        'pruning_roster':{'questions':420,'families':400},'causal_roster':{'questions':131,'families':129},
        'production_expansion':'frozen task rows only; no outcome-dependent enrollment',
        'qualification_expansion':'one original full-context fixture per distinct physical layer/count vector',
        'role_price_expansion':'one untouched original assignment per distinct role/shape and original annotator batch',
        'scoring_and_annotation':'deterministic scoring has one atomic recovery; started stochastic assignments non-redrawable',
        'recovery':'at most one verified infrastructure replacement per registered shard',
        'owners':['held primary','first failure owner','terminal failure owner','October 15 snapshot','terminal export'],
        'cache_lifetime':'serial models, verified durable dependent outputs before v3-only retirement',
        'handoff_bound':'2 * (registered scientific shards + aggregations + fragments + snapshot/export) + 8',
        'resampling':'10000 deterministic paired whole-family draws; saved sufficient statistics; undefined draws retained'}


def resolve(campaign):
    import pandas as pd
    from moe_exp.expert_memory_study.selection import family_folds
    root = Path(campaign["root"])
    parent = load(campaign["family_path"])
    qf = {q: f for f, qs in parent["new_parent_pools"]["families"].items() for q in qs}
    attempts = {m: pd.read_parquet(s["attempts"]) for m, s in campaign["saved_inputs"].items()}
    questions = set(attempts[MODELS[0]].question)
    require(len(questions) == 1547 and all(set(a.question) == questions for a in attempts.values()), "seven-model A population drift")
    require(questions <= set(qf), "question absent from duplicate graph")
    folds = family_folds({qf[q] for q in questions})
    require(len(folds) == 1486, "fold universe differs")
    for m in MODELS:
        require(load(campaign["saved_inputs"][m]["selection"])["folds_sha256"] == digest(folds), "exact saved fold map mismatch")
    fold_receipt = publish(root / "FOLDS.json", {"folds": folds, "source": campaign["family_path"],
        "folds_sha256": digest(folds), "question_families": {q: qf[q] for q in sorted(questions)}})
    sentences = {m: pd.read_parquet(s["sentences"]) for m, s in campaign["saved_inputs"].items()}
    common = set.intersection(*(set(f.question) for f in sentences.values()))
    require(len(common) == 522 and len({qf[q] for q in common}) == 509, "common sentence population differs")
    rows = [{"benchmark": r.dataset, "family": qf[r.question], "question": r.question}
            for r in attempts[MODELS[0]].itertuples()]
    pruning = cohorts(rows, folds, 100)
    causal = cohorts([r for r in rows if r["question"] in common], folds, 30)
    publish(root / "COHORTS.json", {"pruning": pruning, "causal": causal, "folds_sha256": fold_receipt["folds_sha256"],
        "seed": SEED, "correctness_used": False, "classes_used": False, "outcomes_used": False})
    ledgers, prefixes = [], {}
    scores = pd.read_parquet(campaign["outcomes"])
    scores = scores[scores.corpus == "card-v3"]
    for m in MODELS:
        a = attempts[m]
        require(a.attempt_id.is_unique and a.question.is_unique, "A attempt identity duplication")
        a["family"] = a.question.map(qf)
        a["benchmark"] = a.dataset
        # Score lookup is permitted for reuse only, never used in cohort selection.
        s = scores[scores.model == m]
        joined = a.merge(s[["dataset", "problem_id", "sample_id", "trace_sha256", "strict_correct"]],
                         on=["dataset", "problem_id", "sample_id"], suffixes=("", "_score"), validate="one_to_one")
        require(len(joined) == len(a) and (joined.trace_sha256 == joined.trace_sha256_score).all(), "native strict score binding differs")
        table = []
        layouts={}
        allowances={}
        for r in a.to_dict("records"):
            location = json.loads(r["source_location"])
            # Explicitly bind card-v3 rather than matching solely on question IDs.
            require("card-v3" in location["path"] or "campaign-v3" in location["path"], "non-card source population")
            trace=read_source(location)
            generation=trace["metadata"]["generation_config"]
            native=campaign["models"][m]
            require(generation["revision"]==native["revision"] and generation["sampler"]==native["card"]["sampler"] and
                generation["max_model_len"]==native["card"]["max_model_len"] and generation["max_tokens"]==native["card"]["max_tokens"],
                "original sampling or pinned profile differs")
            from moe_exp.schemas import TraceRecord
            from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
            from moe_exp.correlation_pipeline.spans import trace_digest
            record=TraceRecord(**trace)
            require(trace_digest(record)==r["trace_sha256"],"original trace digest mismatch")
            if r["question"] in common:
                layout=saved_layout(record)
                layouts[r["attempt_id"]]=[max(owned)+1 if owned else None for owned in layout["unit_tokens"]]
                allowances[r["attempt_id"]]=min(native["card"]["max_model_len"]-len(trace["metadata"]["token_replay"]["prompt_token_ids"]),
                    native["card"]["max_tokens"] if native["card"]["max_tokens"] is not None else native["card"]["max_model_len"])
            table.append({k: (int(r[k]) if k in ("sample_id", "completion_tokens") else r[k]) for k in (
                "attempt_id", "question", "benchmark", "family", "source_location", "trace_sha256",
                "completion_tokens", "sample_id", "problem_id", "source_problem_id", "tensor_path", "packet_path")})
            table[-1]["source_location"]=json.dumps({k:location[k] for k in ("path","byte_offset","line_bytes","line_sha256")},sort_keys=True)
            from moe_exp.correlation_pipeline.scoring import extract_answer
            final=trace["metadata"].get("assistant_content") or trace["metadata"].get("reasoning_content") or ""
            table[-1].update(finish_reason=trace["metadata"]["finish_reason"],answer_present=bool(extract_answer(final,"math")))
        table.sort(key=lambda r: (r["benchmark"], r["family"], r["question"]))
        publish(root / m / "SOURCES.json", {"records": table, "profile_sha256": campaign["models"][m]["profile_sha256"]})
        sr = []
        for r in sentences[m].to_dict("records"):
            if r["question"] not in common:
                continue
            sr.append({"attempt_id": r["attempt_id"], "question": r["question"], "benchmark": r["dataset"],
                "family": qf[r["question"]], "sentence_index": int(r["sentence_index"]),
                "prefix_tokens": layouts[r["attempt_id"]][int(r["sentence_index"])], "progress": float(r["rel_pos"]),
                "n_tokens": int(r["n_tokens"]), "segment": int(r["segment"])})
            sr[-1]["source_class"]=CLASSES[int(r["cls"])]
            end=sr[-1]["prefix_tokens"]
            sr[-1]["compatible"]=end is not None and end<allowances[r["attempt_id"]]
        prefixes[m] = prefix_roster(sr, causal, folds, m)
        publish(root / m / "PREFIXES.json", {"records": prefixes[m], "folds_sha256": digest(folds)})
        for artifact, status, evidence, missing in (
            ("native_accuracy", "reusable", campaign["outcomes"], None),
            ("balanced_associations", "derivable", campaign["saved_inputs"][m]["counts"], "new benchmark weighting and inclusion contrasts"),
            ("native_dynamics", "derivable", campaign["saved_inputs"][m]["sentences"], "remaining balanced summaries; annotation gaps retained"),
            ("coordination", "derivable", campaign["saved_inputs"][m]["attempts"], "full-top-k joints from referenced raw archives"),
            ("hidden_summaries", "reusable", campaign["saved_inputs"][m]["features"], None),
            ("reap", "genuinely_unmeasured", campaign["saved_inputs"][m]["provenance"], "per-expert output norms absent from native producer schema"),
            ("dose_calibration", "genuinely_unmeasured", str(REPO / "src/moe_exp/verification_study/adapters.py"), "native score bank was transient; selected IDs cannot reconstruct scores"),
            ("causal", "genuinely_unmeasured", str(REPO / "paper/wiki/EXPERIMENT_INVENTORY.md"), "new signed-set/depth/configuration assignments have no completed compatible records"),
            ("resident_accuracy", "genuinely_unmeasured", str(REPO / "report/expert-memory-study-v1/FROZEN_MASKS.json"), "changed masks/native profiles and physical savings have no completed qualification")):
            ledgers.append({"model": m, "artifact": artifact, "status": status, "source": evidence, "required_missing": missing})
    return publish(root / "REUSE.json", {"records": ledgers, "corpus": "card-v3", "questions": 1547,
        "families": 1486, "common_questions": 522, "common_families": 509,
        "folds_sha256": digest(folds), "profiles_and_sentence_alignment_resolved":True,
        "native_archive_compatibility":"checked by saved-data and native-reuse workers before scheduling replacement",
        "compatibility_resolved_before_replacement": True})


def counts_for(campaign, model, rows):
    import pandas as pd
    from moe_exp.expert_memory_study.selection import validate_counts
    path = campaign["saved_inputs"][model]["counts"]
    with np.load(path, allow_pickle=False) as archive:
        ids, counts, tokens = (archive[k] for k in ("attempt_id", "counts", "tokens"))
        require(list(archive["classes"]) == list(CLASSES) + ["ALL_REASONING", "ALL_LABELLED"], "class order mismatch")
    index = pd.Index(ids).get_indexer([r["attempt_id"] for r in rows])
    require((index >= 0).all() and len(index) == len(ids), "archive row join failed")
    counts, tokens = counts[index], tokens[index]
    validate_counts(counts, tokens, campaign["models"][model]["native_k"])
    return counts, tokens
