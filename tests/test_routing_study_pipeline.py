"""End-to-end fixture comparison and lifecycle rejection checks; no model loading."""
import copy
from types import SimpleNamespace
import pytest
from moe_exp.routing_study.common import BENCHMARKS,CHECKPOINT,ENGINE,digest,seal,write_json,file_hash,rows
from moe_exp.routing_study.policy import freeze
from moe_exp.routing_study.analysis import compare,register,freeze_frontier
from moe_exp.routing_study.execution import append_record
from moe_exp.routing_study.vllm_backend import StudyWorkerExtension,verify_reports,make_frame
from moe_exp.routing_study.runtime import RequestHistory


def fixtures(tmp_path):
    policies=[freeze("native",code={}),freeze("temperature",2,code={})]
    planned=[]
    groups={}
    for b in BENCHMARKS:
        for question in range(2):
            key='["'+b+'","'+str(question)+'"]'
            groups[key]=[key]
            for sample in range(4):
                planned.append(dict(id=digest([key,sample]),problem_key=key,sample_id=sample,
                    dataset=b,prompt_token_ids=[1,2],gold_answer="3",
                    sampling={"max_tokens":10,"seed":sample}))
    study=seal(dict(split={"groups":groups},attempts={"tuning":planned}))
    registration=register(study,policies,[[p["sha256"] for p in policies]],"tuning")
    paths=[]
    for condition,policy in enumerate(policies):
        path=tmp_path/str(condition)
        path.mkdir()
        for row in planned:
            question=int(row["problem_key"][-3])
            correct=question==1
            # Two opposite paired transitions. A capped, missing-answer attempt remains incorrect.
            if condition and row["sample_id"]==0: correct=not correct
            capped=row["sample_id"]==1
            if capped: correct=False
            record=dict(id=row["id"],input_sha256=digest(row),policy_sha256=policy["sha256"],
                problem_key=row["problem_key"],dataset=row["dataset"],prompt_token_ids=row["prompt_token_ids"],
                sampling_args=row["sampling"],generated_token_ids=[4]*(10 if capped else 4+condition),
                generated_token_count=10 if capped else 4+condition,is_correct=correct,
                model_answer=None if capped else "3",finish_reason="length" if capped else "stop",
                latency_seconds=1.,infrastructure_failure=False,hook_seconds_cumulative=.1,
                dispatch={str(l):{"diagnostics":{"tokens":10,"reasoning_tokens":8,"dispatch_calls":10,
                    "active":condition*8,"changed_sets":0,"changed_weights":condition*8},
                    "native_sha256":"same","decisions_sha256":"same"} for l in range(40)})
            append_record(path/"attempts.jsonl",record)
        write_json(path/"manifest.json",dict(status="complete",dispatch_verified=True,
            study_sha256=study["sha256"],phase="tuning",policy=policy,expected_count=len(planned),registration_sha256=registration["sha256"],
            attempts_sha256=file_hash(path/"attempts.jsonl"),engine=ENGINE,checkpoint=CHECKPOINT,
            planned_sha256=digest(planned),backend={"version":"mock"},code={},token_scope="generated_reasoning",
            scoring_contract="fixture",failures_sha256=None))
        paths.append(path)
    return study,policies,registration,paths


def test_complete_paired_pipeline_retains_caps_missing_answers(tmp_path):
    study,policies,registration,paths=fixtures(tmp_path)
    report=compare(paths,study,registration)
    base=report["estimates"]["equal_benchmark_mean"][policies[0]["sha256"]]
    assert base["accuracy"]["estimate"]==pytest.approx(3/8)
    assert base["truncation"]["estimate"]==pytest.approx(1/4)
    guided=report["estimates"]["equal_benchmark_mean"][policies[1]["sha256"]]
    assert guided["wrong_to_right"]["estimate"]==pytest.approx(1/8)
    assert guided["right_to_wrong"]["estimate"]==pytest.approx(1/8)
    assert report["replicates"]==1000
    static=freeze_frontier(report,study,policies,stage="static")
    selection=freeze_frontier(report,study,policies,stage="final",static_selection=static)
    assert selection["selected_policy_hashes"]==[]  # Native dominates this synthetic action.
    report.pop("sha256")
    report["phase"]="test"
    with pytest.raises(ValueError,match="tuning"):
        freeze_frontier(seal(report),study,policies,stage="final")


def test_missing_coverage_duplicate_and_unknown_failure_cost_rejected(tmp_path):
    study,policies,registration,paths=fixtures(tmp_path)
    path=paths[1]/"attempts.jsonl"
    content=path.read_text()
    path.write_text("".join(content.splitlines(keepends=True)[:-1]))
    with pytest.raises(ValueError,match="coverage"):
        compare(paths,study,registration)
    path.write_text(content+content.splitlines(keepends=True)[0])
    with pytest.raises(ValueError,match="Duplicate"):
        compare(paths,study,registration)


def test_changed_sampling_baseline_rejected(tmp_path):
    study,policies,registration,paths=fixtures(tmp_path)
    path=paths[1]/"manifest.json"
    import json
    manifest=json.loads(path.read_text())
    manifest["engine"]={**manifest["engine"],"max_num_seqs":1}
    write_json(path,manifest)
    with pytest.raises(ValueError,match="Unmatched"):
        compare(paths,study,registration)


def test_interleaved_chunked_prefill_padding_preemption_and_cancel():
    p=freeze("native",code={})
    specs={k:dict(id=k,prompt_token_ids=[0,1]) for k in ("a","b")}
    worker=StudyWorkerExtension()
    worker.study=SimpleNamespace(policy=p,requests={},retired=set(),decode_piece={0:"prompt",1:"<think>",2:"x"}.__getitem__)
    worker.study_register(specs)
    frame=make_frame(worker.study,["a","b"],[0,2,3],[0,1,0,999],[0,1,0,99])
    assert frame[-1] is None
    frame=make_frame(worker.study,["b","a"],[0,1,2],[1,2],[1,2])
    assert frame[1][1]["reasoning"]==0
    # Recomputed prompt starts at absolute zero; no history duplication.
    make_frame(worker.study,["a"],[0,3],[0,1,2],[0,1,2])
    assert worker.study.requests["a"].tokens.reasoning_count==1
    worker.study_release(["a"])
    with pytest.raises(RuntimeError,match="Unregistered"):
        make_frame(worker.study,["a"],[0,1],[2],[3])
    with pytest.raises(ValueError,match="Duplicate"):
        worker.study_register({"a":specs["a"]})


def test_tp_replica_diagnostics_count_only_once_and_mismatch_rejected():
    layers={str(l):dict(diagnostics={"tokens":10,"dispatch_calls":10}) for l in range(40)}
    reports=[dict(rank=rank,layers=copy.deepcopy(layers)) for rank in (0,1)]
    result=verify_reports(reports)
    assert result["layers"]["0"]["diagnostics"]["tokens"]==10
    reports[1]["layers"]["0"]["diagnostics"]["tokens"]=20
    with pytest.raises(RuntimeError,match="replicas"):
        verify_reports(reports)


def test_config_accepts_only_exact_offline_snapshot_and_native_contract():
    import copy, json
    from pathlib import Path
    from types import SimpleNamespace as NS
    from moe_exp.routing_study.common import MODEL, REVISION
    from moe_exp.routing_study import vllm_backend
    snapshot = json.loads(Path(vllm_backend.__file__).with_name("checkpoint_files.json").read_text())["cache"]
    config = NS(use_v2_model_runner=False, kernel_config=NS(moe_backend="marlin"),
        model_config=NS(model=MODEL, revision=REVISION, quantization="fp8", enforce_eager=True,
        hf_text_config=NS(model_type="qwen3_5_moe_text", num_experts=256, num_experts_per_tok=8,
            num_hidden_layers=40, shared_expert_intermediate_size=512)),
        parallel_config=NS(tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1,
            enable_expert_parallel=False, enable_eplb=False, use_sequence_parallel_moe=False),
        cache_config=NS(enable_prefix_caching=False), speculative_config=None,
        scheduler_config=NS(max_num_seqs=8, async_scheduling=False))
    for identity in (MODEL, snapshot):
        config.model_config.model = identity
        vllm_backend.validate_config(config)
    for identity in (snapshot + "-changed", MODEL.replace("3.6", "3.5")):
        config.model_config.model = identity
        with pytest.raises(RuntimeError, match="FP8 contract"):
            vllm_backend.validate_config(config)
    config.model_config.model = snapshot
    config.use_v2_model_runner = True
    with pytest.raises(RuntimeError, match="V1 runner"):
        vllm_backend.validate_config(config)
    config.use_v2_model_runner = False
    for section, field, value in (("kernel_config", "moe_backend", "triton"), ("model_config", "revision", "other"),
            ("model_config", "quantization", "nf4"), ("model_config", "enforce_eager", False),
            ("parallel_config", "tensor_parallel_size", 1), ("parallel_config", "enable_eplb", True),
            ("cache_config", "enable_prefix_caching", True), ("scheduler_config", "max_num_seqs", 9),
            ("scheduler_config", "async_scheduling", True)):
        changed = copy.deepcopy(config)
        setattr(getattr(changed, section), field, value)
        with pytest.raises(RuntimeError):
            vllm_backend.validate_config(changed)


def test_engine_rejects_unpinned_runner_before_importing_vllm(monkeypatch):
    from moe_exp.routing_study.execution import make_engine
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    with pytest.raises(RuntimeError, match="VLLM_USE_V2_MODEL_RUNNER=0"):
        make_engine({})


def test_runner_requires_modular_marlin_and_preserves_shared_object():
    from types import SimpleNamespace as NS
    from moe_exp.routing_study.vllm_backend import validate_runner
    runner = type("MoERunner", (), {})()
    router = type("FusedTopKRouter", (), {})()
    router.scoring_func, router.renormalize, router.eplb_state = "softmax", True, None
    quant = NS(is_monolithic=False, fp8_backend=NS(name="MARLIN"))
    runner.router = router
    runner.routed_experts = NS(quant_method=quant, forward_modular=lambda **kwargs: None)
    shared = object()
    module = NS(experts=runner, shared_expert=shared)
    assert validate_runner(module) is runner and module.shared_expert is shared
    quant.fp8_backend.name = "TRITON"
    with pytest.raises(RuntimeError, match="Unsupported"):
        validate_runner(module)
    quant.fp8_backend.name = "MARLIN"
    quant.is_monolithic = True
    with pytest.raises(RuntimeError, match="Unsupported"):
        validate_runner(module)


def test_wave_request_ids_unique_across_conditions_and_retries():
    from moe_exp.routing_study.execution import wave_specs
    rows = [(0, {"id": "attempt0"}), (1, {"id": "attempt1"})]
    first, second = wave_specs(rows), wave_specs(rows)
    assert not set(first).intersection(second)
    assert list(first.values()) == list(second.values()) == [r for _, r in rows]
    with pytest.raises(ValueError, match="Duplicate"):
        wave_specs(rows + [rows[0]])


def test_installed_request_id_assignment_respects_registered_ids():
    # Execute the small pinned installed assignment method, without importing vLLM
    # or initializing CUDA. This reproduces the exact failure seen on the GPU job.
    import ast
    from pathlib import Path
    from types import SimpleNamespace as NS
    from importlib.metadata import distribution
    root = Path(distribution("vllm").locate_file("vllm"))
    tree = ast.parse((root / "v1/engine/input_processor.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "InputProcessor")
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "assign_request_id")
    function.decorator_list = []
    function.args.args[0].annotation = None
    envs = NS(VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=False)
    namespace = {"envs": envs, "random_uuid": lambda: "afd64386rest",
                 "logger": NS(warning_once=lambda *args: None)}
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, "installed_vllm_assign_request_id", "exec"), namespace)
    assign = namespace["assign_request_id"]
    request = NS(request_id="0", external_req_id=None)
    assign(request)
    assert request.request_id == "0-afd64386" and request.external_req_id == "0"
    envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION = True
    request = NS(request_id="study-unique-wave-0", external_req_id=None)
    assign(request)
    assert request.request_id == request.external_req_id == "study-unique-wave-0"
    with pytest.raises(ValueError):
        assign(request)
