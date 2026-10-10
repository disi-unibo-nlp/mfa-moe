import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "campaign_v3"))


def load_stage(code_root: Path):
    spec = importlib.util.spec_from_file_location(f"stage_{abs(hash(code_root))}",
                                                  code_root / "scripts/campaign_v3/stage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_t1_approval_binds_to_the_frozen_snapshot(tmp_path):
    import freeze_code

    frozen = freeze_code.freeze(ROOT, tmp_path)
    snapshot = Path(frozen["snapshot"])
    assert freeze_code.freeze(ROOT, tmp_path)["reused"] is True
    (tmp_path / "review").mkdir()
    (tmp_path / "review/T1.json").write_text(json.dumps(dict(
        status="approved", code_snapshot=str(snapshot), tree_sha256=frozen["tree_sha256"])))
    frozen_stage, live_stage = load_stage(snapshot), load_stage(ROOT)
    frozen_stage.D = live_stage.D = tmp_path
    assert frozen_stage.reviewed() is True
    assert live_stage.reviewed() is False
    (tmp_path / "review/T1.json").write_text(json.dumps(dict(
        status="approved", code_snapshot=str(snapshot), tree_sha256="0" * 64)))
    assert frozen_stage.reviewed() is False
    for path in snapshot.rglob("*"):
        path.chmod(0o700)
    snapshot.chmod(0o700)


def test_capture_handoff_waits_for_t2_then_runs_the_named_driver(tmp_path, monkeypatch):
    stage = load_stage(ROOT)
    stage.D = tmp_path
    (tmp_path / "acceptance").mkdir()
    (tmp_path / "review").mkdir()
    stage.capture_handoff("glm", "B")
    marker = tmp_path / "acceptance/capture-waiting-glm-B.json"
    assert json.loads(marker.read_text())["status"] == "waiting_for_T2"
    driver = tmp_path / "driver.py"
    driver.write_text("")
    (tmp_path / "review/T2.json").write_text(json.dumps(dict(
        status="approved", driver=str(driver), bundle=str(tmp_path / "bundle"))))
    calls = []
    monkeypatch.setattr(stage.subprocess, "run", lambda argv, check: calls.append(argv))
    stage.release_capture(types.SimpleNamespace(model="glm"))
    assert calls == [[stage.CLIENT_PY, str(driver), "start", "--model", "glm", "--cohort", "B",
                      "--bundle", str(tmp_path / "bundle")]]
    assert not marker.exists()


def _label_fixture(tmp_path, labeled):
    stage = load_stage(ROOT)
    stage.D = tmp_path
    rows = [dict(dataset="aime24", problem_id="p1", stratum="capped",
                 sentence_selection=dict(indices=[0, 2])),
            dict(dataset="aime24", problem_id="p2", stratum="correct",
                 sentence_selection=dict(indices=[1]))]
    (tmp_path / "cohorts/glm").mkdir(parents=True)
    (tmp_path / "acceptance").mkdir()
    (tmp_path / "cohorts/glm/cohorts.json").write_text(json.dumps(dict(rows=rows)))
    (tmp_path / "cohorts/glm/parts.json").write_text(json.dumps(dict(parts=[dict(part=0, total=len(labeled))])))
    part = tmp_path / "labels/glm/part-00"
    part.mkdir(parents=True)
    records = [dict(identity=dict(dataset="aime24", problem_id=p, sentence_index=i), label="Plan")
               for p, i in labeled]
    (part / "annotations.json").write_text(json.dumps(records))
    (part / "summary.json").write_text(json.dumps(dict(status="complete", completed=len(records),
                                                       expected=len(records))))
    stage.capture_handoff = lambda model, cohort: None
    return stage


def test_label_gate_requires_exact_selection_coverage(tmp_path):
    stage = _label_fixture(tmp_path, [("p1", 0), ("p1", 2), ("p2", 1)])
    stage.label_gate(types.SimpleNamespace(model="glm", round=1))
    receipt = json.loads((tmp_path / "acceptance/labels-glm.json").read_text())
    assert receipt["status"] == "complete" and receipt["strata"] == {"capped": 1, "correct": 1}


def test_label_gate_rejects_a_missing_selected_sentence(tmp_path):
    stage = _label_fixture(tmp_path, [("p1", 0), ("p2", 1)])
    with pytest.raises(SystemExit, match="selection coverage"):
        stage.label_gate(types.SimpleNamespace(model="glm", round=1))


def test_label_parts_share_the_campaign_gpu_slot_pool(tmp_path, monkeypatch):
    stage = load_stage(ROOT)
    stage.D = tmp_path
    (tmp_path / "cohorts/glm").mkdir(parents=True)
    parts = [dict(part=i, trace_root=f"/root/{i}", total=10) for i in range(4)]
    (tmp_path / "cohorts/glm/parts.json").write_text(json.dumps(dict(parts=parts)))
    (tmp_path / "gpu_slots.json").write_text(json.dumps(dict(slots=3)))
    submitted = []
    monkeypatch.setattr(stage, "sbatch", lambda argv, name: submitted.append(argv) or len(submitted))
    stage.submit_labels(types.SimpleNamespace(model="glm", round=1))
    labels = [argv for argv in submitted if str(argv[2]).endswith("label_job.sbatch")]
    assert [argv[:2] for argv in labels] == [["--job-name=v3-gpu-0", "--dependency=singleton"],
                                             ["--job-name=v3-gpu-1", "--dependency=singleton"],
                                             ["--job-name=v3-gpu-2", "--dependency=singleton"],
                                             ["--job-name=v3-gpu-0", "--dependency=singleton"]]
    assert "--dependency=afterany:1:2:3:4" in submitted[-1]
