"""Submit the reviewed labeled-corpus job graph after resource approval."""
import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/correlation_pipeline/labeled-forward-v1"


def publish(path, data):
    text = json.dumps(data, indent=2) + "\n"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(path)
    assert path.read_text() == text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true", help="Requires the reviewed production resource budget approval")
    args = parser.parse_args()
    plan = json.loads((ROOT / "plan.json").read_text())
    gate = json.loads(Path(plan["cpu_gate"]).read_text())
    assert gate["status"] == "passed" and gate["code"] == plan["code"]
    code = Path(plan["code_snapshot"])
    hasher = hashlib.sha256()
    for path in sorted((code / "src/moe_exp").rglob("*.py")):
        hasher.update(path.relative_to(code).as_posix().encode())
        hasher.update(path.read_bytes())
    assert hasher.hexdigest() == plan["code"]["source_sha256"]
    for spec in plan["sources"].values():
        checkpoint = json.loads(Path(spec["checkpoint_report"]).read_text())
        assert checkpoint["status"] == "complete" and checkpoint["revision"] == spec["revision"]
    if not args.submit:
        print(json.dumps({"resources": plan["resources"], "sources": list(plan["sources"]),
                          "stages": ["prepare", "forward", "analyze"],
                          "status": "ready; no jobs submitted"}, indent=2))
        return
    assert subprocess.check_output(["id", "-un"], text=True).strip() == "lmolfett"
    assert "leonardo" in subprocess.check_output(["hostname"], text=True)
    record_path = ROOT / "submissions.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "resources": plan["resources"], "jobs": {}, "status": "submitting",
    }
    def submit(stage, source, dependencies):
        key = source + ":" + stage
        if key in record["jobs"]:
            return record["jobs"][key]["job_id"]
        argv = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes",
                "--job-name=mfa-" + source + "-" + stage]
        if dependencies:
            argv += ["--dependency=" + ",".join(dependencies)]
        if stage == "forward":
            argv += ["--time=%02d:00:00" % plan["resources"]["hours_per_job"]]
        argv += [str(code / "sbatch" / ("native_labeled_" + stage + ".sbatch")), source]
        response = subprocess.check_output(argv, text=True).strip()
        job = response.split(";")[0]
        assert job.isdigit(), response
        record["jobs"][key] = {"job_id": job, "command": argv, "response": response}
        publish(record_path, record)
        observed = subprocess.check_output(["scontrol", "show", "job", job], text=True)
        assert "JobId=" + job in observed and "UserId=lmolfett" in observed
        record["jobs"][key]["verified_scontrol"] = observed
        publish(record_path, record)
        print(key, job, flush=True)
        return job
    previous_prepare = previous_analysis = None
    forwards = []
    for source in plan["sources"]:
        prepare = submit("prepare", source, ["afterany:" + previous_prepare] if previous_prepare else [])
        previous_prepare = prepare
        dependencies = ["afterok:" + prepare]
        if len(forwards) >= 2:
            dependencies.append("afterany:" + forwards[-2])
        forward = submit("forward", source, dependencies)
        forwards.append(forward)
        dependencies = ["afterok:" + forward]
        if previous_analysis:
            dependencies.append("afterany:" + previous_analysis)
        previous_analysis = submit("analyze", source, dependencies)
    record["status"] = "submitted"
    publish(record_path, record)


if __name__ == "__main__":
    main()
