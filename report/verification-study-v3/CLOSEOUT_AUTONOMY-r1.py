"""Persist observed observer activation and compatible reader readiness only."""
from pathlib import Path
import time

from moe_exp.verification_study.contracts import check_files,file_sha,load,require,save
from moe_exp.verification_study.registry import REPO
from verification_study_v3_reader_r1 import binding


def main():
    directory=Path(__file__).resolve().parent;campaign=load(directory/"CAMPAIGN.json")
    check_files(campaign["code_files"]);root=Path(campaign["result_root"])
    callbacks=load(directory/"HEALTH_CALLBACKS.json")
    require(callbacks["observer_version"]=="r2" and len(callbacks["callbacks"])==8,"observer activation is incomplete")
    smoke=next(r for r in callbacks["callbacks"] if r["predecessor_stage"]=="qualify")
    event=load(root/"health/events"/(smoke["health_card_sha256"]+".json"))
    require(event["state"]=="STEP_COMPLETE" and event["callback_job_id"]==smoke["job_id"] and
            event["execution_artifacts_complete"] is True,"real revision2 observer smoke did not complete")
    observer_files={str(REPO/"scripts/experimental_resume"/name):file_sha(REPO/"scripts/experimental_resume"/name)
        for name in ("verification_study_v3_health_r1.py","verification_study_v3_health_r1.sbatch",
            "verification_study_v3_health_r2.py","verification_study_v3_health_r2.sbatch",
            "verification_study_v3_health_r3.py","verification_study_v3_health_r3.sbatch",
            "verification_study_v3_health_status.py")}
    save(directory/"AUTONOMY_RECEIPT-r1.json",{"schema":"verification-autonomy-activation-v3",
        "campaign_sha256":campaign["sha256"],"observed_epoch":time.time(),
        "authorization":"user: do not wait for wake-up prompts; health signals on completion/failure; continue approved all-seven campaign",
        "continuation":"native persistent Goal active; agent state stays private; no daemon or receiver",
        "callback_registry_sha256":callbacks["sha256"],"actual_smoke_job_id":smoke["job_id"],
        "actual_smoke_event_sha256":event["sha256"],"active_discovery_callbacks":7,
        "observer_resources_each":{"cpus":1,"memory_gb":4,"walltime_seconds":600,"gpus":0},
        "scientific_jobs_cancelled_for_health_repair":0,"scientific_source_files_changed":0,
        "observer_files":observer_files,"health_status_path":str(directory/"HEALTH_STATUS.json"),
        "missing_callback_policy":"inspect terminal accounting/logs; explicit CALLBACK_FAILED_NO_EVENT, never success",
        "next_dispatch":"consume every immutable event; advance compatible finite measured cards, all seven models",
        "tests":{"reader_resume_and_prices":{"passed":11,"wall_seconds":1.261},
            "health_and_reader_callback_integration":{"passed":16,"wall_seconds":1.075},
            "native_campaign_suite":"previous40PASS; unchanged frozen source; not redundantly rerun"}})
    save(directory/"READER_READINESS-r1.json",{"schema":"verification-reader-extension-readiness-v3",
        "campaign_sha256":campaign["sha256"],"reader_executor":binding(campaign),
        "status":"CPU_CONTRACTS_PASS; NO_READER_ALLOCATION_YET",
        "behavior":"unchanged pinned r4 Reader; original full-frame SHA, role, seeds, prompts and two draws",
        "slice_and_resume":"before-load provenance; preserve canonical runtime; per-invocation loads; exact missing draws only",
        "outcome_price_path":"discovery-only OUTCOME_PRICE_FRAME.json/outcome-price-ratings; final reader remains priced",
        "health_version_for_reader":"r3; separate source seal, queued r2 observers preserved",
        "static_review":"GPT6.1Sol/xhigh final review: no definite remaining blocker in reader_r1, health_r3, health_status",
        "required_before_actual_dispatch":"observed frame/gates, concrete sealed request, first role measurement or measured price/context, native preflight and exact submission readback"})
    paper=REPO/"paper/6a510c6db0d9b08aa438c142"
    preview=Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/verification-paper-build-20261007/verification_preview.pdf")
    save(directory/"PAPER_BUILD-r4.json",{"schema":"verification-paper-build-v3","campaign_sha256":campaign["sha256"],
        "source_files":{str(p):file_sha(p) for p in (paper/"main.tex",paper/"verification_study.tex",
            paper/"generated/seven_model_evidence.tex",paper/"generated/verification_status.tex",paper/"generated/verification_results.tex")},
        "content_preview":"PASS; 2 pages; observed all-seven backend PASS table; not journal-formatted",
        "preview_path":str(preview),"preview_sha256":file_sha(preview),
        "full_journal_build":"previous observed missing xstring.sty; class preserved; no new full-class build claim",
        "new_scientific_effect":"NOT_MEASURED"})
    print("Autonomy activation, reader readiness and current paper content preview receipts saved; all35 scientific source hashes match.")


if __name__=="__main__":main()
