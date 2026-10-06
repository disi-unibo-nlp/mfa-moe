"""Consolidate verified Slurm IDs from independent stage receipts; no submission."""
from __future__ import annotations
import fcntl
import json
from pathlib import Path

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
DOC = REPO / 'report/experimental-resume-v1'
S = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
BASE = S / 'runs/routing-control-v1'
JOBS = [
    (59172945, 'r3d-qwen-sidecar-first-attempt', 'CPU core-h', None),
    (59173234, 'r3d-qwen-sidecar', 'CPU core-h', S/'runs/resume-v1/r3d-anticipation-qwen-shard-v1/result.json'),
    (59180550, 'r3d-gpt-destination-sidecar', 'CPU core-h', S/'runs/resume-v1/r3d-anticipation-gpt-dest-shard-v1/result.json'),
    (59176972, 'causal-micro-screen-qual4-v1-failed', 'GPU-h', BASE/'micro-screen-qual4-837dd4d99ba79a03/BINDING.json'),
    (59177790, 'causal-micro-screen-qual4-v2-failed', 'GPU-h', BASE/'micro-screen-qual4-6dbc0af8df340880/BINDING.json'),
    (59180099, 'causal-micro-screen-qual4-v3', 'GPU-h', BASE/'micro-screen-qual4-ac4c9651e71fe067/SUMMARY.json'),
    (59181974, 'causal-micro-screen-first-stage', 'CPU core-h', BASE/'micro-screen-first-stage-qual4-ac4c9651e71fe067/FIRST_STAGE.json'),
    (59182208, 'causal-micro-screen-blind-frame', 'CPU core-h', BASE/'micro-screen-blind-qual4-ac4c9651e71fe067/BLIND_FRAME.json'),
    (59179022, 'v22-context-scout-price-v1', 'CPU core-h', DOC/'TRANSITION_V22_QWEN_12_SCOUT_PRICE.json'),
    (59180154, 'v22-context-scout-price-v2', 'CPU core-h', DOC/'TRANSITION_V22_QWEN_12_SCOUT_PRICE_v2.json'),
    (59180589, 'v22-context-scout-ratings', 'GPU-h', BASE/'dense-discovery/ratings-v22-qwen-scout-d8f50b7e-06f8fabd/SUMMARY.json'),
    (59183247, 'v22-full-prefix-audit-price-v2', 'CPU core-h', DOC/'TRANSITION_V22_FULL_PREFIX_START_RATING_PRICE_v2.json'),
    (59186342, 'v22-full-prefix-audit-first-slice', 'GPU-h', BASE/'dense-discovery/ratings-v22-fullprefix-v2-6c10499b-1ef8863f/SUMMARY.json'),
    (59181590, 'gptoss-start-preprice-import-failure', 'CPU core-h', None),
    (59181964, 'gptoss-start-preprice-v1-invalid-count', 'CPU core-h', DOC/'GPTOSS_START_PILOT_PREPRICE_v1.json'),
    (59183252, 'gptoss-start-preprice-v2-scout-only', 'CPU core-h', DOC/'GPTOSS_START_PILOT_PREPRICE_v2.json'),
    (59185128, 'gptoss-start-preprice-v3-all-prefixes', 'CPU core-h', DOC/'GPTOSS_START_PILOT_PREPRICE_v3.json'),
    (59186768, 'candidate-veto-adapted-first-attempt', 'CPU core-h', None),
    (59186966, 'candidate-veto-adapted-second-attempt', 'CPU core-h', None),
    (59187229, 'candidate-veto-adapted-exploratory', 'CPU core-h', DOC/'CANDIDATE_VETO_ADAPTED_EXPLORATORY_v1.json'),
    (59189594, 'native-prefix-semantic-veto-price', 'CPU core-h', DOC/'NATIVE_PREFIX_SEMANTIC_VETO_PRICE_v0.json'),
    (59185125, 'micro-blind-rating-price-v1-superseded', 'CPU core-h', DOC/'CAUSAL_MICROSCREEN_BLIND_RATING_PRICE_v1.json'),
    (59186670, 'micro-blind-rating-price-v2-superseded', 'CPU core-h', DOC/'CAUSAL_MICROSCREEN_BLIND_RATING_PRICE_v2.json'),
    (59187379, 'micro-blind-rating-price-v2p1', 'CPU core-h', DOC/'CAUSAL_MICROSCREEN_BLIND_RATING_PRICE_v2.1.json'),
    (59188622, 'micro-blind-rating-price-v2p2', 'CPU core-h', DOC/'CAUSAL_MICROSCREEN_BLIND_RATING_PRICE_v2.2.json'),
    (59187030, 'native-only-replay-v1-table-failure', 'GPU-h', BASE/'native-only-replay-4001194242cd564b/BINDING.json'),
    (59189256, 'native-only-replay-v2-table-rebuild-failure', 'GPU-h', BASE/'native-only-replay-3f05cdc9f56c7002/BINDING.json'),
    (59189980, 'native-only-replay-v3', 'GPU-h', BASE/'native-only-replay-4b4eda23ab7747ec/SUMMARY.json'),
    (59188107, 'micro-blind-rating-normal-v2p1-held', 'GPU-h', BASE/'micro-screen-semantic-v2p1-27cd70792e1ae110/SUMMARY.json'),
    (59189744, 'micro-blind-rating-debug-v2p2', 'GPU-h', BASE/'micro-screen-semantic-v22-1e56570b050828e3/SUMMARY.json'),
    (59193843, 'native-prefix-semantic-veto-scout-v0', 'GPU-h', BASE/'dense-discovery/ratings-native-veto-scout-v0-d8f50b7e-8d2b37d2/SUMMARY.json'),
    (59194320, 'approach-boundary-class-proxy-shortlist-v1', 'CPU core-h', BASE/'approach-boundary-proxy-v1/RESULT.json'),
    (59186394, 'r3d-qwen-sidecar-verifier-v1-failed', 'CPU core-h', None),
    (59187091, 'r3d-qwen-sidecar-verifier-v2', 'CPU core-h', DOC/'R3D_SIDECAR_VERIFY_QWEN_v2.json'),
    (59188564, 'r3d-qwen-sidecar-guarded-merge', 'CPU core-h', DOC/'R3D_SIDECAR_MERGE_QWEN_v1.json'),
    (59186772, 'r3e-clean-fold-preflight', 'CPU core-h', DOC/'R3E_CLEAN_PREFLIGHT_v1.json'),
    (59187838, 'r3e-clean-b1-cache-path-failure', 'CPU core-h', None),
    (59189035, 'r3e-clean-b1-v2', 'CPU core-h', S/'runs/resume-v1/r3e-b1-clean-v2/result.json'),
    (59197189, 'r3e-clean-b4-features-smoke-v2', 'CPU core-h', S/'runs/resume-v1/r3e-b4-features-v2/FEATURES_FROZEN_B4_smoke.json'),
    (59197144, 'discovery-evidence-snapshot-v2', 'CPU core-h', DOC/'DISCOVERY_EVIDENCE_SNAPSHOT_v2.json'),
    (59197523, 'discovery-evidence-snapshot-v2-recovery', 'CPU core-h', DOC/'DISCOVERY_EVIDENCE_SNAPSHOT_v2.json'),
    (59197542, 'r3e-clean-b4-features-full-v2', 'CPU core-h', S/'runs/resume-v1/r3e-b4-features-v2/FEATURES_FROZEN_B4.json'),
    (59198677, 'r3e-clean-b4-cv-smoke-v1', 'CPU core-h', S/'runs/resume-v1/r3e-cv-clean-v1/smoke.json'),
    (59198802, 'r3e-clean-b4-cv-full-v1', 'CPU core-h', S/'runs/resume-v1/r3e-cv-clean-v1/full.json'),
    (59199203, 'r3e-clean-b4-features-confirm-exclusion-smoke-v3', 'CPU core-h', S/'runs/resume-v1/r3e-b4-features-confirm-exclusion-v3/FEATURES_FROZEN_B4_smoke.json'),
    (59199796, 'r3e-clean-b4-features-confirm-exclusion-full-v3', 'CPU core-h', S/'runs/resume-v1/r3e-b4-features-confirm-exclusion-v3/FEATURES_FROZEN_B4.json'),
    (59185118, 'x3-cost-timing-only-pilot', 'GPU-h', S/'runs/resume-v1/x3-timing-pilot-v1/results-8f7c67d6b2a06e42/shard-0.status.json'),
    (59194916, 'x3-cost-timing-only-pilot-v4', 'GPU-h', S/'runs/resume-v1/x3-timing-pilot-v1/results-8f7c67d6b2a06e42-v4/shard-0.status.json'),
    (59188318, 'counterfactual-qualification-cpu-prep', 'CPU core-h', DOC/'COUNTERFACTUAL_QUALIFICATION_MANIFEST_v1.json'),
    (59189255, 'counterfactual-routing-loss-qualification-v2', 'GPU-h', BASE/'counterfactual-qualification-b7c4c5273ab6fdc7/QUALIFICATION.json'),
    (59196354, 'score-api-calibration-v3', 'GPU-h', BASE/'score-api-calibration-b4aa1c878c25fe32/QUALIFICATION.json'),
    (59198795, 'score-api-calibration-v4-serial', 'GPU-h', BASE/'score-api-serial-e7554bd7efceb410/QUALIFICATION.json'),
    (59197280, 'native-prefix-semantic-veto-full-v0', 'GPU-h', BASE/'dense-discovery/ratings-native-veto-full-v0-6c10499b-191f97fe/SUMMARY.json'),
    (59197533, 'causal-batch-invariant-qualification-v1', 'GPU-h', BASE/'batch-invariant-qual-db182a1debc7465c/SUMMARY.json'),
    (59198296, 'causal-batch-invariant-qualification-audit-v1', 'CPU core-h', DOC/'CAUSAL_BATCH_INVARIANT_QUAL_AUDIT_v1.json'),
    (59199103, 'transition-enrichment-count-price-v0', 'CPU core-h', DOC/'TRANSITION_ENRICHMENT_COUNT_PRICE_v0.json'),
    (59199073, 'native-prefix-veto-all-fire-cost-audit-v0', 'CPU core-h', DOC/'NATIVE_VETO_ALL_FIRE_BURDEN_v0.json'),
    (59199245, 'full-prefix-reader-agreed-eligible-pool-v0', 'CPU core-h', BASE/'dense-discovery/FULLPREFIX_READER_AGREED_ELIGIBLE_POOL_v0.json'),
    (59199761, 'gptoss-start-parity-pilot-price-v1', 'CPU core-h', DOC/'GPTOSS_START_PILOT_PRICE_v1.json'),
    (59199929, 'gptoss-start-parity-pilot-v1', 'GPU-h', BASE/'dense-discovery/ratings-gptoss-start-pilot-v1-18a7f804-eda3cc45/RESULTS.json'),
    (59200002, 'causal-micro-serial-engine-qualification-v1', 'GPU-h', BASE/'micro-serial-qual-68670e1dcdb8f3a2/SUMMARY.json'),
    (59200254, 'causal-micro-serial-engine-qualification-audit-v1', 'CPU core-h', DOC/'CAUSAL_MICRO_SERIAL_QUAL_AUDIT_v1.json'),
    (59200261, 'candidate-veto-prefix-only-cpu-screen-v1', 'CPU core-h', DOC/'CANDIDATE_VETO_FULLPREFIX_DISCOVERY_v1.json'),
    (59200183, 'r3e-postprimary-fixed-exploratory-diagnostic-v1', 'CPU core-h', S/'runs/resume-v1/r3e-postprimary-diagnostics-v1/result.json'),
    (59200275, 'measurement-evidence-snapshot-v3', 'CPU core-h', DOC/'MEASUREMENT_EVIDENCE_SNAPSHOT_v3.json'),
    (59200701, 'r3e-clean-b4-cv-confirm-exclusion-smoke-v1', 'CPU core-h', S/'runs/resume-v1/r3e-cv-clean-confirm-v1/smoke.json'),
    (59200800, 'r3e-clean-b4-cv-confirm-exclusion-full-v1', 'CPU core-h', S/'runs/resume-v1/r3e-cv-clean-confirm-v1/full.json'),
    (59201047, 'r3d-gpt-destination-sidecar-verifier-v2', 'CPU core-h', DOC/'R3D_SIDECAR_VERIFY_GPT_DEST_v2.json'),
    (59201091, 'r3d-gpt-destination-sidecar-guarded-merge-v1', 'CPU core-h', DOC/'R3D_SIDECAR_MERGE_GPT_DEST_v1.json'),
    (59200963, 'approach-supported-expert-rerank-v2-env-failure', 'CPU core-h', None),
    (59201103, 'approach-supported-expert-rerank-v2', 'CPU core-h', BASE/'approach-boundary-supported-v2/RESULT.json'),
    (59201139, 'gptoss-expanded-start-panel-price-v2', 'CPU core-h', DOC/'GPTOSS_START_PANEL_PRICE_v2.json'),
    (59201541, 'gptoss-expanded-start-panel-v2', 'GPU-h', BASE/'dense-discovery/ratings-gptoss-start-panel-v2-9e81a6b6/SUMMARY.json'),
    (59201455, 'within-sentence-prefix-timing-inventory-env-failure', 'CPU core-h', None),
    (59201714, 'within-sentence-prefix-timing-inventory-v0', 'CPU core-h', DOC/'WITHIN_SENTENCE_TIMING_INVENTORY_v0.json'),
    (59201923, 'r3e-b4-precision-smoke-v1-fold-failure', 'CPU core-h', None),
    (59202169, 'r3e-b4-precision-smoke-v2', 'CPU core-h', S/'runs/resume-v1/r3e-b4-precision-v2/smoke.json'),
    (59202682, 'r3e-b4-precision-v2-aggregate', 'CPU core-h', S/'runs/resume-v1/r3e-b4-precision-v2/summary.json'),
    (59202685, 'within-sentence-start-rating-price-v1', 'CPU core-h', DOC/'WITHIN_SENTENCE_START_RATING_PRICE_v1.json'),
    (59203153, 'causal-batched-neighbor-qualification-v1', 'GPU-h', BASE/'micro-neighbor-qual-e0c498b65dce4dc7/SUMMARY.json'),
    (59203204, 'causal-batched-neighbor-qualification-audit-v1', 'CPU core-h', DOC/'CAUSAL_BATCHED_NEIGHBOR_QUAL_AUDIT_v1.json'),
    (59203262, 'within-sentence-qwen-audit-v1', 'GPU-h', BASE/'dense-discovery/ratings-within-sentence-v1-ed8522b7/SUMMARY.json'),
    (59203268, 'within-sentence-qwen-audit-v1-duplicate-cancelled', 'GPU-h', None),
    (59203778, 'joint-reader-exact-pool-v1-host-guard-failure', 'CPU core-h', None),
    (59203918, 'joint-reader-exact-pool-v1', 'CPU core-h', BASE/'dense-discovery/JOINT_QWEN_NATIVE_EXACT_POOL_v1.json'),
    (59203686, 'r3e-b4-precision-v3-aggregate', 'CPU core-h', S/'runs/resume-v1/r3e-b4-precision-v3/summary.json'),
    (59204242, 'eligible-causal-micro-screen-v2', 'GPU-h', BASE/'eligible-micro-serial-v2-0f9dc13adfcae354/SUMMARY.json'),
    (59204384, 'causal-batched-neighbor-qualification-audit-v2-preflight-failure', 'CPU core-h', None),
    (59204381, 'approach-online-gate-discovery-v1', 'CPU core-h', DOC/'APPROACH_ONLINE_GATE_DISCOVERY_v1.json'),
    (59204465, 'x3-native-label-density-price-v1', 'CPU core-h', DOC/'X3_LABEL_DENSITY_PROJECTION_v1.json'),
    (59204536, 'causal-batched-neighbor-qualification-audit-v3', 'CPU core-h', DOC/'CAUSAL_BATCHED_NEIGHBOR_QUAL_AUDIT_v2.json'),
    (59205653, 'eligible-expert-deactivation-screen-v1', 'GPU-h', BASE/'eligible-deactivation-serial-v1-86582a4de3c16c0c/SUMMARY.json'),
    (59206783, 'r3d-qwen-sidecar-guarded-merge-v2', 'CPU core-h', DOC/'R3D_SIDECAR_MERGE_QWEN_v2.json'),
    (59206978, 'r3d-gpt-destination-sidecar-guarded-merge-v2', 'CPU core-h', DOC/'R3D_SIDECAR_MERGE_GPT_DEST_v2.json'),
    (59207468, 'r3d-postmerge-inventory-audit-v1-unexpected-predictions-failure', 'CPU core-h', None),
    (59207811, 'r3d-postmerge-inventory-audit-v2', 'CPU core-h', DOC/'R3D_MERGED_INVENTORY_AUDIT_v2.json'),
    (59208033, 'r3d-finalization-after-merge-v1', 'CPU core-h', DOC/'R3D_FINALIZATION_AFTER_MERGE_v1.json'),
]
JOBS.extend((f'59202240_{i}', f'r3e-b4-precision-v2-shard-{i:02}', 'CPU core-h',
             S/f'runs/resume-v1/r3e-b4-precision-v2/shard-{i:02}.json') for i in range(20))


def main():
    path = DOC/'DIRECT_JOB_REGISTRY.json'
    with (DOC/'.direct-job-registry.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        registry = json.loads(path.read_text())
        if registry['schema'] != 'experimental-resume-direct-slurm-jobs-v1':
            raise ValueError('unknown registry schema')
        known = {str(r['job_id']): r for r in registry['jobs']}
        added = []
        for job_id, task, unit, artifact in JOBS:
            row = {'job_id': job_id, 'task': task, 'unit': unit}
            if artifact is not None:
                row['artifact'] = str(artifact)
            if str(job_id) in known:
                if known[str(job_id)] != row:
                    raise ValueError('different registry row for job ' + str(job_id))
                continue
            registry['jobs'].append(row)
            known[str(job_id)] = row
            added.append(job_id)
        registry['authorization'] = 'RESOURCE_AUTHORIZATION_2026-10-02_v2.json'
        pending = path.with_suffix('.json.part')
        pending.write_text(json.dumps(registry, indent=2) + '\n')
        pending.replace(path)
        print(json.dumps({'registered': len(registry['jobs']), 'added': added}))


if __name__ == '__main__':
    main()
