"""A storage stop resumes compatible saved evidence without another extraction."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moe_exp.research_campaign_v3_r5_r2 import contracts as C, controller as O
from moe_exp.research_campaign_v3_r5_r2.resume import analysis_sources, import_completed_analysis


class Continuation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="continuation-tests-", dir=C.REPO/"report")
        self.root = Path(self.temp.name)
        self.old, self.new = self.root/"parent", self.root/"child"
        names = ("accuracy.py", "derive.py", "design.py", "numeric.py", "inputs.py", "artifacts.py",
                 "contracts.py", "observations.py", "populations.py", "statistics.py", "extraction.py", "native.py")
        def bindings(package):
            paths = [C.REPO/"src/moe_exp"/package/name for name in names]
            return {str(p): C.file_sha(p) for p in paths}
        common = {"models": {"gpt": {"profile_sha256": "native", "layers": [0, 1], "experts": 4}},
                  "saved_inputs": {"gpt": {"attempts": "immutable-original"}},
                  "analysis_binding": "unchanged-statistics", "runtime_files": {}}
        self.parent = C.seal({**common, "root": str(self.old), "code_files": bindings("research_campaign_v3")})
        self.campaign = C.seal({**common, "root": str(self.new), "code_files": bindings("research_campaign_v3_r5_r2")})
        for relative in ("FOLDS.json", "COHORTS.json", "gpt/SOURCES.json", "gpt/PREFIXES.json",
                         "gpt/TOKEN_OWNERSHIP.json", "gpt/NOMINATIONS.json"):
            for root in (self.old, self.new):
                C.publish(root/relative, {"records": [], "identity": relative})
        for name in ("ANALYSIS_PRICE", "COORDINATION_SHARDS", "NATIVE_DYNAMICS_SUMMARY", "SUPPLEMENTARY",
                     "ASSOCIATION_CONFIRMATION"):
            C.publish(self.old/"gpt"/(name+".json"), {"status": "complete"})
        C.publish(self.old/"gpt/DERIVED.json", {"status": "complete", "analysis_binding": common["analysis_binding"]})
        weighted = self.old/"gpt/NATIVE_WEIGHTED.npz"
        weighted.write_bytes(b"durable-weighted-occupancy")
        C.publish(self.old/"gpt/EXTRACTION.json", {"status": "complete",
            "sources_sha256": C.load(self.old/"gpt/SOURCES.json")["sha256"],
            "ownership_sha256": C.load(self.old/"gpt/TOKEN_OWNERSHIP.json")["sha256"],
            "weighted": {"path": str(weighted), "sha256": C.file_sha(weighted)}})
        for name in ("transfer", "source_conditioned_next", "native_boundaries", "native_weight_occupancy", "native_turnover"):
            (self.old/"gpt"/(name+".parquet")).write_bytes(b"retained-original-table")
        self.task = {"shard_index": 1, "layer_a": 0, "layer_b": 1, "depth_gap": 1}

    def tearDown(self):
        self.temp.cleanup()

    def pair(self, corrupt=False):
        directory = self.old/"gpt/coordination/pair-001"
        C.publish(directory/"NOMINATIONS.json", {**self.task, "records": [], "outcomes_used": False})
        payload = directory/"descriptive_joints.npz"
        payload.write_bytes(b"retained-original-joints")
        sufficient = C.publish(directory/"sufficient-0-0.json", {"records": [], "resamples": 10000,
            "analysis_binding": self.campaign["analysis_binding"], "resample_key": "original-key"})
        C.publish(directory/"RESULT.json", {**self.task, "status": "complete",
            "sources_sha256": C.load(self.old/"gpt/SOURCES.json")["sha256"],
            "ownership_sha256": C.load(self.old/"gpt/TOKEN_OWNERSHIP.json")["sha256"],
            "artifacts": [{"path": str(payload), "sha256": "0"*64 if corrupt else C.file_sha(payload)}],
            "panels": [{"fold": 0, "class": "Read", "pairs": [{"expert_a": 0, "expert_b": 1}],
                        "sufficient_statistics": sufficient["sha256"]},
                       {"fold": 0, "class": "Analyze", "pairs": [], "status": "insufficient_support"}]})

    def run_import(self):
        refs = []
        with patch("moe_exp.research_campaign_v3_r5_r2.observations.pair_tasks", return_value=[self.task]):
            imported = import_completed_analysis(self.campaign, [self.parent], refs)
        return imported, refs

    def test_identical_science_across_new_operational_package(self):
        self.assertEqual(analysis_sources(self.parent), analysis_sources(self.campaign))
        self.pair()
        imported, refs = self.run_import()
        self.assertEqual(imported[0]["pairs"], 1)
        self.assertTrue(refs)
        pair = self.new/"gpt/coordination/pair-001/RESULT.json"
        self.assertTrue(pair.is_symlink())
        self.assertEqual(C.load(pair), C.load(self.old/"gpt/coordination/pair-001/RESULT.json"))
        self.assertEqual(C.load(self.new/"stages/derive-gpt-00-00000/RESULT.json")["status"], "reused")
        self.assertTrue(C.load(self.new/"gpt/CACHE_RETIRED.json")["child_temporary_cache_never_allocated"])
        self.assertFalse((self.new/"gpt/compact").exists())
        self.assertEqual(len(C.load(pair)["panels"]), 2)  # Preserve zero-support cells.
        self.assertEqual(self.run_import()[0], imported)  # Restart of interrupted import.

    def test_partial_analysis_not_declared_complete(self):
        self.assertEqual(self.run_import(), ([], []))
        self.assertFalse((self.new/"gpt/CACHE_RETIRED.json").exists())

    def test_55_of_77_components_reused_without_repeating_extraction(self):
        self.pair()
        extraction=C.load(self.old/'gpt/EXTRACTION.json')
        extraction.update(temporary_cache_owner='original-owner',records=[{'artifacts':[]}])
        (self.old/'gpt/EXTRACTION.json').write_bytes(C.canonical(C.seal(extraction)))
        C.publish(self.old/'gpt/compact/OWNER.json',{'campaign_sha256':'original-owner'})
        tasks=[{**self.task,'shard_index':i} for i in range(1,78)]
        original=C.load(self.old/'gpt/coordination/pair-001/RESULT.json')
        import shutil
        for i in range(2,56):
            path=self.old/'gpt/coordination'/f'pair-{i:03d}'
            shutil.copytree(self.old/'gpt/coordination/pair-001',path)
            (path/'RESULT.json').write_bytes(C.canonical(C.seal({**original,'shard_index':i})))
        with patch('moe_exp.research_campaign_v3_r5_r2.observations.pair_tasks',return_value=tasks):
            imported=import_completed_analysis(self.campaign,[self.parent],[])
        self.assertEqual(imported[0]['pairs'],55)
        proof=C.load(self.new/'gpt/ANALYSIS_REUSE.json')
        self.assertEqual(proof['status'],'partial')
        self.assertEqual(proof['required_pair_count'],77)
        self.assertFalse((self.new/'gpt/CACHE_RETIRED.json').exists())
        self.assertTrue((self.new/'gpt/compact').is_symlink())

    def test_scientific_source_change_rejects_operational_reuse(self):
        self.pair()
        path=self.root/'src/moe_exp/research_campaign_v3_r5_r2/native.py'
        path.parent.mkdir(parents=True)
        source=C.REPO/'src/moe_exp/research_campaign_v3_r5_r2/native.py'
        path.write_text(source.read_text()+'\nSCIENTIFIC_CHANGE=True\n')
        bindings={p:h for p,h in self.campaign['code_files'].items() if Path(p).name!='native.py'}
        bindings[str(path)]=C.file_sha(path)
        self.campaign={**self.campaign,'code_files':bindings}
        self.assertEqual(self.run_import(),([],[]))

    def test_corrupt_artifact_blocks_reuse_before_commits(self):
        self.pair(corrupt=True)
        with self.assertRaises(ValueError):
            self.run_import()
        self.assertFalse((self.new/"gpt/ANALYSIS_REUSE.json").exists())

    def test_changed_analysis_binding_does_not_reuse(self):
        self.pair()
        self.campaign = {**self.campaign, "analysis_binding": "changed-statistics"}
        self.assertEqual(self.run_import(), ([], []))

    def test_unrelated_gpu_jobs_do_not_delay_owned_cpu_completion(self):
        snapshot = {"jobs": [{"job_id": "gpu", "partition": "boost_usr_prod"},
                             {"job_id": "cpu", "partition": "lrd_all_serial"},
                             {"job_id": "snapshot", "partition": "lrd_all_serial"}]}
        observed = [{"accepted": {"job_id": "snapshot"}, "card": {"stage": "snapshot"}}]
        work = [{"partition": "lrd_all_serial"}]
        self.assertEqual(O.capacity_events(snapshot, observed, work, ["owned-analysis"], [], "controller"), [])
        self.assertEqual(O.capacity_events(snapshot, observed, work, [], ["new-analysis"], "controller"), [])
        self.assertEqual(O.capacity_events(snapshot, observed, work, [], [], "controller"), ["cpu"])
        self.assertEqual(O.capacity_events(snapshot, observed, [], [], [], "controller"), [])

    def test_unpriced_storage_is_one_pilot_and_bulk_uses_full_measured_bank(self):
        from moe_exp.research_campaign_v3_r5_r2.analysis import pair_storage_bound
        campaign = {"root": str(self.new), "compact_extraction": True,
                    "models": {"gpt": {"layers": list(range(40)), "experts": 256, "native_k": 4}}}
        with patch.object(O, "MODELS", ("gpt",)):
            pilot = O._saved_output_residual(campaign)
            fewer = O._saved_output_residual({**campaign, "models": {"gpt": {"layers": [0,1], "experts": 256, "native_k": 4}}})
            self.assertEqual(pilot, fewer)  # More unpriced pairs do not authorize bulk work.
            self.assertIsNone(pair_storage_bound(campaign, "gpt"))
            C.publish(self.new/"gpt/coordination/pair-001/STORAGE_PRICE.json",
                      {"pair_output_bytes": 40*1024**2, "status": "complete"})
            measured = O._saved_output_residual(campaign)
            self.assertGreater(measured, pilot)
            self.assertEqual(pair_storage_bound(campaign, "gpt"), 50*1024**2)
        from moe_exp.research_campaign_v3_r5_r2.scheduler import admission
        check = admission({"remaining_local_hours": 42050., "consumed_local_hours": 25950.,
                           "storage_headroom_bytes": measured+10*1024**3-1},
                          baseline_consumed=25950., campaign_actual=0., outstanding=0.,
                          residual=1., output_bytes=measured)
        self.assertFalse(check["admitted"])  # The mandatory reserve remains binding.
