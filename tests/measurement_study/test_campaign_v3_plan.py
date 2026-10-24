import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from measurement_study.campaign_v3_plan import (
    build_manifests, build_rows, cohort_duplicate_report, fit_attempt_runtime, load_cohort_manifest,
    outcome_key, prepare_spec, publish_manifests, read_outcomes, token_budget_shards,
    trace_content_hash, validate_cohort_binding, validate_storage_root, verification_pair,
)
from measurement_study.production_plan import digest, read_location, sha

STAGE = '/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp'
REAL_COHORT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/campaign-v3/test-cohorts/out/cohorts.json')
REAL_OUTCOMES = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/campaign-v3/outcomes/outcomes_v3.latest.json')
PREPARED_CAMPAIGN = Path(__file__).resolve().parents[2]/'report/measurement-study/campaign-v3-preparation/campaign-v3.json'


class CampaignV3PlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=STAGE)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root/'checkpoint'
        self.checkpoint.mkdir()
        config = dict(model_type='gpt_oss', num_hidden_layers=2, hidden_size=4,
                      num_local_experts=4, num_experts_per_tok=2)
        (self.checkpoint/'config.json').write_text(json.dumps(config))
        (self.checkpoint/'model.safetensors').write_bytes(b'fixture-only')
        for name in ('tokenizer.json', 'tokenizer_config.json'):
            (self.checkpoint/name).write_text('{}')
        self.spec = dict(generation_model='fixture/model', checkpoint=str(self.checkpoint),
                         revision='pinned', config_sha256=sha(self.checkpoint/'config.json'),
                         layers=[0, 1], experts=4, top_k=2, hidden_size=4,
                         tokenizer_files={name: sha(self.checkpoint/name) for name in ('tokenizer.json', 'tokenizer_config.json')})
        self.traces = []
        for sample, tokens in enumerate((300, 5000)):
            self.traces.append(dict(model_id='fixture/model', dataset='aime24', problem_id='q',
                                   source_problem_id='q', sample_id=sample, prompt='question',
                                   generation_messages=None, system_prompt=None, cot_text='Reason.',
                                   is_correct=False, metadata={'token_replay': {
                                       'prompt_token_ids': [1, 2], 'completion_token_ids': [3]*tokens,
                                       'completion_offsets': [[0, 7]]*tokens}}))
        path = self.root/'traces.jsonl'
        payloads = [(json.dumps(trace)+'\n').encode() for trace in self.traces]
        path.write_bytes(b''.join(payloads))
        self.rows, self.outcomes = [], []
        offset = 0
        for i, (trace, raw) in enumerate(zip(self.traces, payloads)):
            trace_hash = trace_content_hash(trace)
            base = dict(dataset='aime24', problem_id='q', source_problem_id='q', sample_id=i,
                        trace_sha256=trace_hash, completion_tokens=len(trace['metadata']['token_replay']['completion_token_ids']),
                        termination='length' if i else 'stop')
            self.rows.append(dict(base, cohort='B', trace_path=str(path), line_index=i, byte_offset=offset,
                                  line_sha256=hashlib.sha256(raw).hexdigest(), stratum='capped' if i else 'correct',
                                  inclusion_probability=None, sentence_selection={'schema_version':1,'selection':'fixture','indices':[0]}))
            self.outcomes.append(dict(base, corpus='card-v3', model='gpt', strict_correct=None if i else True,
                                      adjudicated_correct=None if i else True, minerva_tolerance_correct=None,
                                      adjudication_status='capped' if i else 'math_verify_true', scorer_contract_version=3))
            offset += len(raw)
        self.outpath = self.root/'outcomes_v3.json'
        self.write_outcomes()
        self.rewrite_cohort()
        self.storage = self.root/'campaign-v3'
        self.annotations = None

    def write_outcomes(self):
        self.outpath.write_text(json.dumps(self.outcomes))
        self.outpath.with_suffix('.manifest.json').write_text(json.dumps({'sha256':sha(self.outpath),'scorer_contract_version':3,'inputs':[]}))

    def rewrite_cohort(self, rows=None):
        """Rebuild the cohort manifest the way sample_cohorts.write() does."""
        if rows is not None:
            self.rows = rows
        core = dict(schema_version=1, model='gpt', seed=20260925, label_total=100, per_b=2, floor=1,
                    traces_sha256='e'*64)
        self.cohort = dict(core, manifest_sha256=digest(core), corpus='card-v3',
                           counts=dict(A=sum(1 for r in self.rows if r['cohort']=='A'),
                                       B=sum(1 for r in self.rows if r['cohort']=='B'), labeled_sentences=2),
                           rows=self.rows)
        self.cohort_path = self.root/'cohorts.json'
        self.cohort_path.write_text(json.dumps(self.cohort))
        return self.cohort

    def a_rows(self):
        return [{**row, 'cohort':'A', 'inclusion_probability':.25} for row in self.rows if row['cohort']=='B']

    def build(self, **kw):
        args = dict(cohort_path=self.cohort_path, spec=self.spec, source='gpt',
                    storage_root=self.storage, code_binding='frozen-code', p90_seconds_per_token=.01,
                    startup_seconds=10, per_attempt_seconds=1., bytes_per_token=100, selection='B',
                    layout_fn=lambda row: {'reasoning_tokens':list(range(row['completion_tokens'])), 'token_count':row['completion_tokens'], 'units':[{'index':0,'start':0,'end':7,'text':'Reason.'}]})
        args.update(kw)
        return build_manifests(**args)

    def test_label_free_capture_binds_only_cohort_trace_and_code(self):
        manifests, verify, ready = self.build()
        rows = sorted(manifests[0]['attempts'], key=lambda row: row['identity']['sample_id'])
        self.assertNotIn('outcomes', rows[0])
        self.assertNotIn('is_correct', rows[0]['identity'])
        self.assertEqual(rows[1]['identity']['termination'], 'length')
        self.assertTrue(rows[1]['identity']['capped'])
        self.assertTrue(all('annotation' not in row['cohorts'][0] for row in rows))
        self.assertIsNone(rows[0]['cohorts'][0]['inclusion_probability'])
        self.assertEqual(manifests[0]['binding'], digest({k:v for k,v in manifests[0].items() if k != 'binding'}))
        self.assertEqual(verify['expected_attempts'], 2)
        self.assertEqual(manifests[0]['validation_binding'], verify['validation_binding'])
        self.assertEqual(verify['maximum_input_tokens'], 5002)
        self.assertEqual(ready['attempts'], 2)
        self.assertIn('/campaign-v3/capture/gpt/B/', manifests[0]['capture_dir'])
        self.assertEqual(verify['cohort_manifest_path'], str(self.cohort_path))
        self.assertEqual(verify['cohort_binding'], load_cohort_manifest(self.cohort_path)[1])
        serialized = json.dumps(manifests[0])
        for forbidden in ('outcome_binding','outcomes','is_correct','adjudicated_correct','adjudication_status','outcome_sidecar'):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(validate_cohort_binding(manifests[0]['cohort_binding'])['model'], 'gpt')

    def test_locations_attach_selection_without_source_mutation(self):
        manifests, _, _ = self.build()
        row = manifests[0]['attempts'][0]
        raw = read_location(row['source_location'])
        derived = read_location(row['cohorts'][0]['locations'][0])
        self.assertNotIn('sentence_selection', raw['metadata'])
        self.assertEqual(derived['metadata']['sentence_selection']['indices'], [0])

    def test_cohort_manifest_sha256_and_file_tampering_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=STAGE) as temp:
            tampered = Path(temp)/'cohorts.json'
            document = json.loads(self.cohort_path.read_text())
            document['label_total'] = 101
            tampered.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, 'manifest_sha256'):
                load_cohort_manifest(tampered)
        manifests, _, _ = self.build()
        self.cohort_path.write_text(self.cohort_path.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'changed since planning'):
            validate_cohort_binding(manifests[0]['cohort_binding'], self.cohort_path)
        # The stale published binding never rebinds to the edited file.
        with self.assertRaisesRegex(ValueError, 'changed since planning'):
            validate_cohort_binding(manifests[0]['cohort_binding'])

    def test_cohort_b_null_probability_and_cohort_a_range(self):
        manifests, _, _ = self.build()
        self.assertIsNone(manifests[0]['attempts'][0]['cohorts'][0]['inclusion_probability'])
        a_rows = self.a_rows()
        self.rewrite_cohort([{**self.rows[0], 'inclusion_probability':.5}])
        with self.assertRaisesRegex(ValueError, 'Cohort B inclusion'):
            self.build()
        self.rewrite_cohort([{**row, 'inclusion_probability':None} for row in a_rows])
        with self.assertRaisesRegex(ValueError, 'cohort A inclusion'):
            self.build(selection='A', verification_only=True, verification_seconds=18000,
                       p90_seconds_per_token=None, startup_seconds=None, per_attempt_seconds=None, bytes_per_token=None)
        self.rewrite_cohort([{**row, 'inclusion_probability':0.0} for row in a_rows])
        with self.assertRaisesRegex(ValueError, 'cohort A inclusion'):
            self.build(selection='A', verification_only=True, verification_seconds=18000,
                       p90_seconds_per_token=None, startup_seconds=None, per_attempt_seconds=None, bytes_per_token=None)
        self.rewrite_cohort(a_rows)
        self.build(selection='A', verification_only=True, verification_seconds=18000,
                   p90_seconds_per_token=None, startup_seconds=None, per_attempt_seconds=None, bytes_per_token=None)

    def test_outcome_sidecar_pointer_filters_and_explicit_nullable_rows(self):
        rows, binding = read_outcomes(REAL_OUTCOMES)
        pointer = json.loads(REAL_OUTCOMES.read_text())
        # the live sidecar grows as continuation/card-v3 corpora are adjudicated; the
        # historical corpus is fixed at 32,529 attempts
        import pyarrow.parquet as pq
        self.assertEqual(len(rows), pq.read_metadata(pointer['path']).num_rows)
        self.assertEqual(sum(row['corpus'] == 'historical' for row in rows), 32529)
        self.assertIn('capped', {row['adjudication_status'] for row in rows})
        self.assertEqual(binding['path'], pointer['path'])
        self.assertEqual(binding['sha256'], sha(pointer['path']))
        self.assertEqual(binding['pointer_sha256'], sha(REAL_OUTCOMES))
        filtered, filtered_binding = read_outcomes(REAL_OUTCOMES, corpus='card-v3', model='glm')
        self.assertEqual(filtered, [])
        self.assertEqual(filtered_binding['rows'], 0)
        with tempfile.TemporaryDirectory(dir=STAGE) as temp:
            pointer = Path(temp)/'outcomes.latest.json'
            pointer.write_text(json.dumps({'path':binding['path'],'sha256':'0'*64}))
            with self.assertRaisesRegex(ValueError, 'pointer/checksum'):
                read_outcomes(pointer)
        # Sidecars may carry inherited nulls, capped rows and arbitrary vLLM
        # finish reasons; none of those may fail a card-v3 read.
        explicit = [dict(rows[0], corpus='historical', sample_id=None, completion_tokens=None,
                         termination=None, adjudication_status='capped', adjudicated_correct=False)]
        sidecar = self.root/'explicit.json'
        sidecar.write_text(json.dumps(explicit))
        sidecar.with_suffix('.manifest.json').write_text(json.dumps({'sha256':sha(sidecar)}))
        parsed, explicit_binding = read_outcomes(sidecar)
        self.assertIsNone(parsed[0]['completion_tokens'])
        self.assertIsNone(parsed[0]['sample_id'])
        self.assertEqual(parsed[0]['termination'], 'unknown')
        self.assertEqual(explicit_binding['manifest_sha256'], sha(sidecar.with_suffix('.manifest.json')))
        explicit[0]['termination'] = 'abort'
        sidecar.write_text(json.dumps(explicit))
        sidecar.with_suffix('.manifest.json').write_text(json.dumps({'sha256':sha(sidecar)}))
        self.assertEqual(read_outcomes(sidecar)[0][0]['termination'], 'abort')
        explicit[0]['completion_tokens'] = 1.5
        sidecar.write_text(json.dumps(explicit))
        sidecar.with_suffix('.manifest.json').write_text(json.dumps({'sha256':sha(sidecar)}))
        with self.assertRaisesRegex(ValueError, 'completion_tokens'):
            read_outcomes(sidecar)

    def test_outcome_and_cohort_checksums_fail_closed(self):
        manifests, _, _ = self.build(outcomes_path=self.outpath)
        self.assertEqual(manifests[0]['cohort_model'], 'gpt')
        self.outcomes.append(dict(self.outcomes[0]))
        self.write_outcomes()
        with self.assertRaisesRegex(ValueError, 'Duplicate outcome identity'):
            read_outcomes(self.outpath)
        self.outcomes.pop()
        self.write_outcomes()
        self.outcomes[0]['adjudicated_correct'] = False
        self.outpath.write_text(json.dumps(self.outcomes))
        with self.assertRaisesRegex(ValueError, 'sidecar checksum'):
            self.build(outcomes_path=self.outpath)
        self.write_outcomes()
        self.build(outcomes_path=self.outpath)

    def test_same_identity_different_content_is_not_a_join(self):
        self.outcomes[0]['trace_sha256'] = 'f'*64
        self.write_outcomes()
        with self.assertRaisesRegex(ValueError, 'Outcome trace identity differs'):
            self.build(outcomes_path=self.outpath)
        self.outcomes[0]['trace_sha256'] = trace_content_hash(self.traces[0])
        self.outcomes[0]['completion_tokens'] += 1
        self.write_outcomes()
        with self.assertRaisesRegex(ValueError, 'completion tokens differ'):
            self.build(outcomes_path=self.outpath)

    def test_duplicate_membership_rejected_shared_ab_merged(self):
        spec = prepare_spec('gpt', self.spec)
        rows = self.rows + self.a_rows()
        with self.assertRaisesRegex(ValueError, 'Duplicate attempt membership'):
            build_rows({**self.cohort, 'rows':rows+self.a_rows()}, [], 'gpt', spec)
        merged = build_rows({**self.cohort, 'rows':rows}, [], 'gpt', spec)
        self.assertEqual(len(merged), 2)
        self.assertTrue(all({c['id'] for c in row['cohorts']} == {'A','B'} for row in merged))

    def test_source_token_count_and_cap_rejected(self):
        with self.assertRaisesRegex(ValueError, 'input cap'):
            self.build(spec={**self.spec, 'max_input_tokens':4000})
        self.rewrite_cohort([{**self.rows[0], 'completion_tokens':self.rows[0]['completion_tokens']+1}, self.rows[1]])
        with self.assertRaisesRegex(ValueError, 'token coverage'):
            self.build()

    def test_shards_obey_runtime_bound_and_keep_atomic_traces(self):
        rows = [{'id':str(i),'completion_tokens':10000} for i in range(9)]
        shards = token_budget_shards(rows, 1, 100, 0)
        self.assertEqual(sum(len(s['attempts']) for s in shards), 9)
        self.assertTrue(all(s['requested_seconds'] <= 16*3600 for s in shards))
        self.assertTrue(all(s['requested_seconds'] >= 1.25*(100+s['tokens']) for s in shards))
        with self.assertRaisesRegex(ValueError, 'One attempt'):
            token_budget_shards([{'id':'big','completion_tokens':100000}], 1, 100, 0)
        # A fixed per-attempt cost consumes the envelope even when tokens fit.
        with_attempt_cost = token_budget_shards([{'id':'a','completion_tokens':1}]*8, .001, 10, 8000)
        self.assertGreater(len(with_attempt_cost), 1)
        self.assertTrue(all(s['requested_seconds'] <= 16*3600 for s in with_attempt_cost))

    def test_runtime_fit_uses_fixed_plus_per_token_cost(self):
        # Attempt seconds exclude startup, so startup never enters the intercept.
        runtime = fit_attempt_runtime([{'tokens':1000,'seconds':220.},{'tokens':11000,'seconds':620.}], 120.)
        self.assertAlmostEqual(runtime['seconds_per_token'], .04, places=6)
        self.assertAlmostEqual(runtime['per_attempt_seconds'], 180., places=6)
        self.assertEqual(runtime['startup_seconds'], 120.)
        self.assertFalse(runtime['conservative'])
        self.assertEqual(fit_attempt_runtime([{'tokens':1000,'seconds':220.},{'tokens':11000,'seconds':620.}], 5000.)
                         ['per_attempt_seconds'], runtime['per_attempt_seconds'])
        clamped = fit_attempt_runtime([{'tokens':1000,'seconds':1.},{'tokens':10000,'seconds':100.}], 120.)
        self.assertEqual(clamped['per_attempt_seconds'], 0.)
        self.assertTrue(clamped['conservative'])
        for point in clamped['points']:
            self.assertGreaterEqual(clamped['seconds_per_token']*point['tokens'], point['seconds'])
        with self.assertRaisesRegex(ValueError, 'distinct-token'):
            fit_attempt_runtime([{'tokens':1000,'seconds':4.}], 1.)

    def test_realistic_startup_keeps_per_attempt_cost_in_shard_sizing(self):
        # T2 re-review counterexample: startup 1500 s must not zero the per-attempt cost.
        runtime = fit_attempt_runtime([{'tokens':1000,'seconds':80.},{'tokens':131000,'seconds':3000.}], 1500.)
        self.assertGreater(runtime['per_attempt_seconds'], 50.)
        rows = [{'id':f'a{i:04d}','completion_tokens':800} for i in range(2480)]
        shards = token_budget_shards(rows, runtime['seconds_per_token'], runtime['startup_seconds'],
                                     runtime['per_attempt_seconds'])
        self.assertGreater(len(shards), 3)
        for shard in shards:
            true_cost = 1500. + sum(runtime['per_attempt_seconds'] + runtime['seconds_per_token']*800
                                    for _ in shard['attempts'])
            self.assertGreaterEqual(shard['requested_seconds'], true_cost)
            self.assertLessEqual(shard['requested_seconds'], 16*3600)

    def test_verification_receipt_drives_production_sizing(self):
        bootstrap, verifier, pending = self.build(verification_only=True, verification_seconds=18000,
            p90_seconds_per_token=None, startup_seconds=None, per_attempt_seconds=None, bytes_per_token=None)
        self.assertEqual(bootstrap, [])
        self.assertEqual(verifier['requested_seconds'], 18000)
        root = publish_manifests(self.storage, 'gpt', 'B', bootstrap, verifier, pending)
        receipt = dict(status='passed', validation_binding=verifier['validation_binding'],
                       maximum_input_tokens=verifier['maximum_input_tokens'],
                       measurements=dict(startup_seconds=120.,
                           attempts=[{'id':'short','tokens':300,'seconds':135.,'artifact_bytes':6000},
                                     {'id':'long','tokens':5000,'seconds':320.,'artifact_bytes':100000}],
                           bytes_per_token=20.))
        receipt_path = Path(verifier['verification_path'])
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt))
        manifests, measured_verifier, ready = self.build(verification_receipt_path=receipt_path,
            p90_seconds_per_token=None, startup_seconds=None, per_attempt_seconds=None, bytes_per_token=None)
        self.assertEqual(ready['status'], 'ready')
        self.assertEqual(ready['sizing']['status'], 'measured')
        self.assertGreater(ready['sizing']['per_attempt_seconds'], 0)
        self.assertEqual(ready['verification_receipt_binding']['sha256'], sha(receipt_path))
        expected = 1.25*(ready['sizing']['startup_seconds'] + ready['sizing']['per_attempt_seconds']*2
                         + ready['sizing']['seconds_per_token']*5300)
        self.assertTrue(all(m['requested_seconds'] >= expected for m in manifests))
        publish_manifests(self.storage, 'gpt', 'B', manifests, measured_verifier, ready)
        receipt['validation_binding'] = 'other'
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'does not bind'):
            self.build(verification_receipt_path=receipt_path,
                p90_seconds_per_token=None, startup_seconds=None, per_attempt_seconds=None, bytes_per_token=None)
        self.assertTrue(root.is_dir())

    def test_duplicate_report_states_separate_plan_cost(self):
        self.rewrite_cohort(self.rows + self.a_rows())
        report = cohort_duplicate_report(self.cohort)
        self.assertEqual(report['shared_attempts'], 2)
        self.assertEqual(report['duplicated_completion_tokens'], 5300)
        separate, _, separate_ready = self.build(selection='A', verification_only=True, verification_seconds=18000,
            p90_seconds_per_token=None, startup_seconds=None, per_attempt_seconds=None, bytes_per_token=None)
        self.assertEqual(separate, [])
        self.assertEqual(separate_ready['dedup']['duplicated_completion_tokens'], 5300)
        _, _, combined_ready = self.build(selection='all', verification_only=True, verification_seconds=18000,
            p90_seconds_per_token=None, startup_seconds=None, per_attempt_seconds=None, bytes_per_token=None)
        self.assertEqual(combined_ready['attempts'], 2)
        self.assertEqual(combined_ready['dedup']['union_attempts'], 2)

    def test_input_cap_and_pending_checkpoint(self):
        with self.assertRaisesRegex(ValueError, 'pending resolution'):
            prepare_spec('qwen35', {'checkpoint':None, 'revision':None})
        with self.assertRaisesRegex(ValueError, 'family|GPTQ'):
            prepare_spec('qwen35', self.spec)
        with self.assertRaisesRegex(ValueError, 'tokenizer'):
            prepare_spec('gpt', {**self.spec, 'tokenizer_files':{}})
        with self.assertRaisesRegex(ValueError, 'topology'):
            prepare_spec('gpt', {**self.spec, 'layers':[1]})

    def test_storage_and_immutable_publish(self):
        with self.assertRaisesRegex(ValueError, 'campaign-v3'):
            validate_storage_root(self.root/'old-campaign')
        with self.assertRaisesRegex(ValueError, 'campaign-v3'):
            validate_storage_root('/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/other/campaign-v3')
        self.assertEqual(str(validate_storage_root('/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3/capture')),
                         '/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3/capture')
        manifests, verification, ready = self.build()
        root = publish_manifests(self.storage, 'gpt', 'B', manifests, verification, ready)
        self.assertEqual(json.loads((root/'ready.json').read_text())['attempts'], 2)
        publish_manifests(self.storage, 'gpt', 'B', manifests, verification, ready)
        with self.assertRaisesRegex(ValueError, 'differs'):
            publish_manifests(self.storage, 'gpt', 'B', manifests, verification, {**ready,'status':'changed'})

    def test_bootstrap_has_no_production_shards_and_preserves_verifier_on_upgrade(self):
        options = dict(verification_only=True, verification_seconds=18000,
                       p90_seconds_per_token=None, startup_seconds=None, per_attempt_seconds=None, bytes_per_token=None)
        manifests, verification, ready = self.build(**options)
        self.assertEqual(manifests, [])
        self.assertEqual(ready['status'], 'needs_measurement')
        self.assertIsNone(verification['seconds_per_token'])
        self.assertEqual(verification['requested_seconds'], 18000)
        root = publish_manifests(self.storage, 'gpt', 'B', manifests, verification, ready)
        original = (root.parent/'verify.json').read_bytes()
        measured, measured_verify, measured_ready = self.build()
        self.assertEqual(verification['validation_binding'], measured_verify['validation_binding'])
        publish_manifests(self.storage, 'gpt', 'B', measured, measured_verify, measured_ready)
        self.assertEqual((root.parent/'verify.json').read_bytes(), original)
        self.assertEqual(json.loads((root/'ready.json').read_text())['status'], 'ready')
        self.assertTrue((root/'0.json').is_file())
        publish_manifests(self.storage, 'gpt', 'B', measured, measured_verify, measured_ready)
        self.assertEqual((root.parent/'verify.json').read_bytes(), original)
        with self.assertRaisesRegex(ValueError, 'differs'):
            publish_manifests(self.storage, 'gpt', 'B', manifests, verification, ready)

    def test_bootstrap_requires_explicit_bounded_walltime(self):
        for seconds in (None, 0, 3601, 7200, 17*3600):
            with self.assertRaisesRegex(ValueError, 'erification walltime'):
                self.build(verification_only=True, verification_seconds=seconds,
                           p90_seconds_per_token=None, startup_seconds=None, per_attempt_seconds=None, bytes_per_token=None)

    def test_short_verifier_includes_prompt_in_32k_limit(self):
        rows = [{'id':'long','total_input_tokens':40000,'completion_tokens':5000},
                {'id':'short-completion-long-prompt','total_input_tokens':35000,'completion_tokens':300}]
        with self.assertRaisesRegex(ValueError, 'total input'):
            verification_pair(rows)
        rows.append({'id':'short','total_input_tokens':1000,'completion_tokens':300})
        self.assertEqual([r['id'] for r in verification_pair(rows)], ['long','short'])
        # the longest eligible short is preferred (>=~4k tokens for the dynamics differential)
        rows.append({'id':'short-4k','total_input_tokens':4300,'completion_tokens':4000})
        self.assertEqual([r['id'] for r in verification_pair(rows)], ['long','short-4k'])

    def fake_gate_fixture(self):
        """Exercise metadata/marker handshakes only; payloads are NOT native tensors."""
        bundle = self.root/'bundle'
        bundle.mkdir()
        (bundle/'fixture.py').write_text('# CPU metadata fixture\n')
        (bundle/'SHA256.json').write_text(json.dumps({'fixture.py':sha(bundle/'fixture.py')}))
        code_binding = sha(bundle/'SHA256.json')
        bootstrap, verifier, pending = self.build(code_binding=code_binding, verification_only=True,
            verification_seconds=18000, p90_seconds_per_token=None, startup_seconds=None,
            per_attempt_seconds=None, bytes_per_token=None)
        root = publish_manifests(self.storage, 'gpt', 'B', bootstrap, verifier, pending)
        original_verifier = (root.parent/'verify.json').read_bytes()
        receipt = dict(status='passed', validation_binding=verifier['validation_binding'],
                       manifest_binding=verifier['binding'], bundle_sha256=code_binding,
                       maximum_input_tokens=verifier['maximum_input_tokens'],
                       all_routed_layers=verifier['spec']['layers'], exact_saved_token_coverage=True,
                       unchanged_resume_verified=True, differential={'status':'passed'},
                       verified_ids=[row['id'] for row in verifier['attempts']])
        receipt_path = Path(verifier['verification_path'])
        receipt_path.parent.mkdir(parents=True)
        receipt_path.write_text(json.dumps(receipt))
        manifests, measured_verifier, ready = self.build(code_binding=code_binding)
        publish_manifests(self.storage, 'gpt', 'B', manifests, measured_verifier, ready)
        self.assertEqual((root.parent/'verify.json').read_bytes(), original_verifier)
        for manifest in manifests:
            output = Path(manifest['capture_dir'])
            output.mkdir(parents=True)
            records = []
            for row in manifest['attempts']:
                files = []
                for extension in ('pt.gz','json.gz'):
                    artifact = output/(row['id']+'.'+extension)
                    artifact.write_bytes(b'metadata-handshake-only: not a native tensor')
                    files.append(dict(path=str(artifact),bytes=artifact.stat().st_size,sha256=sha(artifact)))
                record = dict(id=row['id'], trace_sha256=row['trace_sha256'],
                              manifest_binding=manifest['binding'], tokens=row['completion_tokens'], files=files)
                (output/(row['id']+'.complete.json')).write_text(json.dumps(record))
                records.append(record)
            result = dict(status='complete_native_collection',manifest_binding=manifest['binding'],
                          tokens=sum(r['tokens'] for r in records), records=records,
                          artifact_bytes=sum(info['bytes'] for r in records for info in r['files']))
            (output/'result.json').write_text(json.dumps(result))
        return root, bundle, manifests

    def test_planner_bootstrap_receipt_capture_marker_gate_handshake(self):
        from measurement_study.capture_gate import gate
        root, bundle, manifests = self.fake_gate_fixture()
        result = gate(root, bundle)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['attempts'], 2)
        self.assertEqual(result['tokens'], 5300)
        self.assertEqual(len(result['results']), len(manifests))
        document = json.loads(self.cohort_path.read_text())
        document['rows'][0]['trace_sha256'] = '0'*64
        self.cohort_path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, 'changed since planning|manifest_sha256'):
            gate(root, bundle)

    def test_handshake_gate_rejects_unready_and_missing_shards(self):
        from measurement_study.capture_gate import gate
        root, bundle, _ = self.fake_gate_fixture()
        ready_path = root/'ready.json'
        ready = json.loads(ready_path.read_text())
        ready_path.write_text(json.dumps({**ready,'status':'needs_measurement'}))
        with self.assertRaisesRegex(ValueError, 'planning is incomplete'):
            gate(root, bundle)
        ready_path.write_text(json.dumps(ready))
        (root/'0.json').unlink()
        with self.assertRaisesRegex(ValueError, 'missing capture manifests'):
            gate(root, bundle)
        ready_path.unlink()
        with self.assertRaises(FileNotFoundError):
            gate(root, bundle)

    def test_handshake_gate_rejects_missing_attempt_marker(self):
        from measurement_study.capture_gate import gate
        root, bundle, manifests = self.fake_gate_fixture()
        manifest = manifests[0]
        (Path(manifest['capture_dir'])/(manifest['attempts'][0]['id']+'.complete.json')).unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete capture shard'):
            gate(root, bundle)

    def labelled_ab_fixture(self):
        # A owns the longest trace; B alone would only expose the short trace.
        self.rewrite_cohort([self.rows[0], {**self.rows[1],'cohort':'A','inclusion_probability':.5},
                             {**self.rows[0],'cohort':'A','inclusion_probability':.25}])
        records = []
        unique = {(row['dataset'],row['problem_id'],row['sample_id']):row for row in self.rows}
        for row in unique.values():
            identity = {key:row[key] for key in ('dataset','problem_id','source_problem_id','sample_id','trace_sha256')}
            identity.update(sentence_index=0,start=0,end=7)
            record = dict(identity=identity,unit={'index':0,'start':0,'end':7,'text':'Reason.'})
            if row['sample_id'] == 0:
                record['label'] = 'Explore'
            else:
                record.update(status='unknown',raw_completion='?',failure={'kind':'fixture'})
            records.append(record)
        path = self.root/'annotations.json'
        path.write_text(json.dumps(records))
        return path, records

    def test_a_requires_labels_and_freezes_exact_inline_unknowns(self):
        from measurement_study.production_capture import load_annotation
        path, _ = self.labelled_ab_fixture()
        with self.assertRaisesRegex(ValueError, 'A production requires'):
            self.build(selection='A')
        manifests, verification, ready = self.build(selection='A',annotation_paths=[path])
        self.assertEqual(ready['annotation_binding']['inputs'][0]['sha256'], sha(path))
        self.assertIsNone(verification['annotation_binding'])
        self.assertTrue(all('annotation' not in member for row in verification['attempts'] for member in row['cohorts']))
        rows = sorted(manifests[0]['attempts'],key=lambda row:row['identity']['sample_id'])
        self.assertEqual(load_annotation(rows[0]['cohorts'][0])['units'][0]['label'], 'Explore')
        unknown = load_annotation(rows[1]['cohorts'][0])['units'][0]
        self.assertEqual(unknown['status'], 'unknown')
        self.assertNotIn('label', unknown)
        payload = rows[0]['cohorts'][0]['annotation']
        self.assertEqual(payload['sha256'], digest(payload['inline']))
        load_annotation(rows[0]['cohorts'][0])['units'][0]['label'] = 'Monitor'
        self.assertEqual(payload['inline']['units'][0]['label'], 'Explore')

    def test_inline_planner_rejects_wrong_offsets_and_missing_selected_units(self):
        path, records = self.labelled_ab_fixture()
        records[0]['identity']['start'] = 1
        records[0]['unit']['start'] = 1
        path.write_text(json.dumps(records))
        with self.assertRaisesRegex(ValueError, 'offsets/text'):
            self.build(selection='A', annotation_paths=[path])
        path.write_text(json.dumps(records[1:]))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.build(selection='A', annotation_paths=[path])

    def test_ab_share_global_label_free_verifier_when_a_is_longer(self):
        from measurement_study.campaign_v3_capture import validate_verification
        labels, _ = self.labelled_ab_fixture()
        b_empty, bootstrap, pending = self.build(selection='B',verification_only=True,verification_seconds=18000,
            p90_seconds_per_token=None,startup_seconds=None,per_attempt_seconds=None,bytes_per_token=None)
        b_root = publish_manifests(self.storage,'gpt','B',b_empty,bootstrap,pending)
        original = (b_root.parent/'verify.json').read_bytes()
        b_shards, b_verify, b_ready = self.build(selection='B')
        a_shards, a_verify, a_ready = self.build(selection='A',annotation_paths=[labels])
        self.assertEqual(b_shards[0]['expected_tokens'],300)
        self.assertEqual(a_shards[0]['expected_tokens'],5300)
        for value in (bootstrap, b_verify, a_verify, b_shards[0], a_shards[0]):
            self.assertEqual(value['maximum_input_tokens'],5002)
            self.assertEqual(value['validation_binding'],bootstrap['validation_binding'])
            self.assertEqual(value['verification_path'],str(self.storage/'capture/gpt/verify/verification.json'))
        publish_manifests(self.storage,'gpt','B',b_shards,b_verify,b_ready)
        a_root = publish_manifests(self.storage,'gpt','A',a_shards,a_verify,a_ready)
        self.assertEqual((a_root.parent/'verify.json').read_bytes(),original)
        self.assertFalse((a_root/'verify.json').exists())
        self.assertEqual(a_ready['verification_manifest'],'../verify.json')
        receipt = dict(status='passed',validation_binding=bootstrap['validation_binding'],
                       maximum_input_tokens=5002,all_routed_layers=[0,1],exact_saved_token_coverage=True,
                       unchanged_resume_verified=True,differential={'status':'passed'})
        validate_verification(a_shards[0],receipt)
        validate_verification(b_shards[0],receipt)
        self.assertEqual(bootstrap['attempts'],a_verify['attempts'])
        with self.assertRaisesRegex(ValueError, 'label-free'):
            self.build(selection='B',verification_only=True,verification_seconds=18000,annotation_paths=[labels])

    def test_unresolved_outcome_must_remain_null(self):
        self.outcomes[1]['adjudication_status'] = 'unscored'
        self.outcomes[1]['adjudicated_correct'] = False
        self.write_outcomes()
        with self.assertRaisesRegex(ValueError, 'Unresolved'):
            read_outcomes(self.outpath)

    def test_real_cohort_and_real_outcome_contract(self):
        """B1-B3 contract against the real sample_cohorts output and sidecar."""
        campaign = json.loads(PREPARED_CAMPAIGN.read_text())
        spec = campaign['sources']['glm']
        storage = self.root/'campaign-v3'
        manifests, verification, ready = build_manifests(
            cohort_path=REAL_COHORT, outcomes_path=REAL_OUTCOMES, spec=spec, source='glm',
            storage_root=storage, code_binding='frozen-code-glm', selection='A',
            verification_only=True, verification_seconds=18000,
            p90_seconds_per_token=None, startup_seconds=None, per_attempt_seconds=None, bytes_per_token=None)
        self.assertEqual(manifests, [])
        self.assertEqual(ready['status'], 'needs_measurement')
        self.assertEqual(ready['cohort_model'], 'glm')
        self.assertEqual(ready['cohort_binding']['manifest_sha256'],
                         json.loads(REAL_COHORT.read_text())['manifest_sha256'])
        self.assertEqual(ready['cohort_binding']['file_sha256'], sha(REAL_COHORT))
        self.assertEqual(verification['expected_attempts'], 2)
        self.assertEqual(validate_cohort_binding(ready['cohort_binding'])['model'], 'glm')
        publish_manifests(storage, 'glm', 'A', manifests, verification, ready)
        self.assertTrue((storage/'manifests/glm/verify.json').is_file())
        self.assertTrue((storage/'manifests/glm/A/ready.json').is_file())


if __name__ == '__main__':
    unittest.main()
