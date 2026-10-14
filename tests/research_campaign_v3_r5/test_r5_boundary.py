"""Import-boundary, finite recovery, scope and independent-delivery contracts."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from moe_exp.research_campaign_v3_r5 import contracts as C, controller as O
from moe_exp.research_campaign_v3_r5.recovery import DeterministicUnits, recoverable
from moe_exp.research_campaign_v3_r5.lifecycle import DeadlineExhausted, ShapeLimitation, failure_scope, exception_record


class Boundary(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=C.REPO/'report',prefix='r5-boundary-')
        self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_interrupted_score_recovers_once_stochastic_draw_never_does(self):
        row={'uid':C.digest('original'),'source':'unchanged'}
        draws=C.Units(self.root/'executions');scores=DeterministicUnits(self.root/'scores')
        self.assertTrue(draws.begin(row,'first'));self.assertFalse(draws.begin(row,'recovery'))
        self.assertTrue(scores.begin(row,'first'));self.assertTrue(scores.begin(row,'recovery'))
        self.assertFalse(scores.begin(row,'third'))
        scores.commit(row,{'status':'complete','unknown_lower':0,'unknown_upper':1})
        self.assertEqual(scores.state(row['uid'])['unknown_upper'],1)

    def test_typed_rpc_failure_preserves_numerical_diagnostics_and_scope(self):
        cause=ShapeLimitation('packed scale storage mapping differs')
        cause.maximum_absolute_error=.03125
        remote=RuntimeError('collective RPC failed');remote.campaign_worker_failures=[exception_record(cause,'compact')]
        self.assertEqual(failure_scope(remote),'shape')
        self.assertEqual(failure_scope(DeadlineExhausted('allocation exhausted')),'deadline')
        self.assertTrue(recoverable('contexts','shared',0))
        self.assertFalse(recoverable('contexts','shared',1))
        self.assertFalse(recoverable('execute','shape',0))

    def test_publication_retries_only_failed_destination_and_then_terminates(self):
        from moe_exp.research_campaign_v3_r5.publication import deliver
        report={'sha256':C.digest('report')};directory=self.root/'publication'/report['sha256']
        C.publish(directory/'LOCAL_PDF.json',{'report_sha256':report['sha256'],'phase':'memory'})
        calls=[]
        def push(campaign,local,destination):
            calls.append(destination)
            if destination=='github':raise RuntimeError('transport unavailable')
            return {'commit':'observed','remote_readback':'observed'}
        with patch('moe_exp.research_campaign_v3_r5.publication.push_snapshot',side_effect=push):
            a=deliver({'root':str(self.root)},report);b=deliver({'root':str(self.root)},report)
        self.assertEqual(calls,['overleaf','github','github'])
        self.assertEqual(a,b);self.assertEqual(a['status'],'publication_failed')
        self.assertFalse(a['scientific_progression_affected'])
        self.assertEqual(a['destinations']['overleaf']['status'],'complete')

    def test_phase_export_failure_has_explicit_terminal_disposition(self):
        from moe_exp.research_campaign_v3_r5.phases import ready
        campaign={'root':str(self.root),'models':{}}
        C.publish(self.root/'stages'/O.stage_key('phase_export','all',1)/'RESULT.json',{'status':'execution_failed'})
        self.assertFalse(ready(campaign,[],lambda *a,**k:None))
        self.assertEqual(C.load(self.root/'TERMINAL.json')['status'],'upstream_unavailable')

    def test_qwen_pilot_manifest_identity_prefers_qualified_component(self):
        from moe_exp.research_campaign_v3_r5.workers import measurement_rows
        source={'attempt_id':'source','completion_tokens':100}
        C.publish(self.root/'qwen35/SOURCES.json',{'records':[source]})
        good={'model':'qwen35','assignment':{'source':source},'uid':C.digest('good'),'reap_qualified':True}
        bad={**good,'uid':C.digest('old'),'reap_qualified':False}
        C.publish(self.root/'MEASUREMENT_COMPONENTS.json',{'records':[good,bad]})
        table=measurement_rows({'root':str(self.root),'models':{'qwen35':{'profile_sha256':'p'}},'reap_binding':'r'},'qwen35')
        self.assertEqual(table['records'][0]['uid'],good['uid'])
        self.assertNotIn('supersession',table['records'][0])

    def test_singleton_undefined_resampling_matches_sealed_receipt_exactly(self):
        from moe_exp.research_campaign_v3_r4.statistics import bootstrap_panel as oracle
        from moe_exp.research_campaign_v3_r5.statistics import bootstrap_panel
        for value in (None,0.,1.):
            records=[{'benchmark':'math500','family':'one','a':value}]
            self.assertEqual(oracle(records,['a'],panel='test:singleton'),bootstrap_panel(records,['a'],panel='test:singleton'))

    def test_final_durable_report_still_dispatches_independent_publication(self):
        campaign=C.seal({'root':str(self.root),'independent_publication':True})
        report=C.publish(self.root/'output/REPORT.json',{'SCIENCE_COMPLETE':False})
        C.publish(self.root/'output/PAPER_LOCAL.json',{'report_sha256':report['sha256']})
        C.publish(self.root/'REPORT_COMPLETE.json',{'complete':True})
        work=O.ready(campaign,None,[])
        self.assertEqual([(c['stage'],c['shard_index']) for c in work],[('publish',0)])
