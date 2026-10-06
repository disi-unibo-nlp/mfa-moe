"""Exact pretreatment joins, unchanged contrast family, and safe CPU attachment."""
import copy
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts/experimental_resume')]
import analyze_fresh_veto_sensitivity_v2 as a
import submit_fresh_veto_sensitivity_v2 as attach


def fixtures():
    protocol=a.sealed(a.PROTOCOL)
    transitions=list(protocol['arms_by_transition'])
    rows=[]
    formats=[]
    records=[]
    for i,(transition,category) in enumerate([(transitions[0],'strict'),(transitions[0],'strict'),
                                             (transitions[1],'strict'),(transitions[1],'strict'),
                                             (transitions[0],'unknown'),(transitions[1],'prior')]):
        uid='p'+str(i)
        row={'uid':uid,'family':'f'+str(i),'transition':transition,'strict_veto_sensitivity_eligible':False}
        rows.append(row)
        votes=[{'reader':reader,'adjudicated_valid':not(category=='unknown' and reader==1),
                'adjudicated_value':None if category=='unknown' and reader==1 else category=='prior'}
               for reader in (0,1)]
        formats.append({**row,'original_strict_veto_sensitivity_eligible':False,'veto_readers':votes,
                        'format_adjudicated_veto_pair_complete':category!='unknown',
                        'format_adjudicated_strict_veto_sensitivity_eligible':category=='strict'})
        for seed in (0,1):
            for arm in protocol['arms_by_transition'][transition]:
                records.append({'uid':f'{uid}|{seed}|{arm}','prefix_uid':uid,'family':row['family'],
                    'transition':transition,'seed':seed,'arm':arm,
                    'both_positive':arm=='bias' and i%2==0,'emitted_tokens':256+seed+i,
                    'measurement_unknown':False})
    manifest={'rows':rows,'seeds':[0,1],'analysis_seed':20261004,
              'arms_by_transition':{t:[{'name':name} for name in names] for t,names in protocol['arms_by_transition'].items()},
              'analysis_scopes':['all',*transitions],
              'planned_contrasts_scoped':[{'scope':p['transition'],'left':p['a'],'right':p['b']} for p in protocol['primary_contrasts']]}
    return manifest,formats,records


class JoinTests(unittest.TestCase):
    def test_unknown_status_is_distinct_and_all_arms_seeds_retained(self):
        manifest,formats,records=fixtures()
        status=a.start_statuses(manifest,formats)
        self.assertEqual(status[4]['veto_status'],'UNKNOWN_VETO')
        self.assertEqual(status[4]['unknown_veto_readers'],[1])
        self.assertEqual(status[5]['veto_status'],'KNOWN_PRIOR_COMPLETION')
        restricted,subset=a.subset_join(manifest,status,records)
        self.assertEqual(len(restricted['rows']),4)
        self.assertEqual(len(subset),2*2*14+2*2*10)
        self.assertEqual({r['prefix_uid'] for r in subset},{'p0','p1','p2','p3'})

    def test_missing_duplicate_or_foreign_status_uid_rejected(self):
        manifest,formats,_=fixtures()
        for wrong in (formats[:-1],formats+[formats[0]]):
            with self.assertRaisesRegex(ValueError,'UID'):
                a.start_statuses(manifest,wrong)
        bad=copy.deepcopy(formats);bad[0]['family']='foreign'
        with self.assertRaisesRegex(ValueError,'native family'):
            a.start_statuses(manifest,bad)

    def test_UID_join_is_invariant_to_format_source_order(self):
        manifest,formats,records=fixtures()
        normal=a.start_statuses(manifest,formats)
        reordered=a.start_statuses(manifest,list(reversed(formats)))
        self.assertEqual(normal,reordered)
        original_subset=a.subset_join(manifest,normal,records)
        reordered_subset=a.subset_join(manifest,reordered,records)
        self.assertEqual(original_subset,reordered_subset)

    def test_manifest_validates_runner_hash_order_without_changing_enrollment(self):
        manifest,formats,records=fixtures()
        protocol=a.sealed(a.PROTOCOL)
        for row in manifest['rows']:
            row.update(prompt_ids_sha256='prompt',prefix_ids_sha256='prefix')
        enrollment={'sha256':a.veto.ENROLLMENT_SHA,'rows':copy.deepcopy(manifest['rows'])}
        manifest.update(schema='overnight-routing-manifest-v2',source_enrollment_sha256=enrollment['sha256'],
            source_enrollment_path=str(a.veto.enrollment.OUT),horizon=1024,bootstrap_replicates=50000,
            transitions=list(protocol['arms_by_transition']))
        manifest['rows']=sorted(manifest['rows'],key=lambda r:(manifest['transitions'].index(r['transition']),a.shared.base.digest(['overnight-v2-enrollment',r['uid']])))
        a.validate_manifest(manifest,enrollment,protocol)
        manifest['rows']=list(reversed(manifest['rows']))
        with self.assertRaisesRegex(ValueError,'deterministic request order'):
            a.validate_manifest(manifest,enrollment,protocol)

    def test_original_flags_and_vote_unknowns_cannot_be_rewritten(self):
        manifest,formats,_=fixtures()
        bad=copy.deepcopy(formats);bad[0]['original_strict_veto_sensitivity_eligible']=True
        with self.assertRaisesRegex(ValueError,'original flag'):
            a.start_statuses(manifest,bad)
        bad=copy.deepcopy(formats);bad[4]['veto_readers'][1]['adjudicated_value']=False
        with self.assertRaisesRegex(ValueError,'malformed'):
            a.start_statuses(manifest,bad)

    def test_complete_assignment_validation_precedes_subset(self):
        manifest,formats,records=fixtures();status=a.start_statuses(manifest,formats)
        # Even an omitted outcome in the excluded unknown-start group must fail.
        omitted=[r for r in records if r['uid']!='p4|1|bias']
        with self.assertRaisesRegex(ValueError,'missing assigned arm'):
            a.subset_join(manifest,status,omitted)
        bad=copy.deepcopy(records);bad[-1]['family']='wrong'
        with self.assertRaisesRegex(ValueError,'exact native status'):
            a.subset_join(manifest,status,bad)
        with self.assertRaisesRegex(ValueError,'duplicate outcome'):
            a.subset_join(manifest,status,records+[records[0]])

    def test_28_contrasts_and_separate_token_family_invariant(self):
        manifest,formats,records=fixtures();original=copy.deepcopy((manifest,formats,records))
        status=a.start_statuses(manifest,formats)
        restricted,subset=a.subset_join(manifest,status,records)
        expected=a.frozen.frozen_pairs(manifest)
        self.assertEqual(a.frozen.frozen_pairs(restricted),expected)
        semantic=a.frozen.clustered_contrasts(restricted,subset,n_boot=100)
        tokens=a.frozen.clustered_contrasts(restricted,subset,('emitted_tokens',),n_boot=100)
        self.assertEqual(semantic['multiplicity'],28)
        self.assertEqual(tokens['multiplicity'],28)
        self.assertEqual(len(semantic['contrasts']),28)
        self.assertEqual(len(tokens['contrasts']),28)
        self.assertEqual((manifest,formats,records),original)

    def test_empty_transition_keeps_28_comparisons(self):
        manifest,formats,records=fixtures()
        for row in formats[2:4]:
            row['veto_readers'][0]['adjudicated_value']=True
            row['format_adjudicated_strict_veto_sensitivity_eligible']=False
        restricted,subset=a.subset_join(manifest,a.start_statuses(manifest,formats),records)
        result=a.frozen.clustered_contrasts(restricted,subset,n_boot=100)
        self.assertEqual(result['multiplicity'],28)
        unsupported=[r for r in result['contrasts'] if r['scope']=='approach_to_commit']
        self.assertEqual(len(unsupported),11)
        self.assertTrue(all(r['precision_status']=='INSUFFICIENT_FAMILIES' for r in unsupported))


def response(out='',err='',code=0):
    return subprocess.CompletedProcess([],code,out,err)


class DependencyTests(unittest.TestCase):
    def test_live_job_gets_afterok_without_artifact_read(self):
        check=mock.Mock()
        args,proof=attach.dependency('123',check,run=mock.Mock(return_value=response('JobId=123 UserId=lmolfett(42) JobState=RUNNING')))
        self.assertEqual(args,['--dependency=afterok:123'])
        self.assertEqual(proof['mode'],'LIVE_AFTEROK');check.assert_not_called()

    def test_completed_live_job_requires_artifact_and_no_stale_dependency(self):
        check=mock.Mock(return_value={'sha256':'bound'})
        args,proof=attach.dependency('123',check,run=mock.Mock(return_value=response('JobId=123 UserId=lmolfett(42) JobState=COMPLETED ExitCode=0:0')))
        self.assertEqual(args,[]);check.assert_called_once_with()
        self.assertEqual(proof['completed_artifacts'],{'sha256':'bound'})

    def test_purged_job_requires_exact_accounting_and_artifacts(self):
        run=mock.Mock(side_effect=[response(err='Invalid job id specified',code=1),response('123|COMPLETED|0:0|lmolfett|\n')])
        args,proof=attach.dependency('123',lambda:{'sha256':'bound'},run=run)
        self.assertEqual(args,[])
        self.assertEqual(proof['mode'],'PURGED_COMPLETED_ARTIFACT_VERIFIED')
        self.assertEqual(run.call_args_list[1].args[0][0],'sacct')

    def test_failed_or_foreign_parent_is_never_rebound(self):
        for output in ('123|FAILED|1:0|lmolfett|\n','123|COMPLETED|0:0|other|\n','124|COMPLETED|0:0|lmolfett|\n'):
            run=mock.Mock(side_effect=[response(code=1),response(output)])
            with self.assertRaisesRegex(ValueError,'successful accounting'):
                attach.dependency('123',lambda:{'sha256':'bound'},run=run)
        with self.assertRaisesRegex(ValueError,'successfully'):
            attach.dependency('123',lambda:{'sha256':'bound'},run=mock.Mock(return_value=response('JobId=123 UserId=lmolfett(42) JobState=FAILED ExitCode=1:0')))

    def test_completed_without_artifacts_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'bound artifacts'):
            attach.dependency('123',lambda:{},run=mock.Mock(return_value=response('JobId=123 UserId=lmolfett(42) JobState=COMPLETED ExitCode=0:0')))


if __name__=='__main__':
    unittest.main()
