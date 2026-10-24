import gzip,hashlib,json,os,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from measurement_study.production_plan import balanced,read_location
from measurement_study.production_capture import pack_bytes
from measurement_study.production_analyze import splits
STAGE='/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp'
class ProductionTests(unittest.TestCase):
    def test_compact_annotation_restores_only_missing_selection(self):
        from moe_exp.schemas import TraceRecord
        from measurement_study.production_plan import normalize_annotation
        from moe_exp.correlation_pipeline.spans import trace_digest, sentence_spans
        trace=TraceRecord(prompt='x',gold_answer='1',model_answer='1',dataset='math500',problem_id='1',sample_id=0,model_id='fixture',cot_text='<think>First statement. Second statement.</think>',metadata={})
        units=sentence_spans(trace)
        trace.metadata['sentence_selection']={'schema_version':1,'indices':[0]}
        a=dict(schema_version=1,trace_sha256=trace_digest(trace),units=[dict(units[0],label='Analyze')])
        result=normalize_annotation(trace,a)
        self.assertEqual(result['sentence_selection'],trace.metadata['sentence_selection'])
        self.assertNotIn('sentence_selection',a)
        self.assertEqual(result['units'],a['units'])
        with self.assertRaisesRegex(ValueError,'selection mismatch'):
            normalize_annotation(trace,{**a,'sentence_selection':None})
        bad={**a,'units':[{**a['units'][0],'start':999}]}
        with self.assertRaisesRegex(ValueError,'offsets/text'):
            normalize_annotation(trace,bad)
    def test_cohort_uses_source_copy_matching_explicit_selection(self):
        from unittest.mock import patch
        from measurement_study.production_plan import bind_cohort_location
        cohort={'locations':[{'copy':0},{'copy':1}]}
        annotation={'sentence_selection':{'schema_version':1,'indices':[2]}}
        sources=[{'metadata':{}},{'metadata':annotation}]
        with patch('measurement_study.production_plan.read_location',side_effect=lambda loc:sources[loc['copy']]):
            self.assertEqual(bind_cohort_location(cohort,annotation),sources[1])
            self.assertEqual(cohort['locations'][0],{'copy':1})
            with self.assertRaisesRegex(ValueError,'No source copy'):
                bind_cohort_location(cohort,{'sentence_selection':{'indices':[7]}})
    def test_balancing_covers_each_attempt_once(self):
        rows=[{'id':str(i),'completion_tokens':i+1} for i in range(17)]
        groups=balanced(rows);ids=[r['id'] for g in groups for r in g]
        self.assertEqual(sorted(ids),sorted(r['id'] for r in rows));self.assertEqual(len(ids),len(set(ids)))
        sizes=[sum(r['completion_tokens'] for r in g) for g in groups]
        self.assertLessEqual(max(sizes)-min(sizes),17)
        self.assertEqual(groups[0][0]['id'],'16')
    def test_splits_keep_repeated_questions_together(self):
        rows=[{'dataset':'math500','identity':{'question_id':str(i)}} for i in range(11) for _ in range(3)]
        result=splits(rows);self.assertEqual(result,splits(list(reversed(rows))))
        self.assertEqual([list(result.values()).count(k) for k in ['calibration','tuning','test']],[6,2,3])
    def test_read_rejects_changed_content_with_same_stat(self):
        with tempfile.TemporaryDirectory(dir=STAGE) as d:
            p=Path(d)/'rows.jsonl';raw=b'{"x":1}\n';p.write_bytes(raw);st=p.stat()
            info=dict(path=str(p),size=st.st_size,mtime_ns=st.st_mtime_ns,byte_offset=0,line_bytes=len(raw),line_sha256=hashlib.sha256(raw).hexdigest())
            self.assertEqual(read_location(info),{'x':1})
            p.write_bytes(b'{"x":2}\n');os.utime(p,ns=(st.st_atime_ns,st.st_mtime_ns))
            with self.assertRaisesRegex(ValueError,'checksum'):read_location(info)
    def test_lossless_atomic_cap_preserves_existing_output(self):
        with tempfile.TemporaryDirectory(dir=STAGE) as d:
            p=Path(d)/'data.gz';raw=b'complete result'*100
            result=pack_bytes(p,raw,10000);self.assertEqual(gzip.decompress(p.read_bytes()),raw)
            before=p.read_bytes()
            with self.assertRaisesRegex(RuntimeError,'storage cap'):pack_bytes(p,b'new',1)
            self.assertEqual(p.read_bytes(),before);self.assertFalse(p.with_suffix('.gz.partial').exists())
            self.assertEqual(result['sha256'],hashlib.sha256(before).hexdigest())
if __name__=='__main__':unittest.main()
