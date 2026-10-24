import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from measurement_study.attach_labels import assemble_annotation, count_experts_by_class, load_annotation_records
from measurement_study.production_plan import digest

STAGE = '/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp'


class AttachLabelsTests(unittest.TestCase):
    def record(self, index, label='Explore'):
        identity = dict(dataset='aime24', problem_id='q', sample_id=2, trace_sha256='bound',
                        sentence_index=index, start=index*3, end=index*3+3)
        unit = dict(index=index, start=index*3, end=index*3+3, text='abc')
        if label is None:
            return dict(identity=identity, unit=unit, status='unknown', failure={'kind':'fixture'}, raw_completion='?')
        return dict(identity=identity, unit=unit, label=label)

    def test_batch_records_keep_unknown_and_require_exact_coverage(self):
        with tempfile.TemporaryDirectory(dir=STAGE) as temp:
            path = Path(temp)/'annotations.json'
            records = [self.record(0), self.record(1, None)]
            path.write_text(json.dumps(records))
            path.with_name('summary.json').write_text(json.dumps({'status':'complete','annotations_sha256':digest(records)}))
            groups, inputs = load_annotation_records([path])
            row = dict(identity={'dataset':'aime24','problem_id':'q','sample_id':2}, trace_sha256='bound')
            member = dict(sentence_selection={'schema_version':1,'indices':[0,1]})
            annotation = assemble_annotation(row, member, groups, {'model_id':'fixture'})
            self.assertEqual(annotation['units'][1]['status'], 'unknown')
            self.assertNotIn('label', annotation['units'][1])
            self.assertEqual(len(inputs), 2)
            subset = assemble_annotation(row, {**member,'sentence_selection':{'indices':[0]}}, groups, {'model_id':'fixture'})
            self.assertEqual([unit['index'] for unit in subset['units']], [0])
            with self.assertRaisesRegex(ValueError, 'coverage'):
                assemble_annotation(row, {**member,'sentence_selection':{'indices':[0,2]}}, groups, {'model_id':'fixture'})
            repeated, _ = load_annotation_records([path,path])
            self.assertEqual(repeated, groups)
            conflict_path = Path(temp)/'other.json'
            conflict_path.write_text(json.dumps([self.record(0,'Monitor')]))
            with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
                load_annotation_records([path,conflict_path])

    def test_counts_exclude_unknown_tokens_and_preserve_layers(self):
        ids = [[[0,1],[1,2],[2,3]], [[2,3],[0,3],[1,2]]]
        sentences = [{'label':'Explore','token_indices':[0]}, {'label':None,'token_indices':[1]}, {'label':'Monitor','token_indices':[2]}]
        result = count_experts_by_class(ids, sentences, 4, layers=[2,7])
        self.assertEqual(result['tokens'], 2)
        self.assertEqual(result['counts'], [[1,1,1,1],[0,1,2,1]])
        self.assertEqual(result['classes']['Explore']['counts'], [[1,1,0,0],[0,0,1,1]])
        self.assertEqual(result['classes']['Read']['tokens'], 0)
        empty = count_experts_by_class(ids, [{'label':None,'token_indices':[0,1,2]}], 4)
        self.assertEqual(empty['tokens'], 0)
        self.assertEqual(empty['counts'], [[0]*4,[0]*4])

    def test_reject_overlap_invalid_ids_and_missing_alignment(self):
        ids = [[[0,1],[1,2]]]
        with self.assertRaisesRegex(ValueError, 'overlap'):
            count_experts_by_class(ids, [{'label':'Explore','token_indices':[0]}, {'label':'Monitor','token_indices':[0]}], 3)
        with self.assertRaisesRegex(ValueError, 'alignment'):
            count_experts_by_class(ids, [{'label':'Explore','token_indices':None}], 3)
        with self.assertRaisesRegex(ValueError, 'Invalid native'):
            count_experts_by_class([[[0,3]]], [], 3)
        with self.assertRaisesRegex(ValueError, 'Duplicate expert'):
            count_experts_by_class([[[0,0]]], [], 3)

    def test_inline_loader_is_hash_checked_and_legacy_gzip_still_reads(self):
        from measurement_study.production_capture import load_annotation
        payload = {'schema_version':1,'units':[{'index':0,'label':'Explore'}]}
        member = {'annotation':{'inline':payload,'sha256':digest(payload),'inputs':[]}}
        self.assertEqual(load_annotation(member),payload)
        returned = load_annotation(member)
        returned['units'][0]['label'] = 'Monitor'
        self.assertEqual(payload['units'][0]['label'],'Explore')
        with self.assertRaisesRegex(ValueError,'Inline annotation changed'):
            load_annotation({'annotation':{**member['annotation'],'sha256':'changed'}})
        with tempfile.TemporaryDirectory(dir=STAGE) as temp:
            raw = json.dumps(payload).encode()
            path = Path(temp)/'legacy.json.gz'
            path.write_bytes(gzip.compress(raw))
            self.assertEqual(load_annotation({'annotation':{'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}}),payload)

    def test_summary_mismatch_rejected(self):
        with tempfile.TemporaryDirectory(dir=STAGE) as temp:
            path = Path(temp)/'annotations.json'
            path.write_text(json.dumps([self.record(0)]))
            path.with_name('summary.json').write_text(json.dumps({'status':'complete','annotations_sha256':'wrong'}))
            with self.assertRaisesRegex(ValueError, 'summary binding'):
                load_annotation_records([path])


if __name__ == '__main__':
    unittest.main()
