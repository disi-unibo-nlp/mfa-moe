import json
from unittest.mock import patch
from measurement_study.production_report import required_recovery_status

def test_required_model_stays_incomplete_after_successful_input_audit(tmp_path):
 target={'id':'qwen35_gptq','model':'Qwen3.5 GPTQ','saved_generations':4647,'labeling_status':'partial'}
 (tmp_path/'qwen35_gptq.json').write_text(json.dumps({'status':'inputs_audited','exact_token_versions':114,'missing_token_versions':6}))
 with patch('measurement_study.production_report.observed_capture',return_value={'status':'not_complete','shards':[]}),patch('measurement_study.artifact_layout.result_dir',return_value=tmp_path/'absent'):
  r=required_recovery_status(target,tmp_path)
 assert r['required'] and r['saved_generations']==4647
 assert r['preparation']['exact_token_versions']==114
 assert r['stages']['capture']['status']=='not_complete'
 assert r['stages']['primary']['status']=='not_complete'

def test_unbound_analysis_does_not_complete_required_model(tmp_path):
 (tmp_path/'result.json').write_text(json.dumps({'status':'complete'}))
 capture={'status':'complete','shards':[{'shard':0,'result_sha256':'accepted','manifest_binding':'bound'}]}
 with patch('measurement_study.production_report.observed_capture',return_value=capture),patch('measurement_study.artifact_layout.result_dir',return_value=tmp_path):
  r=required_recovery_status({'id':'qwen35_gptq','model':'Qwen3.5 GPTQ'},tmp_path)
 assert r['stages']['primary']['status']=='invalid_capture_binding'
 assert r['stages']['supplement']['status']=='invalid_capture_binding'
 assert r['preparation']['status']=='pending_audit'
