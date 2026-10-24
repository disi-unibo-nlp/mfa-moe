import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import pytest

spec=importlib.util.spec_from_file_location('gate',Path(__file__).parents[2]/'scripts/retirement_live_gate.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)

def test_unknown_live_job_prevents_retirement(tmp_path):
 p=tmp_path/'proof.json';p.write_text(json.dumps({'jobs':{'123':{}}}))
 with patch.dict(gate.os.environ,{'SLURM_JOB_ID':'999'}),patch.object(gate,'live',return_value={'456'}),patch('sys.argv',['gate',str(p)]):
  with pytest.raises(ValueError,match='unaudited'):gate.main()

def test_changed_submitted_script_prevents_retirement(tmp_path):
 p=tmp_path/'proof.json';p.write_text(json.dumps({'jobs':{'123':{'script_sha256':gate.digest(b'original')}}}))
 with patch.dict(gate.os.environ,{'SLURM_JOB_ID':'999'}),patch.object(gate,'live',return_value={'123'}),patch.object(gate,'script',return_value=b'changed'),patch('sys.argv',['gate',str(p)]):
  with pytest.raises(ValueError,match='script changed'):gate.main()
