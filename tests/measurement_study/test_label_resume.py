"""Transport regression checks for the isolated labeling resume bundle."""
import importlib.util
import io
import json
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

FILE=Path(__file__).resolve().parents[2]/"results/measurement-study/label-resume-v1/code/src/moe_exp/correlation_pipeline/batch_judge.py"
spec=importlib.util.spec_from_file_location("isolated_batch_judge",FILE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class ResumeTransportTests(unittest.TestCase):
    def test_context_error_is_diagnostic_and_redacts_key(self):
        key="private-test-key"
        err=urllib.error.HTTPError("http://127.0.0.1/v1/chat/completions/batch",400,"Bad Request",{},io.BytesIO(json.dumps({"error":{"message":"prompt exceeds context "+key}}).encode()))
        with patch.object(module.urllib.request,"urlopen",side_effect=err):
            with self.assertRaises(RuntimeError) as caught:module.post_batch({"messages":[]},base_url="http://127.0.0.1/v1",api_key=key)
        self.assertIn("Batch HTTP 400",str(caught.exception))
        self.assertIn("prompt exceeds context",str(caught.exception))
        self.assertNotIn(key,str(caught.exception))
    def test_success_preserves_response_and_auth_transport(self):
        result={"choices":[{"index":0,"message":{"content":"Verify"}}]}
        with patch.object(module.urllib.request,"urlopen",return_value=io.BytesIO(json.dumps(result).encode())) as call:
            actual=module.post_batch({"model":"judge","messages":[]},base_url="http://127.0.0.1:43500/v1",api_key="test")
        self.assertEqual(actual,result)
        req=call.call_args.args[0]
        self.assertEqual(req.full_url,"http://127.0.0.1:43500/v1/chat/completions/batch")
        self.assertEqual(req.get_header("Authorization"),"Bearer test")
        self.assertEqual(json.loads(req.data),{"model":"judge","messages":[]})
    def test_network_failure_stays_fatal(self):
        with patch.object(module.urllib.request,"urlopen",side_effect=urllib.error.URLError("connection refused")):
            with self.assertRaises(urllib.error.URLError):module.post_batch({},base_url="http://127.0.0.1/v1",api_key="test")

if __name__=="__main__":unittest.main()
