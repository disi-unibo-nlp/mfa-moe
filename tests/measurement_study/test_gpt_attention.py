import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from types import SimpleNamespace
from transformers.models.gpt_oss.modeling_gpt_oss import eager_attention_forward
from measurement_study.gpt_attention import query_chunked_attention
class GPTAttentionTests(unittest.TestCase):
    def test_native_sink_causal_and_sliding_masks_preserved(self):
        torch.manual_seed(42);q=torch.randn(1,4,17,8);k=torch.randn(1,2,17,8);v=torch.randn_like(k)
        module=SimpleNamespace(training=False,num_key_value_groups=2,sinks=torch.randn(4))
        for sliding in (None,5):
            row=torch.arange(17)[:,None];col=torch.arange(17)[None,:];allowed=col<=row
            if sliding:allowed &= col>row-sliding
            mask=torch.zeros(1,1,17,17).masked_fill(~allowed,float('-inf'))
            with torch.inference_mode():
                expected,_=eager_attention_forward(module,q,k,v,mask,8**-.5)
                actual,weights=query_chunked_attention(eager_attention_forward,4)(module,q,k,v,mask,8**-.5)
            torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6);self.assertIsNone(weights)
    def test_broadcast_mask_and_inference_guard(self):
        module=SimpleNamespace(training=False,num_key_value_groups=1,sinks=torch.ones(2))
        q=torch.ones(1,2,7,4);k=torch.ones(1,2,9,4);v=k.clone();mask=torch.zeros(1,1,1,9)
        fun=query_chunked_attention(eager_attention_forward,3)
        with self.assertRaisesRegex(RuntimeError,'inference-only'):fun(module,q,k,v,mask,.5)
        with torch.inference_mode():
            x,_=fun(module,q,k,v,mask,.5);y,_=eager_attention_forward(module,q,k,v,mask,.5)
        torch.testing.assert_close(x,y)
if __name__=='__main__':unittest.main()
