import unittest
from measurement_study.gemma_cache_policy import cache_options,SENSITIVITY_ROOT,VARIANT
class CachePolicyTests(unittest.TestCase):
    def test_default_keeps_primary_unchanged(self):
        self.assertEqual(cache_options({'generation_model':'nvidia/Gemma'},'/primary'),{})
    def test_variant_requires_explicit_label_and_isolated_outputs(self):
        spec=dict(generation_model='nvidia/Gemma-4-26B-A4B-NVFP4',precision='nvfp4',kv_cache_dtype='bfloat16',replay_variant=VARIANT)
        self.assertEqual(cache_options(spec,SENSITIVITY_ROOT/'capture/gemma/3'),{'kv_cache_dtype':'bfloat16'})
        with self.assertRaisesRegex(ValueError,'isolated'):cache_options(spec,'/primary')
        with self.assertRaisesRegex(ValueError,'label'):cache_options(dict(spec,replay_variant='primary'),SENSITIVITY_ROOT)
        with self.assertRaisesRegex(ValueError,'label'):cache_options(dict(spec,precision='bfloat16'),SENSITIVITY_ROOT)
if __name__=='__main__':unittest.main()
