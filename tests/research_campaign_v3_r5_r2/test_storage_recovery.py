"""Cleanup refuses bound sources, symlinks, live jobs and unallocated execution."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moe_exp.research_campaign_v3_r5_r2.contracts import REPO

spec = importlib.util.spec_from_file_location('cache_cleanup', REPO/'scripts/research_campaign_v3_r5_r2/reclaim_storage.py')
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


class StorageRecovery(unittest.TestCase):
    def test_bound_and_symlinked_caches_are_protected(self):
        with tempfile.TemporaryDirectory(dir=REPO/'report') as directory:
            root = Path(directory)
            cache = root/'triton'; cache.mkdir()
            source = cache/'bound.py'; source.write_text('original')
            self.assertTrue(cleanup.safe_cache(cache,set()))
            self.assertFalse(cleanup.safe_cache(cache,{str(source.resolve())}))
            link = root/'torch'; link.symlink_to(cache,target_is_directory=True)
            self.assertFalse(cleanup.safe_cache(link,set()))
            scientific = root/'captures'; scientific.mkdir()
            self.assertFalse(cleanup.safe_cache(scientific,set()))

    def test_live_array_parent_blocks_all_tasks_and_step_required(self):
        self.assertTrue(cleanup.is_live('59900000_2',{'59900000_[1-9]'}))
        self.assertFalse(cleanup.is_live('59800000',{'59900000_2'}))
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'allocated Slurm step'):
                cleanup.apply(Path('/unused'))
