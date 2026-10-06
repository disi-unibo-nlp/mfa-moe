"""CPU-only manifest guards for the high-cost exploratory GPU screen."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, '/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/code/s1-9a61e32f48c04c24')
DRIVER = REPO / 'scripts/experimental_resume/run_boundary_micro_screen.py'
spec = importlib.util.spec_from_file_location('boundary_micro_screen_driver', DRIVER)
micro = importlib.util.module_from_spec(spec)
spec.loader.exec_module(micro)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def example():
    freeze = json.loads(micro.FAMILY_FREEZE.read_text())
    scout = json.loads(micro.SCOUT.read_text())
    prepared = json.loads(micro.PREPARED.read_text())
    worker_prep = json.loads(micro.WORKER_PREP.read_text())
    worker_qual = json.loads(micro.WORKER_QUAL.read_text())
    rows = [{key: record[key] for key in ('uid', 'question', 'family', 'prompt_ids', 'prefix_ids')}
            for record in scout['records'][:4]]
    overlay = Path(worker_prep['overlay'])
    actions = [
        {'name': f'target_bias{dose:g}', 'transition': 'candidate_to_verify',
         'experts': [[28, [189, 9]]], 'bias': dose} for dose in (.5, 1.)]
    for index, ids in enumerate(((139, 120), (255, 133), (43, 5), (196, 24))):
        for dose in (.5, 1.):
            actions.append({'name': f'random{index}_bias{dose:g}',
                            'transition': 'candidate_to_verify',
                            'experts': [[28, list(ids)]], 'bias': dose})
    return {
        'schema': 'routing-boundary-micro-screen-v1',
        'stage': 'pilot',
        'base_tree_sha256': micro.REQUIRED_BASE_TREE,
        'driver_sha256': sha(DRIVER),
        'prefix_scout_sha256': scout['sha256'],
        'prepared_sha256': prepared['sha256'],
        'qualified_worker_sha256': worker_qual['sha256'],
        'family_freeze_sha256': freeze['sha256'],
        'code_files': {str(path): sha(path) for path in (
            DRIVER, overlay / 'moe_exp/routing_control/worker_adapter.py',
            overlay / 'moe_exp/routing_control/ordered_vllm.py')},
        'max_tokens': 256, 'seeds': [0, 1],
        'rows': rows,
        'actions': actions,
        'random_set_by_family_seed': {row['family']: [i, (i + 1) % 4]
                                      for i, row in enumerate(rows)},
        'arms': [
            {'name': 'native_a', 'policy': 'zero', 'role': 'native'},
            {'name': 'native_b', 'policy': 'zero', 'role': 'native'},
            {'name': 'target_0.5', 'policy': 'target_bias0.5', 'role': 'target'},
            {'name': 'target_1.0', 'policy': 'target_bias1', 'role': 'target'},
            {'name': 'random_0.5', 'policy': 'random_selector_bias0.5', 'role': 'random'},
            {'name': 'random_1.0', 'policy': 'random_selector_bias1', 'role': 'random'}],
        'expected_requests': len(rows) * 12,
        'expected_prefill_tokens': sum((len(r['prompt_ids']) + len(r['prefix_ids'])) * 12
                                       for r in rows),
        'maximum_decode_tokens': len(rows) * 12 * 256,
        'maximum_context_tokens': max(len(r['prompt_ids']) + len(r['prefix_ids']) + 256
                                      for r in rows),
    }


class ManifestTests(unittest.TestCase):
    def test_valid_sparse_discovery_screen(self):
        rows, actions, arms = micro.validate_manifest(example(), DRIVER)
        self.assertEqual((len(rows), len(actions), len(arms)), (4, 10, 6))

    def test_random_must_have_distinct_expert_identities(self):
        manifest = example()
        manifest['actions'][2]['experts'] = manifest['actions'][0]['experts']
        with self.assertRaisesRegex(ValueError, 'reuses target'):
            micro.validate_manifest(manifest, DRIVER)

    def test_validation_family_cannot_enter_discovery(self):
        manifest = example()
        freeze = json.loads(micro.FAMILY_FREEZE.read_text())
        family = freeze['new_parent_pools']['parent_pools']['mechanism'][0]
        manifest['rows'][0]['family'] = family
        manifest['rows'][0]['question'] = freeze['new_parent_pools']['families'][family][0]
        with self.assertRaisesRegex(ValueError, 'validation or utility'):
            micro.validate_manifest(manifest, DRIVER)

    def test_prefix_cannot_drift_from_native_trace(self):
        manifest = example()
        manifest['rows'][0]['prefix_ids'][-1] += 1
        with self.assertRaisesRegex(ValueError, 'sealed native scout'):
            micro.validate_manifest(manifest, DRIVER)

    def test_complete_price_must_match_requests(self):
        manifest = example()
        manifest['expected_prefill_tokens'] -= 1
        with self.assertRaisesRegex(ValueError, 'resource quantities'):
            micro.validate_manifest(manifest, DRIVER)

    def test_real_policy_spec_accepts_all_frozen_expert_sets(self):
        table = micro.build_policy_table(example()['actions'])
        self.assertEqual(len(table.policies), 11)
        target = table.policies[table.index_of('target_bias1')].targets
        self.assertEqual(target.experts, ((28, (9, 189)),))

    def test_real_qualified_request_builder_keeps_paired_prefixes(self):
        manifest = example()
        manifest['sha256'] = micro.digest(manifest)
        table = micro.build_policy_table(manifest['actions'])
        world = SimpleNamespace(infos={row['question']: {'question': row['question']}
                                       for row in manifest['rows']})
        cases = micro.build_requests(manifest, manifest['rows'], manifest['arms'], world, table)
        self.assertEqual(len(cases), 48)
        first = cases[:6]
        self.assertEqual(len({tuple(req.prompt) for req, _ in first}), 1)
        self.assertEqual(len({req.sampling['seed'] for req, _ in first}), 1)
        self.assertEqual({item['role'] for _, item in first}, {'native', 'target', 'random'})

    def test_driver_must_bind_qualified_overlay(self):
        manifest = example()
        path = REPO / 'src/moe_exp/routing_control/worker_adapter.py'
        qualified = next(p for p in manifest['code_files'] if p.endswith('/worker_adapter.py'))
        manifest['code_files'].pop(qualified)
        manifest['code_files'][str(path)] = sha(path)
        with self.assertRaisesRegex(ValueError, 'qualified worker overlay'):
            micro.validate_manifest(manifest, DRIVER)


if __name__ == '__main__':
    unittest.main()
