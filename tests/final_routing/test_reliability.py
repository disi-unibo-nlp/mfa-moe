import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from moe_exp.final_routing import common
from moe_exp.final_routing.budget import admit, reader_price, reader_hours, preparation
from moe_exp.final_routing.scheduler import Slurm, card


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        # Tests stay inside the user's explicitly authorized project staging area.
        staging = common.WORK/'tools/tmp'
        staging.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='final-routing-test-', dir=staging)
        self.root = Path(self.temp.name)
        self.config = {'root': str(self.root), 'config_file_sha256': 'frozen', 'environment': '/env',
            'budget': {'ceiling_hours': 42050, 'margin': 1.25, 'closeout_hours': 40, 'preparation_hours': 1400},
            'slurm': {'account': 'iscrc_miosr', 'partition': 'boost_usr_prod', 'qos': 'normal',
                      'gpu': {'gpus': 2, 'cpus': 16, 'memory_gib': 120, 'billing_units': 16},
                      'cpu': {'gpus': 0, 'cpus': 4, 'memory_gib': 24, 'billing_units': 4},
                      'validation_minutes': 120, 'qualification_minutes': 240,
                      'reader_pilot_minutes': 120, 'production_minutes': 360}}

    def tearDown(self):
        self.temp.cleanup()

    def test_ambiguous_admission_is_never_redrawn_and_raw_output_recovers(self):
        units = common.Units(self.root/'units')
        row = {'uid': common.digest('unit')}
        self.assertTrue(units.intent(row, '123'))
        self.assertEqual(units.state(row['uid'])['status'], 'unknown')
        self.assertFalse(units.intent(row, '124'))
        units.admitted(row, '123')
        units.output(row, {'token_ids': [1, 2], 'finish_reason': 'stop'})
        self.assertEqual(units.state(row['uid'])['status'], 'output')
        from moe_exp.final_routing.runtime import recover_outputs
        recover_outputs(units, [row])
        self.assertEqual(units.state(row['uid'])['status'], 'complete')
        self.assertFalse(units.intent(row, '125'))

    def test_atomic_publication_rejects_conflicting_results(self):
        p = self.root/'receipt.json'
        common.publish(p, {'result': 1})
        common.publish(p, {'result': 1})
        with self.assertRaisesRegex(ValueError, 'conflict'):
            common.publish(p, {'result': 2})

    def test_batch_price_does_not_divide_by_shortest_output_and_tails_are_full_batches(self):
        profile = reader_price([{'status': 'complete', 'batch_size': 64, 'seconds': 110.087,
            'load_seconds': 402.744, 'input_bin': 0, 'input_min': 1, 'input_max': 10000,
            'max_output_tokens': 16384}])
        cost = reader_hours({'0': 65}, profile, allocations=1)
        self.assertAlmostEqual(cost, (2*110.087+402.744)/3600*16)

    def test_fallback_and_hard_cap(self):
        full = {'profiles_complete': True, 'base_hours': {'generation': 900}}
        fallback = {'profiles_complete': True, 'base_hours': {'generation': 500}}
        result = admit(self.config, {'full': full, 'fallback': fallback}, actual_hours=0,
                       committed_hours=0, available_hours=1000)
        self.assertEqual(result['mode'], 'fallback')
        result = admit(self.config, {'full': full, 'fallback': fallback}, actual_hours=0,
                       committed_hours=0, available_hours=100)
        self.assertEqual(result['status'], 'not_admitted')
        self.assertFalse(result['production_released'])
        self.assertEqual(preparation(self.config)['margin_inclusive_hours'], 1400)

    def test_all_dependencies_afterany_duplicate_submit_only_once(self):
        c = card(self.config, 'validate')
        calls = []
        def run(args):
            calls.append(args)
            if args[0] == 'sbatch':
                return '123'
            return 'JobId=123 UserId=lmolfett(133943) Account=iscrc_miosr JobName='+c['job_name']+' Partition=boost_usr_prod NumCPUs=4 ReqTRES=cpu=4,billing=4 QOS=normal NumNodes=1-1 NumTasks=1 MinMemoryNode=24G TimeLimit=02:00:00'
        scheduler = Slurm(self.config, run)
        scheduler.submit(c, ['99'])
        scheduler.submit(c, ['99'])
        batches = [args for args in calls if args[0] == 'sbatch']
        self.assertEqual(len(batches), 1)
        self.assertIn('--dependency=afterany:99', batches[0])
        self.assertFalse(any(arg.startswith('--gres') for arg in batches[0]))

    def test_cpu_readback_rejects_hidden_gpu_request(self):
        c = card(self.config, 'validate')
        scheduler = Slurm(self.config, lambda args: 'JobId=123 UserId=lmolfett(133943) Account=iscrc_miosr JobName='+c['job_name']+
                          ' Partition=boost_usr_prod NumCPUs=4 ReqTRES=cpu=4,billing=4,gres/gpu=1 QOS=normal NumNodes=1 NumTasks=1 MinMemoryNode=24G TimeLimit=02:00:00')
        with self.assertRaisesRegex(ValueError, 'requests GPUs'):
            scheduler.readback('123', c)


if __name__ == '__main__':
    unittest.main()
