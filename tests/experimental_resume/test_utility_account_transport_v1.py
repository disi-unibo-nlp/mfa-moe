"""No-network tests of strict finite accounting transport and qualification."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))
import utility_account_transport_v1 as transport

BALANCE = 'IscrC_MIOSR 20260504 20270204 68000 21350 21350 31.4 7391 1251\n'


class TransportTests(unittest.TestCase):
    def test_checks_remote_identity_before_native_live_balance_and_uses_strict_nonpersistent_ssh(self):
        replies = [SimpleNamespace(stdout=value, stderr='', returncode=0)
                   for value in (transport.TARGET + '\n', 'lmolfett\n', BALANCE)]
        with patch.object(transport.subprocess, 'run', side_effect=replies) as run:
            balance, records = transport.read_balance()
        self.assertEqual(balance, BALANCE)
        self.assertEqual(len(records), 3)
        self.assertEqual([call.args[0][-1] for call in run.call_args_list], list(transport.REMOTE_COMMANDS))
        for call in run.call_args_list:
            self.assertEqual(call.kwargs['timeout'], 20)
            for setting in ('BatchMode=yes', 'StrictHostKeyChecking=yes', 'ControlMaster=no',
                            'ControlPath=none', 'ForwardAgent=no', 'UpdateHostKeys=no'):
                self.assertIn(setting, call.args[0])

    def test_wrong_host_auth_failure_or_wrong_user_never_reads_balance(self):
        replies = [
            [SimpleNamespace(stdout='another-host\n', stderr='', returncode=0)],
            [SimpleNamespace(stdout='', stderr='Permission denied', returncode=255)],
            [SimpleNamespace(stdout=transport.TARGET + '\n', stderr='', returncode=0),
             SimpleNamespace(stdout='foreign-user\n', stderr='', returncode=0)]]
        for sequence in replies:
            with self.subTest(sequence=sequence), patch.object(transport.subprocess, 'run', side_effect=sequence) as run:
                with self.assertRaises(ValueError):
                    transport.read_balance()
                self.assertLess(run.call_count, 3)

    def test_success_artifact_requires_exact_successful_compute_allocation_accounting(self):
        with tempfile.TemporaryDirectory(prefix='.utility-transport-test-', dir=REPO) as temporary:
            path = Path(temporary) / 'PROBE.json'
            transport.P.save(path, {'schema': 'utility-account-transport-probe-v1',
                'status': 'PASS_NATIVE_LIVE_ACCOUNT_READ', 'source_files': transport.source_files(),
                'ssh_target': transport.TARGET, 'ssh_options': transport.OPTIONS, 'user': 'lmolfett',
                'partition': 'lrd_all_viz', 'compute_hostname': 'viz16.leonardo.local', 'probe_job_id': '123'})
            for raw in ('123|FAILED|1:0|\n', '124|COMPLETED|0:0|\n', ''):
                with patch.object(transport.subprocess, 'run', return_value=SimpleNamespace(stdout=raw)):
                    with self.assertRaises(ValueError):
                        transport.validate_probe(path)
            with patch.object(transport.subprocess, 'run',
                              return_value=SimpleNamespace(stdout='123|COMPLETED|0:0|\n')):
                self.assertEqual(transport.validate_probe(path)['probe_job_id'], '123')

    def test_failed_or_login_node_probe_is_never_qualified(self):
        with tempfile.TemporaryDirectory(prefix='.utility-transport-test-', dir=REPO) as temporary:
            for index, (state, hostname) in enumerate((('FAIL_NATIVE_ACCOUNT_TRANSPORT', 'viz16'),
                                                     ('PASS_NATIVE_LIVE_ACCOUNT_READ', 'login05'))):
                path = Path(temporary) / (str(index) + '.json')
                transport.P.save(path, {'schema': 'utility-account-transport-probe-v1',
                    'status': state, 'source_files': transport.source_files(),
                    'ssh_target': transport.TARGET, 'ssh_options': transport.OPTIONS, 'user': 'lmolfett',
                    'partition': 'lrd_all_viz', 'compute_hostname': hostname, 'probe_job_id': '123'})
                with patch.object(transport.subprocess, 'run') as run:
                    with self.assertRaises(ValueError):
                        transport.validate_probe(path)
                    run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
