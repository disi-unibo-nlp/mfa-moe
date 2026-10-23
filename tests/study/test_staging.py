"""Private staging must satisfy the installed native IPC transport before loading.

These bounded CPU tests exercise real local Unix sockets, without an engine or
GPU. They reproduce the allocated long-path failure and verify its correction.
"""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moe_exp.io import WORK
from moe_exp.study.ownership import task_staging, worker


class PrivateStagingTests(unittest.TestCase):
    def setUp(self):
        """Keep test sockets and temporary data in explicit user WORK staging."""
        self.temporary = tempfile.TemporaryDirectory(dir=WORK/'tools/tmp')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_installed_ipc_helper_reproduces_long_failure_and_binds_short_work_path(self):
        import zmq
        from vllm.utils.network_utils import get_open_zmq_ipc_path
        long_root = self.root/('long-'*12)
        long_root.mkdir()
        context = zmq.Context()
        sockets = []
        try:
            with patch.dict(os.environ, {'VLLM_RPC_BASE_PATH': str(long_root)}):
                address = get_open_zmq_ipc_path()
                socket = context.socket(zmq.PUB)
                sockets.append(socket)
                with self.assertRaisesRegex(zmq.ZMQError, f'longer than {zmq.IPC_PATH_MAX_LEN}'):
                    socket.bind(address)
            with patch.dict(os.environ, {'VLLM_RPC_BASE_PATH': str(self.root)}):
                address = get_open_zmq_ipc_path()
                self.assertTrue(address.startswith('ipc://'+str(self.root)+'/'))
                self.assertLessEqual(len(address.removeprefix('ipc://').encode()), zmq.IPC_PATH_MAX_LEN)
                socket = context.socket(zmq.PUB)
                sockets.append(socket)
                socket.bind(address)
        finally:
            for socket in sockets:
                socket.close(linger=0)
            context.term()

    def test_long_private_path_rejects_before_owner_or_engine_access(self):
        root = self.root/('long-'*12)
        root.mkdir()
        with patch.dict(os.environ, {'TMPDIR': str(root), 'SLURM_JOB_ID': '7'}):
            with self.assertRaisesRegex(ValueError, 'IPC socket path limit'):
                with worker(self.root/'store', {}, run=lambda args: self.fail('native access before staging rejection')):
                    self.fail('long path admitted')

    def test_worker_pins_rpc_base_to_its_private_task_directory(self):
        with patch.dict(os.environ, {'TMPDIR': str(self.root), 'VLLM_RPC_BASE_PATH': '/tmp'}), \
                patch.object(tempfile, 'tempdir', '/tmp'):
            self.assertEqual(task_staging(), self.root)
            self.assertEqual(os.environ['VLLM_RPC_BASE_PATH'], str(self.root))
            self.assertEqual(tempfile.gettempdir(), str(self.root))


if __name__ == '__main__':
    unittest.main()
