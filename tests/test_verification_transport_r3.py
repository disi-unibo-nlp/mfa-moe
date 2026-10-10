"""Transport repair contracts; no model import, inference or scheduler call."""
import os
from pathlib import Path
import socket
import tempfile
import time
import unittest
from uuid import uuid4
from unittest.mock import patch

from moe_exp.verification_study.contracts import file_sha, load, save, seal
import verification_study_v3_transport_r3 as transport


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.saved_tempdir = tempfile.tempdir
        self.tmp = tempfile.TemporaryDirectory(prefix="t3t-", dir=str(transport.STAGING))
        self.base = Path(self.tmp.name)
        source = self.base / "scientific.py"
        source.write_text("# frozen fixture\n")
        self.files = {str(source): file_sha(source)}
        self.cp = self.base / "CAMPAIGN.json"
        self.campaign = save(self.cp, {"schema": "fixture", "code_files": self.files,
            "runtime_sha256": "runtime", "reader_sha256": "reader", "result_root": str(self.base)})
        self.parent = seal({"schema": "verification-job-card-v3", "campaign_path": str(self.cp),
            "campaign_sha256": self.campaign["sha256"], "code_files": self.files, "stage": "run",
            "model": "gpt", "resources": {"gpus": 2, "cpus": 16, "memory_gb": 120,
                "partition": "boost_usr_prod", "walltime_seconds": 3600},
            "payload_arguments": ["run", "--campaign", str(self.cp)],
            "measured_budget": {"immutable_price": "original"}})
        self.hf = self.base / "hf"
        (self.hf / "hub").mkdir(parents=True)
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.hf_patch = patch.object(transport, "HF_ROOT", self.hf)
        self.hf_patch.start()

    def tearDown(self):
        tempfile.tempdir = self.saved_tempdir
        self.hf_patch.stop()
        self.environment.stop()
        self.tmp.cleanup()

    def reader(self):
        executor = seal({"schema": "verification-reader-executor-v3", "code_files": self.files,
            "campaign_sha256": self.campaign["sha256"], "runtime_sha256": "runtime",
            "reader_contract_sha256": "reader"})
        return seal({**self.parent, "stage": "rate", "reader_executor": executor,
            "selected_uids": ["unchanged"], "full_frame_sha256": "original-frame"})

    def extension(self):
        executor = seal({"schema": "verification-j1-executor-v3", "code_files": self.files,
            "campaign_sha256": self.campaign["sha256"], "runtime_sha256": "runtime",
            "reader_contract_sha256": "reader"})
        return seal({**self.parent, "stage": "j1-audit", "j1_executor": executor,
            "extension_files": self.files, "extension": {"kind": "j1-audit-slice",
                "parameters": {"begin": 1, "count": 99}, "selected_uids": ["original"]},
            "input_artifacts": {"original-full100": "original-hash"}})

    def test_child_environment_strips_scheduler_prefixes_only(self):
        source = {"SLURM_CPUS_PER_TASK": "4", "SRUN_CPU_BIND": "mask_cpu:bad", "SBATCH_PARTITION": "serial",
            "PMI_RANK": "0", "PMIX_NAMESPACE": "x", "PATH": "safe", "TMPDIR": "own", "HOME": "private"}
        self.assertEqual(transport.clean_submission_environment(source),
            {"PATH": "safe", "TMPDIR": "own", "HOME": "private"})
        self.assertEqual(source["SLURM_CPUS_PER_TASK"], "4")

    def test_default_child_environment_does_not_mutate_allocated_process(self):
        os.environ.update({"SLURM_JOB_ID": "123", "SBATCH_MEM_PER_NODE": "24G", "PATH": "safe"})
        child = transport.clean_submission_environment()
        self.assertEqual(child, {"PATH": "safe"})
        self.assertEqual(os.environ["SLURM_JOB_ID"], "123")

    def test_proof_whitelist_does_not_include_credentials(self):
        proof = transport.environment_proof({"TMPDIR": "/scratch_local", "HF_TOKEN": "fixture-secret",
            "HOME": "private", "AWS_SECRET_ACCESS_KEY": "fixture-secret"})
        self.assertEqual(proof["TMPDIR"], "/scratch_local")
        self.assertEqual(set(proof), set(transport.CACHE_VARIABLES))
        self.assertNotIn("fixture-secret", str(proof))

    def test_pin_replaces_python_cached_tempdir_and_every_cache(self):
        os.environ.update({"TMPDIR": "/scratch_local", "HOME": "untouched", "SLURM_JOB_ID": "123"})
        tempfile.tempdir = "/scratch_local"
        base = transport.pin_runtime("extension", "123", staging=self.base)
        self.assertEqual(tempfile.gettempdir(), str(base / "tmp"))
        self.assertEqual(os.environ["TMP"], os.environ["TMPDIR"])
        self.assertEqual(os.environ["TEMP"], os.environ["TMPDIR"])
        for variable in transport.CACHE_VARIABLES:
            self.assertTrue(Path(os.environ[variable]).is_dir(), variable)
            self.assertNotIn("scratch_local", os.environ[variable])
        self.assertEqual(os.environ["HOME"], "untouched")
        self.assertEqual(os.environ["SLURM_JOB_ID"], "123")
        self.assertEqual(base.stat().st_mode & 0o777, 0o700)

    def test_controller_cache_pin_keeps_slurm_metadata(self):
        os.environ["SLURM_CPUS_PER_TASK"] = "4"
        base = transport.pin_runtime("controller", "123", staging=self.base)
        self.assertEqual(os.environ["SLURM_CPUS_PER_TASK"], "4")
        self.assertEqual(tempfile.gettempdir(), str(base / "tmp"))

    def test_pin_rejects_foreign_runtime_and_unsafe_identity(self):
        with self.assertRaisesRegex(ValueError, "write outside"):
            transport.pin_runtime_environment("/tmp/cv3-not-approved")
        with self.assertRaisesRegex(ValueError, "unsafe"):
            transport.pin_runtime("native", "../bad", staging=self.base)

    def test_pin_rejects_symlink_cache(self):
        base = self.base / "runtime"
        base.mkdir(mode=0o700)
        (base / "tmp").symlink_to(self.hf, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            transport.pin_runtime_environment(base)

    def test_native_wrap_preserves_all_original_scientific_fields(self):
        wrapped = transport.wrap_card(self.cp, self.parent, "native")
        self.assertNotEqual(wrapped["sha256"], self.parent["sha256"])
        self.assertEqual(wrapped["transport_parent_card_sha256"], self.parent["sha256"])
        campaign, original = transport.verify_card(self.cp, wrapped, "native")
        self.assertEqual(original, self.parent)
        self.assertEqual(campaign, self.campaign)
        self.assertEqual(wrapped["measured_budget"], self.parent["measured_budget"])

    def test_reader_wrap_keeps_old_receipt_executor_and_frame(self):
        parent = self.reader()
        wrapped = transport.wrap_card(self.cp, parent, "reader")
        self.assertEqual(wrapped["reader_executor"], parent["reader_executor"])
        self.assertEqual(wrapped["selected_uids"], parent["selected_uids"])
        self.assertEqual(transport.verify_card(self.cp, wrapped, "reader")[1], parent)

    def test_j1_wrap_keeps_qualified_original100_deployment(self):
        parent = self.extension()
        wrapped = transport.wrap_card(self.cp, parent, "extension")
        self.assertEqual(wrapped["j1_executor"], parent["j1_executor"])
        self.assertEqual(wrapped["input_artifacts"], parent["input_artifacts"])
        self.assertEqual(wrapped["extension"], parent["extension"])
        self.assertEqual(transport.verify_card(self.cp, wrapped, "extension")[1], parent)

    def test_wrapping_verified_transport_card_is_idempotent(self):
        card = transport.wrap_card(self.cp, self.parent, "native")
        self.assertEqual(transport.wrap_card(self.cp, card, "native"), card)

    def test_wrapped_scientific_mutation_is_rejected(self):
        card = transport.wrap_card(self.cp, self.parent, "native")
        changed = seal({**card, "model": "foreign"})
        with self.assertRaisesRegex(ValueError, "original scientific card"):
            transport.verify_card(self.cp, changed, "native")

    def test_source_file_change_rejected(self):
        card = transport.wrap_card(self.cp, self.parent, "native")
        Path(next(iter(self.files))).write_text("# changed\n")
        with self.assertRaisesRegex(ValueError, "bound source changed"):
            transport.verify_card(self.cp, card, "native")

    def test_wrong_kind_and_resource_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, "not a semantic reader"):
            transport.wrap_card(self.cp, self.parent, "reader")
        changed = seal({**self.parent, "resources": {**self.parent["resources"], "cpus": 4}})
        with self.assertRaisesRegex(ValueError, "resource|shape"):
            transport.wrap_card(self.cp, changed, "native")

    def test_source_files_cover_all_three_explicit16cpu_shells(self):
        files = transport.source_files()
        self.assertEqual(set(files), {str(transport.SCRIPT), *(str(p) for p in transport.SHELLS.values())})
        for path in transport.SHELLS.values():
            text = path.read_text()
            self.assertIn("--cpus-per-task=16 --cpu-bind=none --export=ALL", text)
            self.assertIn("verification_study_v3_transport_r3.py", text)

    def test_all_runtime_kinds_maximum_20_digit_job_id_fit_actual_work_path(self):
        for kind, code in transport.RUNTIME_CODES.items():
            with self.subTest(kind=kind):
                base = transport.runtime_path(kind, "18446744073709551615")
                proof = transport.ipc_path_bound(base / "r")
                self.assertLessEqual(proof["socket_path_bytes_max"], 107)
                self.assertEqual(proof["socket_path_bytes_max"],
                    len(os.fsencode(base / "r" / transport.IPC_UUID)))
        with self.assertRaisesRegex(ValueError, "unsafe"):
            transport.pin_runtime("native", "1" * 21)

    def test_runtime_names_are_disjoint_by_kind_and_job(self):
        bases = {transport.pin_runtime(kind, job, staging=self.base)
            for kind in transport.RUNTIME_CODES for job in ("123", "124")}
        self.assertEqual(len(bases), 8)

    def test_ipc_byte_bound_accepts107_and_rejects108_including_non_ascii(self):
        suffix_bytes = len(os.fsencode("/" + transport.IPC_UUID))
        self.assertEqual(transport.ipc_path_bound("/" + "a" * (106 - suffix_bytes))["socket_path_bytes_max"], 107)
        with self.assertRaisesRegex(ValueError, "108"):
            transport.ipc_path_bound("/" + "a" * (107 - suffix_bytes))
        # UTF-8 bytes are the kernel limit, not Python's character count.
        with self.assertRaisesRegex(ValueError, "exceeds"):
            transport.ipc_path_bound("/" + "é" * 36)

    def test_old122_byte_ipc_runtime_is_rejected_before_changes_or_import(self):
        old = transport.STAGING / "cv3-transport-r2-extension-59652408"
        os.environ["TMPDIR"] = "/scratch_local"
        with patch.object(Path, "mkdir") as mkdir:
            with self.assertRaisesRegex(ValueError, "IPC socket path"):
                transport.pin_runtime_environment(old)
        mkdir.assert_not_called()
        self.assertEqual(os.environ["TMPDIR"], "/scratch_local")

    def test_actual_bounded_local_unix_socket_bind_and_cleanup(self):
        base = transport.pin_runtime("extension", "123", staging=self.base)
        path = base / "r" / str(uuid4())
        transport.ipc_path_bound(path.parent)
        handle = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            handle.bind(str(path))
            self.assertTrue(path.exists())
        finally:
            handle.close()
            if path.exists():
                path.unlink()
        self.assertFalse(path.exists())

    def test_actual_bounded_local_zmq_ipc_bind_and_cleanup(self):
        import zmq
        base = transport.pin_runtime("reader", "123", staging=self.base)
        path = base / "r" / str(uuid4())
        self.assertLessEqual(len(os.fsencode(path)), 107)
        context = zmq.Context()
        handle = context.socket(zmq.PAIR)
        try:
            handle.bind("ipc://" + str(path))
            self.assertTrue(path.exists())
        finally:
            handle.close(linger=0)
            context.term()
            if path.exists():
                path.unlink()
        self.assertFalse(path.exists())

    def test_worker_arguments_preserve_original_main_and_deadline(self):
        path = self.base / "JOB_CARD.json"
        deadline = time.time() + 600
        native = transport.worker_arguments(path, "native", deadline)
        extension = transport.worker_arguments(path, "extension", deadline)
        self.assertEqual(native[:2], [str(transport.WORKERS["native"]), str(path)])
        self.assertEqual(extension[:4], [str(transport.WORKERS["extension"]), "execute", "--job-card", str(path)])
        self.assertEqual(native[-2:], ["--deadline-epoch", str(deadline)])
        with self.assertRaisesRegex(ValueError, "deadline"):
            transport.worker_arguments(path, "native", time.time() - 1)

    def test_execute_records_entering_values_before_pinned_worker(self):
        card = transport.wrap_card(self.cp, self.parent, "native")
        path = self.base / "JOB_CARD.json"
        save(path, card)
        os.environ.update({"SLURM_JOB_ID": "123", "SLURM_STEP_ID": "0", "TMPDIR": "/scratch_local"})
        def pin(kind, job):
            return transport.pin_runtime_environment(self.base / "runtime")
        def worker(*args, **kwargs):
            self.assertEqual(tempfile.gettempdir(), str(self.base / "runtime" / "tmp"))
            self.assertEqual(kwargs["run_name"], "__main__")
            self.assertEqual(sys_argv_path(), str(path))
        with patch.object(transport, "require_step"), patch.object(transport, "pin_runtime", side_effect=pin), \
             patch.object(transport.runpy, "run_path", side_effect=worker):
            result = transport.execute(path, "native", time.time() + 600)
        self.assertEqual(result["status"], "WORKER_RETURNED")
        slot = next((self.base / "transport" / "invocations").iterdir())
        self.assertEqual(load(slot / "ENTERING_STEP.json")["cache_environment"]["TMPDIR"], "/scratch_local")
        self.assertEqual(load(slot / "START.json")["python_tempdir"], str(self.base / "runtime" / "tmp"))
        self.assertIn("REQUIRES_ORIGINAL", result["scientific_success"])

    def test_execute_failure_records_full_traceback_and_reraises(self):
        card = transport.wrap_card(self.cp, self.parent, "native")
        path = self.base / "JOB_CARD.json"
        save(path, card)
        os.environ.update({"SLURM_JOB_ID": "123", "SLURM_STEP_ID": "0"})
        with patch.object(transport, "require_step"), \
             patch.object(transport, "pin_runtime", side_effect=lambda *a: transport.pin_runtime_environment(self.base / "runtime")), \
             patch.object(transport.runpy, "run_path", side_effect=PermissionError("diagnostic fixture")):
            with self.assertRaisesRegex(PermissionError, "diagnostic fixture"):
                transport.execute(path, "native", time.time() + 600)
        slot = next((self.base / "transport" / "invocations").iterdir())
        error = load(slot / "ERROR.json")["error"]
        self.assertEqual(error["type"], "PermissionError")
        self.assertIn("Traceback", error["traceback"])
        self.assertIn("PermissionError: diagnostic fixture", error["traceback"])
        self.assertFalse((slot / "RESULT.json").exists())

    def test_compute_guard_rejects_login_and_batch_only(self):
        with patch.object(transport.getpass, "getuser", return_value="lmolfett"), \
             patch.object(transport.socket, "gethostname", return_value="login05.leonardo.local"):
            with self.assertRaisesRegex(ValueError, "compute node"):
                transport.require_step(self.parent)
        os.environ.update({"SLURM_JOB_ID": "123", "SLURM_STEP_ID": "batch"})
        with patch.object(transport.getpass, "getuser", return_value="lmolfett"), \
             patch.object(transport.socket, "gethostname", return_value="lrdn0087.leonardo.local"):
            with self.assertRaisesRegex(ValueError, "scientific Slurm step"):
                transport.require_step(self.parent)


def sys_argv_path():
    import sys
    return sys.argv[1]


if __name__ == "__main__":
    unittest.main()
