"""Behaviour tests for the stalled-write watchdog around the annotation client.

The wrapper is exercised with tiny fake clients: publishing checkpoints, stalling
forever, stalling once and then finishing, and failing immediately with a real error.
"""
from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

WATCHDOG = Path(__file__).parents[1] / "sbatch/qwen_annotation_watchdog.sh"


def write_client(directory: Path, body: str) -> Path:
    client = directory / "fake-client.sh"
    client.write_text("#!/bin/bash\nset -euo pipefail\n" + body, encoding="utf-8")
    client.chmod(0o755)
    return client


def run_watchdog(directory: Path, client: Path, *, stall_seconds: int, max_restarts: int, timeout: float = 120.0):
    output_dir = directory / "part"
    output_dir.mkdir(parents=True, exist_ok=True)
    client_log = output_dir / "logs/annotation_client.log"
    completed = subprocess.run(
        [
            "/usr/bin/bash", str(WATCHDOG),
            "--output-dir", str(output_dir),
            "--client-log", str(client_log),
            "--stall-seconds", str(stall_seconds),
            "--poll-seconds", "1",
            "--max-restarts", str(max_restarts),
            "--", str(client),
        ],
        capture_output=True, text=True, timeout=timeout,
    )
    return completed, output_dir, client_log


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


class WatchdogTests(unittest.TestCase):
    def test_finishing_client_is_not_restarted(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            output_dir = directory / "part"
            checkpoints = output_dir / "checkpoints"
            client = write_client(
                directory,
                f"mkdir -p {checkpoints}\n"
                f"printf '{{}}' > {checkpoints}/batch-000000.json\n"
                "printf 'PART_PROGRESS completed=64 expected=25000\\n'\n",
            )
            completed, output_dir, client_log = run_watchdog(directory, client, stall_seconds=2, max_restarts=1)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("annotation_client_complete attempt=1", completed.stdout)
            self.assertNotIn("annotation_watchdog_restart", completed.stdout)
            self.assertEqual(sorted(p.name for p in (output_dir / "checkpoints").iterdir()), ["batch-000000.json"])
            self.assertIn("PART_PROGRESS completed=64", read(client_log))

    def test_stalled_client_is_killed_diagnosed_and_restarted(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            output_dir = directory / "part"
            checkpoints = output_dir / "checkpoints"
            rounds = directory / "rounds"
            client = write_client(
                directory,
                f"round=0\n[[ -f {rounds} ]] && round=$(<{rounds})\n"
                f"round=$((round + 1)); printf '%s' \"$round\" > {rounds}\n"
                'if (( round == 1 )); then\n'
                "    printf 'PART_PROGRESS completed=128 expected=25000\\n'\n"
                "    exec sleep 30\n"
                "fi\n"
                f"mkdir -p {checkpoints}\n"
                f"printf '{{}}' > {checkpoints}/batch-000000.json\n"
                "printf 'PART_PROGRESS completed=64 expected=25000\\n'\n",
            )
            completed, output_dir, client_log = run_watchdog(directory, client, stall_seconds=2, max_restarts=2)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("annotation_watchdog_restart attempt=1", completed.stdout)
            self.assertIn("annotation_client_restart attempt=1", completed.stdout)
            self.assertIn("annotation_client_complete attempt=2", completed.stdout)
            diagnostics = sorted((output_dir / "logs").glob("stall-diagnostics-*.log"))
            self.assertEqual(len(diagnostics), 1)
            text = diagnostics[0].read_text(encoding="utf-8")
            self.assertIn("MFA_STALL_DETECTED", text)
            self.assertIn("parent_wchan=", text)
            self.assertIn("newest_checkpoints:", text)
            # The restarted client ran again and published its own progress line.
            self.assertIn("PART_PROGRESS completed=128", read(client_log))
            self.assertIn("PART_PROGRESS completed=64", read(client_log))

    def test_stalling_client_gives_up_after_max_restarts(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            client = write_client(directory, "printf 'still stalling\\n'\nexec sleep 40\n")
            completed, output_dir, _log = run_watchdog(directory, client, stall_seconds=2, max_restarts=1)
            self.assertEqual(completed.returncode, 1)
            self.assertEqual(completed.stdout.count("annotation_watchdog_restart"), 2)
            self.assertIn("annotation_watchdog_giving_up attempts=2 max_restarts=1", completed.stderr)

    def test_immediate_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            client = write_client(directory, "printf 'HTTP Error 400: Bad Request\\n' >&2\nexit 3\n")
            completed, _output_dir, _log = run_watchdog(directory, client, stall_seconds=2, max_restarts=3)
            self.assertEqual(completed.returncode, 3)
            self.assertIn("not retried: non-stall error", completed.stderr)
            self.assertNotIn("annotation_watchdog_restart", completed.stdout)
            self.assertNotIn("annotation_client_complete", completed.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
