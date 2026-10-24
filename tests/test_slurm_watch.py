"""No cluster or terminal mutations: test notification guards and accounting."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
spec = importlib.util.spec_from_file_location('watch', Path(__file__).parents[1] / 'scripts/watch_slurm.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


class WatchTests(unittest.TestCase):
    def test_accounting_parent_and_cancel_suffix(self):
        raw = '12.batch|lmolfett|COMPLETED|0:0|3\n12|lmolfett|CANCELLED by 100|0:15|4'
        self.assertEqual(w.parse_accounting(raw, '12')['state'], 'CANCELLED')

    def test_missing_is_not_completion(self):
        self.assertIsNone(w.parse_accounting('', '12'))

    def test_other_user_rejected(self):
        with self.assertRaises(RuntimeError):
            w.parse_accounting('12|other|FAILED|1:0|4', '12')

    def test_draft_and_ui_changes_rejected(self):
        self.assertTrue(w.empty_prompt('› Ask Codex to do anything\n\n  footer'))
        for screen in ['› user draft\n\n footer', 'shell $', '› Ask Codex to do anything\nwrapped input\n\n footer']:
            self.assertFalse(w.empty_prompt(screen))

    def test_requeued_job_overrides_terminal_accounting(self):
        with patch.object(w, 'command', return_value='12|lmolfett|PENDING') as cmd:
            self.assertEqual(w.poll('12')['state'], 'PENDING')
            self.assertEqual(cmd.call_count, 1)

    def test_finished_job_missing_from_live_queue_uses_accounting(self):
        err = w.subprocess.CalledProcessError(1, 'squeue', stderr='Invalid job id specified')
        with patch.object(w, 'command', side_effect=[err, '12|lmolfett|FAILED|1:0|4']):
            self.assertEqual(w.poll('12')['state'], 'FAILED')

    def test_scheduler_outage_does_not_claim_completion(self):
        err = w.subprocess.CalledProcessError(1, 'squeue', stderr='Connection refused')
        with patch.object(w, 'command', side_effect=err):
            with self.assertRaises(w.subprocess.CalledProcessError):
                w.poll('12')

    def test_draft_never_sends_keys(self):
        with patch.object(w, 'verify_target'), patch.object(w, 'capture', return_value='draft'), patch.object(w.subprocess, 'run') as run:
            self.assertEqual(w.notify({}, 'message'), 'busy_prompt')
            run.assert_not_called()

    def test_wrong_process_never_sends_keys(self):
        with patch.object(w, 'verify_target', side_effect=RuntimeError('changed')), patch.object(w.subprocess, 'run') as run:
            with self.assertRaises(RuntimeError):
                w.notify({}, 'message')
            run.assert_not_called()

    def test_busy_or_queued_session_defers_without_keys(self):
        for screen in ('Working (esc to interrupt)', 'Queued follow-up inputs'):
            with patch.object(w, 'verify_target'), patch.object(w, 'capture', return_value=screen), patch.object(w.subprocess, 'run') as run:
                self.assertEqual(w.notify({'pane':'%3'}, 'message'), 'deferred_until_idle')
                run.assert_not_called()

    def test_idle_submits_once(self):
        blank = '› Ask Codex to do anything\n\n footer'
        with patch.object(w, 'verify_target'), patch.object(w, 'capture', return_value=blank), patch.object(w.time, 'sleep'), patch.object(w.subprocess, 'run') as run:
            self.assertEqual(w.notify({'pane':'%3'}, 'message'), 'sent_to_tui')
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args.args[0][-1], 'Enter')


if __name__ == '__main__':
    unittest.main()
