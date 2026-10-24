"""The pulsed-steering and outcome launchers, and the command lines of the steering modules they call.

The launchers run with a stand-in `sbatch` that only keeps a record, on a made-up study folder.
"""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from steering import lfm25_generation, pulsed_effects, pulsed_steering
from steering.outcome import __main__ as outcome_cli
from store import CLASSES, MODELS, REPO, WORK, publish, verified

PY = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python')


class ModelArguments(unittest.TestCase):
    """A model is given as its Hugging Face id or its short key, and every module works with the short key."""
    def test_model_key_takes_either_spelling_and_refuses_strangers(self):
        self.assertEqual(pulsed_steering.model_key('openai/gpt-oss-20b'), 'gpt')
        self.assertEqual(pulsed_steering.model_key('qwen330b'), 'qwen330b')
        with self.assertRaisesRegex(Exception, 'unknown model'):
            pulsed_steering.model_key('someone/else')

    def test_the_outcome_command_line_resolves_lists_of_models(self):
        self.assertEqual(outcome_cli.models_of('openai/gpt-oss-20b,glm'), ['gpt', 'glm'])


class StudyRoster(unittest.TestCase):
    """The models of a design are the ones whose manifest the study holds, in the order of store.MODELS."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=WORK/'tools/tmp')
        self.addCleanup(temporary.cleanup)
        self.driver = Path(temporary.name)/'driver'
        for model, name in (('glm', 'ASSIGNMENTS.json'), ('gpt', 'ASSIGNMENTS.json'), ('gemma', 'ASSIGNMENTS.json'),
                            ('gemma', 'STRONG.json'), ('gpt', 'STRONG.json')):
            publish(self.driver/model/name, {})
        self.enterContext(patch.object(pulsed_steering, 'DRIVER', self.driver))

    def test_design_models_follow_store_order_not_alphabetical_order(self):
        self.assertEqual(pulsed_steering.design_models('main'), ['gpt', 'gemma', 'glm'])
        self.assertEqual(pulsed_steering.design_models('strong'), ['gpt', 'gemma'])
        self.assertEqual(pulsed_steering.design_models('main'), [m for m in MODELS if m in {'gpt', 'gemma', 'glm'}])

    def run_check(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            pulsed_steering.main(['check', *argv])
        return out.getvalue().strip()

    def test_check_prints_the_short_key_when_the_manifest_is_there(self):
        self.assertEqual(self.run_check('--model', 'openai/gpt-oss-20b'), 'gpt')
        self.assertEqual(self.run_check('--model', 'gemma', '--design', 'strong'), 'gemma')

    def test_check_fails_when_the_study_has_no_manifest_of_the_design(self):
        with self.assertRaisesRegex(ValueError, 'plan the strong design of glm first'):
            self.run_check('--model', 'zai-org/GLM-4.7-Flash', '--design', 'strong')
        with self.assertRaisesRegex(ValueError, 'check needs one --model'):
            self.run_check()

    def test_effects_default_to_the_models_of_the_study_and_accept_either_spelling(self):
        seen = []
        with patch.object(pulsed_effects, 'analyze', lambda models, out, design: seen.append((models, design)) or {'primary_pooled': {}}), \
                patch.object(pulsed_effects, 'log'):
            pulsed_effects.main(['--out', str(self.driver/'x')])
            pulsed_effects.main(['--out', str(self.driver/'y'), '--design', 'strong'])
            pulsed_effects.main(['--out', str(self.driver/'z'), '--models', 'nvidia/Gemma-4-26B-A4B-NVFP4,gpt'])
        self.assertEqual(seen, [(['gpt', 'gemma', 'glm'], 'main'), (['gpt', 'gemma'], 'strong'), (['gemma', 'gpt'], 'main')])


class PlanInAStudy(unittest.TestCase):
    def test_plan_reads_and_writes_the_study_it_is_given_not_the_one_the_store_names(self):
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as temporary:
            study = Path(temporary)
            publish(study/'gpt/ASSIGNMENTS-full.json', {'records': []})
            publish(study/'gpt/COUNT_NOMINATIONS.json', {'records': [
                {'status': 'nominated', 'fold': fold, 'class': name, 'depth_third': 0, 'layer': 3, 'association': .5,
                 'stability': .5} for fold in range(5) for name in CLASSES]})
            result = pulsed_steering.plan(['gpt'], study=study)
            self.assertEqual((result['gpt']['executable'], result['gpt']['unavailable']), (0, 0))
            self.assertEqual(verified(study/'driver/gpt/ASSIGNMENTS.json')['model'], 'gpt')


class Lfm25Stages(unittest.TestCase):
    def test_the_stages_are_native_annotate_and_replay_and_need_a_gpu_job(self):
        self.assertEqual(sorted(lfm25_generation.STAGES), ['annotate', 'native', 'replay'])
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, 'Slurm GPU allocation'):
            lfm25_generation.main(['native'])

    def test_the_pulsed_module_keeps_the_judge_engine_and_no_lfm25_stage(self):
        self.assertTrue(callable(pulsed_steering.annotate_rows))
        for name in ('lfm_native', 'lfm_annotate', 'lfm_replay', 'native_kwargs'):
            self.assertFalse(hasattr(pulsed_steering, name), name)


@unittest.skipUnless(PY.exists(), 'needs the cluster environment')
class Launchers(unittest.TestCase):
    """The launchers, run with an `sbatch` that records its arguments (one line per call)."""
    SBATCH = ('n=$(( $(cat "$FAKE_SLURM/count" 2>/dev/null || echo 0) + 1 )); echo $n > "$FAKE_SLURM/count"\n'
              'echo "$@" >> "$FAKE_SLURM/calls"\n'
              'echo $((1000 + n))\n')

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=WORK/'tools/tmp')
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name)
        for folder in ('src', 'scripts'):
            os.symlink(REPO/folder, self.repo/folder)
        (self.repo/'bin').mkdir()
        (self.repo/'bin/sbatch').write_text('#!/bin/bash\n'+self.SBATCH)
        (self.repo/'bin/sbatch').chmod(0o755)
        self.study = self.repo/'study'
        publish(self.study/'BINDING.json', {})
        for model in ('gpt', 'glm'):
            publish(self.study/'driver'/model/'ASSIGNMENTS.json', {})
        publish(self.study/'driver/gpt/STRONG.json', {})

    def run_launcher(self, script, *args):
        env = {**os.environ, 'PATH': f"{self.repo/'bin'}:{os.environ['PATH']}", 'FAKE_SLURM': str(self.repo)}
        return subprocess.run(['bash', str(self.repo/'scripts'/script), *map(str, args)], cwd=self.repo, env=env,
                              capture_output=True, text=True)

    def calls(self):
        path = self.repo/'calls'
        return [line.split() for line in path.read_text().splitlines()] if path.exists() else []

    @staticmethod
    def option(call, name):
        return next(word.split('=', 1)[1] for word in call if word.startswith(name+'='))

    def test_one_gpu_job_per_shard_with_the_study_and_the_design(self):
        run = self.run_launcher('steering/run_pulsed_steering.sh', self.study, 'openai/gpt-oss-20b', 2, 12, 'strong')
        self.assertEqual(run.returncode, 0, run.stderr)
        first, second = self.calls()
        self.assertEqual(first[first.index('-J')+1], 'fr-gpt-0-of-2')
        self.assertEqual(first[first.index('-t')+1], '12:00:00')
        self.assertEqual(self.option(first, '--export'), f'ALL,STUDY_DATA_ROOT={self.study}')
        self.assertEqual(first[-8:], ['steering.pulsed_steering', 'run', '--model', 'openai/gpt-oss-20b', '--shard', '0/2',
                                      '--design', 'strong'])
        self.assertEqual(second[second.index('-J')+1], 'fr-gpt-1-of-2')
        self.assertTrue(first[-9].endswith('scripts/slurm/run_module_gpu.sbatch'))

    def test_a_model_without_a_manifest_in_the_study_submits_nothing(self):
        run = self.run_launcher('steering/run_pulsed_steering.sh', self.study, 'zai-org/GLM-4.7-Flash', 1, 4, 'strong')
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('plan the strong design of glm first', run.stderr)
        self.assertEqual(self.calls(), [])
        self.assertNotEqual(self.run_launcher('steering/run_pulsed_steering.sh', self.study, 'openai/gpt-oss-20b').returncode, 0)

    def test_the_analysis_runs_outcomes_of_each_manifest_then_effects_after_them(self):
        out = self.repo/'out'
        run = self.run_launcher('steering/analyze_pulsed_steering.sh', self.study, out)
        self.assertEqual(run.returncode, 0, run.stderr)
        calls = {call[call.index('-J')+1]: call for call in self.calls()}
        self.assertEqual(list(calls), ['pulsed-outcomes-main-glm', 'pulsed-outcomes-main-gpt', 'pulsed-effects-main',
                                       'pulsed-outcomes-strong-gpt', 'pulsed-effects-strong'])
        outcomes = calls['pulsed-outcomes-main-glm']
        self.assertEqual(outcomes[-5:], ['steering.pulsed_outcomes', '--model', 'glm', '--design', 'main'])
        self.assertEqual(self.option(outcomes, '--export'), f'ALL,STUDY_DATA_ROOT={self.study}')
        effects = calls['pulsed-effects-main']
        self.assertEqual(self.option(effects, '--dependency'), 'afterok:1001:1002')
        self.assertIn('--kill-on-invalid-dep=yes', effects)
        self.assertEqual(effects[-4:], ['--design', 'main', '--out', f'{out}/main'])
        self.assertEqual(self.option(calls['pulsed-effects-strong'], '--dependency'), 'afterok:1004')
        self.assertEqual(calls['pulsed-effects-strong'][-4:], ['--design', 'strong', '--out', f'{out}/strong'])

    def test_the_analysis_skips_a_design_the_study_lacks_and_refuses_an_old_output_folder(self):
        (self.study/'driver/gpt/STRONG.json').unlink()
        self.assertEqual(self.run_launcher('steering/analyze_pulsed_steering.sh', self.study, self.repo/'out').returncode, 0)
        self.assertNotIn('pulsed-effects-strong', [call[call.index('-J')+1] for call in self.calls()])
        count = len(self.calls())
        again = self.run_launcher('steering/analyze_pulsed_steering.sh', self.study, self.repo/'study')
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(len(self.calls()), count)

    def test_the_usage_job_gets_the_reports_the_new_folder_and_the_models(self):
        reports = self.repo/'reports'
        reports.mkdir()
        run = self.run_launcher('steering/measure_expert_usage.sh', reports, self.repo/'usage', 'openai/gpt-oss-20b', 'glm')
        self.assertEqual(run.returncode, 0, run.stderr)
        (call,) = self.calls()
        self.assertEqual(call[-8:], ['steering.outcome', 'usage', '--reports', str(reports), '--out', str(self.repo/'usage'),
                                     '--models', 'openai/gpt-oss-20b,glm'])
        self.assertNotEqual(self.run_launcher('steering/measure_expert_usage.sh', reports, reports, 'glm').returncode, 0)

    def test_sealing_needs_its_three_arguments(self):
        run = self.run_launcher('steering/seal_release.sh', 'capture', 'name')
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('TESTS', run.stderr)


if __name__ == '__main__':
    unittest.main()
