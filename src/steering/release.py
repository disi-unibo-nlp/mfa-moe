"""Seal a whole-reasoning style release: its receipts and a frozen copy of the code its jobs run.

  python -m steering.release complement NAME --tests N   suppress the states the base releases left out (r18)
  python -m steering.release relink NAME --tests N       the final combine again under the eight-model link (r19, r20)
  python -m steering.release outcome NAME --tests N      the outcome-selected nudge (--models, --shards)
  python -m steering.release capture NAME --tests N      native attempts with routing saved (data of the class-conditioned selection)
  python -m steering.release sweep NAME --tests N --half A|B --capture C --models M   one half of the class-conditioned nudge sweep
  python -m steering.release continuation NAME --tests N --parent P   the rows a complement left unfinished

N is the number of tests that passed locally on this commit. Run it from the repository with the committed code
clean: what runs is `git archive HEAD`, unpacked once into mfa-moe/releases/<commit>. It writes report/NAME/
(RELEASE.json, the kind file COMPLEMENT/RELINK/OUTCOME/CAPTURE_NATIVE/OUTCOME_SWEEP.json, SOURCES.json, README.md and
the receipts they cite) and prints the snapshot path. scripts/steering/submit_release_jobs.sh report/NAME then submits the jobs.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo

from models.roster import resolve
from steering.plan_continuation import APPROVED_GPU_HOURS, parse_shards, plan_continuation
from store import FAST, MODELS, REPO, WORK, check_files, file_sha, publish, require, verified

REPORTS = REPO/'report'
SNAPSHOTS = WORK/'mfa-moe/releases'
TEMPLATE = 'whole-reasoning-20261021-r7'  # pins the inputs, the sentence classifier and the nomination reference
CHAIN = 'whole-reasoning-20261021-r20'  # the release whose source list a continuation extends
LINK = 'final-routing-20261015-r1/driver/EFFICIENCY_LINK_V2.json'  # the eight-model efficiency link, under report/
# The jobs also read scripts/setup and requirements (scripts/slurm/env.sh finds the judge client environment there).
RUN_PATHS = ('src', 'tests/steering', 'tests/routing', 'scripts/slurm', 'scripts/setup', 'scripts/steering', 'requirements')
# What a snapshot unpacks: the paths of the repository that `git archive` takes at the sealed commit.
ARCHIVED = ('src', 'scripts', 'configs', 'tests', 'pyproject.toml', 'requirements', 'AGENTS.md')
# The kind file of a release is KIND.json unless the kind names its file here.
KIND_FILES = {'capture': 'CAPTURE_NATIVE.json', 'sweep': 'OUTCOME_SWEEP.json'}
FRACTIONS = (0.5, 0.7, 0.85)  # where the length-limit baseline stops the native reasoning
ANSWER_TOKENS = 2048          # answer allowance of a length-limit row
SWEEP_DELTAS = (0.4, 0.8, 1.2)  # logit shifts of the class-conditioned sweep
OUTCOME_SCOPE = ('tests/steering/outcome plus tests/steering/test_stream_batch.py; the allocated validation runs '
                 'both suites in full')


def prior_usage() -> dict:
    """The billing totals of the earlier whole-reasoning and outcome jobs (scripts/slurm/billing_hours_used.sh), as of now."""
    ledger = str(REPO/'scripts/slurm/billing_hours_used.sh')
    totals = {prefix: subprocess.run([ledger, prefix], capture_output=True, text=True, check=True).stdout.splitlines()[-1]
              for prefix in ('whole-', 'os-', 'oc-')}
    return {'ledger_totals': totals, 'snapshot_time': time.time(), 'budget_reset': False}


def rome(text: str) -> float:
    """A Unix time from an ISO date-time (Europe/Rome unless it names a zone) or from epoch seconds."""
    if text.isdigit():
        return float(text)
    moment = datetime.fromisoformat(text)
    return (moment if moment.tzinfo else moment.replace(tzinfo=ZoneInfo('Europe/Rome'))).timestamp()


def complement(root: Path, args: argparse.Namespace) -> dict:
    """r18: the suppress cells of the classes the base releases left out, at each model's base strength.

    The base releases (r7; GPT r8) suppress only the classes tied to longer reasoning, so the per-cell heatmap
    of the paper has empty suppress cells. The new cells run on the same families and seeds with no native rows
    and no canary, and the release hosts a new combine of r7-r17 plus itself.
    """
    from steering.whole_reasoning import manifest_path
    cutoff = args.cutoff
    require(cutoff, '--cutoff is required: no GPU row may start after it (Rome time, e.g. 2026-10-24T06:00)')
    parents = {m: REPORTS/('whole-reasoning-20261021-r8' if m == 'gpt' else TEMPLATE) for m in MODELS}
    # 2-GPU shards of 4 hours, sized from the base releases' GPU time per row so that a shard's expected
    # work is about half its walltime (a row still running at the walltime stays unknown).
    shards = {'gpt': 3, 'nemotron': 4, 'qwen35': 12, 'qwen36': 12, 'gemma': 6, 'glm': 27, 'qwen330b': 3, 'lfm25': 8}
    require(set(shards) == set(MODELS), 'the shard sizes name other models than store.MODELS')
    sweep = {m: verified(parents[m]/m/'SELECTED.json')['dose'] for m in MODELS}
    # The combine: every source of r17, with this release joined to each model's base source.
    sources = verified(REPORTS/'whole-reasoning-20261021-r17/SOURCES.json')['sources']
    for model in MODELS:
        source = next(s for s in sources if s['model'] == model and Path(s['root']).resolve() == parents[model].resolve()
                      and s['dose'] == sweep[model])
        source.setdefault('complements', []).append(str(root))
    # The complement reads each parent's base manifest (natives) and nominations outside the snapshot.
    inputs = {}
    for model in MODELS:
        for path in (manifest_path(model, sweep[model], parents[model]), parents[model]/model/'WHOLE_NOMINATIONS.json'):
            inputs[str(path)] = file_sha(path)
    return dict(
        kind_file={'models': {m: {'parent': str(parents[m])} for m in MODELS}},
        template=verified(REPORTS/TEMPLATE/'RELEASE.json'), models=MODELS, shards=shards, sweep=sweep,
        cutoff=rome(cutoff), cpu_ceiling=200., sources=sources, inputs=inputs,
        scope='tests/steering/test_whole*.py; the allocated validation runs tests/steering, '
              'tests/steering/outcome and tests/routing in full',
        readme='The base releases suppress only the states tied to longer reasoning. This release suppresses the\n'
               "other states at each model's base strength (GPT and GLM half, the others full) on the same families\n"
               'and seeds, so the per-cell results cover both operations for all seven states. The new cells are\n'
               'not efficiency-directed and never enter the primary or the dose response. Its combine job writes\n'
               'the final combined analysis of r7-r17 plus these cells to analysis-combined/EFFECTS_WHOLE.json.\n'
               f'Shards: {", ".join(f"{m} {shards[m]}" for m in MODELS)}. GPU work stops at {cutoff} Rome.\n')


def relink(root: Path, args: argparse.Namespace) -> dict:
    """r19, r20: the combined analysis of the parent's sources again, with another efficiency link. No steering.

    The releases r7-r18 chose each model's efficiency-directed cells from the seven-model pre-registration, so
    LFM2.5 used the pooled signs. The eight-model receipt gives LFM its own signs; its directed cells are a subset
    of the pooled ones and every one of them already ran at every strength.
    """
    parent_name = 'whole-reasoning-20261021-r18'
    parent = verified(REPORTS/parent_name/'RELEASE.json')
    link = REPORTS/LINK
    sealed_link = verified(link)
    require(set(sealed_link['models']) == set(MODELS), 'the link receipt does not cover the eight models')
    return dict(
        kind_file={'parent': str(REPORTS/parent_name), 'link': str(link), 'link_sha256': sealed_link['sha256'],
                   'reason': 'LFM2.5 uses its own native efficiency-link signs (eight-model receipt) '
                             'instead of the pooled seven-model signs'},
        template=parent, models=MODELS, shards={}, sweep={}, cutoff=parent['cutoff'],
        cpu_ceiling=24., sources=verified(REPORTS/parent_name/'SOURCES.json')['sources'],
        inputs={str(link): file_sha(link)},
        scope='tests/steering/test_whole*.py; the allocated run repeats tests/steering, tests/steering/outcome '
              'and tests/routing in full',
        readme=f'Re-analysis only, no steering. The combined analysis of {parent_name} (r7-r18) is rerun with the\n'
               'eight-model efficiency link (driver/EFFICIENCY_LINK_V2.json), in which LFM2.5 has its own native\n'
               'signs instead of the pooled seven-model ones. Its directed cells are a subset of the pooled ones,\n'
               'all of which ran at every strength, so only LFM2.5 and the pooled rows can change. The seven other\n'
               'models keep their pre-registered signs. Output: analysis-combined/EFFECTS_WHOLE.json.\n')


def models_and_shards(args: argparse.Namespace, default_models: str) -> tuple[tuple[str, ...], dict[str, int]]:
    """The models of a GPU release and their 2-GPU shards (4 hours each): --models, and --shards as N or model:N,...

    A model is a Hugging Face id or a short key; a release records the short key."""
    models = tuple(resolve(m) for m in (args.models or default_models).split(','))
    shards = args.shards or '3'
    count = ({m: int(shards) for m in models} if shards.isdigit() else
             {resolve(m): int(n) for m, n in (item.split(':') for item in shards.split(','))})
    require(set(count) == set(models), '--shards must name exactly the --models')
    return models, count


def outcome(root: Path, args: argparse.Namespace) -> dict:
    """The outcome-selected gentle nudge (steering.outcome).

    --models: models to steer. --shards: 2-GPU shards (4 hours each) per model, N or model:N,...
    """
    from steering.outcome.manifest import DEFAULT_MODELS, default_spec
    from steering.outcome.select import BULK
    models, count = models_and_shards(args, DEFAULT_MODELS)
    cutoff = args.cutoff
    require(cutoff, '--cutoff is required: no GPU row may start after it (Rome time, e.g. 2026-10-23T23:00)')
    deltas, fractions, answer_tokens = (0.4, 0.8, 1.6), FRACTIONS, ANSWER_TOKENS
    # The selection reads these outside the snapshot: per-attempt usage and the class nominations.
    inputs = {str(path): file_sha(path) for model in models
              for path in (BULK/model/'USAGE.json', BULK/model/'USAGE.npz', REPORTS/TEMPLATE/model/'WHOLE_NOMINATIONS.json')}
    # The CPU ceiling is the time limits submit_release_jobs.sh gives the tests, the manifests and the analysis (24 core-hours)
    # plus those of each model's extraction (12).
    return dict(
        kind_file={'models': {m: {**default_spec(m, REPORTS), 'deltas': list(deltas), 'fractions': list(fractions),
                                  'answer_tokens': answer_tokens} for m in models}},
        template=verified(REPORTS/TEMPLATE/'RELEASE.json'), models=models, shards=count,
        sweep={m: 'main' for m in models}, cutoff=rome(cutoff), cpu_ceiling=24.+12.*len(models), sources=None, inputs=inputs,
        scope=OUTCOME_SCOPE,
        readme='Nudges a small set of experts chosen by outcome (used more in wrong than right attempts of the same\n'
               'question, cohort B) by subtracting a delta from their router logits at every prompt and\n'
               "reasoning token, against matched random experts, and compares both with unedited routing (the base\n"
               "releases' native runs) and a stop-thinking-earlier baseline cut from those native runs.\n"
               f'Models: {", ".join(models)}; deltas {list(deltas)}; length fractions {list(fractions)}.\n'
               'Code: src/steering/outcome (README there).\n'
               f'GPU work stops at {cutoff} Rome.\n')


def capture(root: Path, args: argparse.Namespace) -> dict:
    """Native attempts with their routing saved, the data of the class-conditioned outcome selection.

    Every model answers the same roster of 300 questions six times with unedited routing; every reasoning token keeps
    its native top-k ids and weights at every routed layer. --models defaults to all eight, --shards to 3 per model.
    The stores go to FAST (report/NAME/stores is a link), because WORK cannot hold the captures. The jobs are named
    oc-*, so scripts that count whole-* jobs never see them.
    """
    from steering.outcome.capture import ATTEMPTS, CAP_TOKENS, pinned_inputs, roster
    models, count = models_and_shards(args, ','.join(MODELS))
    cutoff = args.cutoff
    require(cutoff, '--cutoff is required: no GPU row may start after it (Rome time, e.g. 2026-10-30T23:00)')
    questions = roster(list(models), REPORTS)
    return dict(
        kind_file={'models': {m: {'questions': questions, 'attempts': ATTEMPTS, 'cap_tokens': CAP_TOKENS} for m in models}},
        template=verified(REPORTS/TEMPLATE/'RELEASE.json'), models=models, shards=count,
        sweep={m: 'main' for m in models}, cutoff=rome(cutoff), cpu_ceiling=200., sources=None,
        inputs={str(path): file_sha(path) for path in pinned_inputs(REPORTS, list(models))},
        scope=OUTCOME_SCOPE, job_prefix='oc', stores=FAST/root.name/'stores',
        readme=f'Every model answers the same {len(questions)} questions {ATTEMPTS} times with unedited routing; every\n'
               'reasoning token keeps its native top-k expert ids and weights at every routed layer (AUDIT.npz per\n'
               f'unit). Completion allowance {CAP_TOKENS} tokens. These attempts feed the class-conditioned outcome\n'
               'selection, the attempt-selection analysis and serve as the unedited arm of the nudge releases.\n'
               f'Models: {", ".join(models)}. Code: src/steering/outcome (capture.py).\n'
               f'GPU work stops at {cutoff} Rome.\n')


def sweep(root: Path, args: argparse.Namespace) -> dict:
    """One half of the class-conditioned sweep.

    --capture: the capture release (report/NAME) whose manifests, expert sets (SELECTION_R2.json) and finished
    attempts the sweep reads. --half A runs every available expert set at every shift of SWEEP_DELTAS; --half B runs
    the cells of --pick (the PICK.json that steering.outcome pick wrote for half A). --models is required.
    The jobs are named os-*; the stores go to FAST like a capture's.
    """
    from steering.outcome.manifest import halves
    require(args.half in ('A', 'B') and (args.half == 'A') == (args.pick is None),
            '--half is A or B; half A has no --pick and half B needs one')
    require(args.capture and args.models, '--capture and --models are required')
    models, count = models_and_shards(args, '')
    cutoff = args.cutoff
    require(cutoff, '--cutoff is required: no GPU row may start after it (Rome time, e.g. 2026-10-30T23:00)')
    capture_root = REPORTS/args.capture
    picked = verified(args.pick) if args.pick else None
    spec, inputs = {}, {}
    for model in models:
        selection_path = capture_root/model/'SELECTION_R2.json'
        selection = verified(selection_path)
        questions = verified(capture_root/'CAPTURE_NATIVE.json')['models'][model]['questions']
        available = [name for name, entry in selection['sets'].items() if 'sets' in entry]
        cells = ([{'set': name, 'delta': delta} for name in available for delta in SWEEP_DELTAS] if args.half == 'A'
                 else [cell for cell in picked['confirm'] if cell['set'] in available])
        require(cells, f'no cell to run for {model}')
        spec[model] = {'capture': str(capture_root), 'selection': str(selection_path),
                       'selection_sha256': selection['sha256'], 'half': args.half,
                       'questions': halves(questions)[args.half], 'cells': cells, 'fractions': list(FRACTIONS),
                       'answer_tokens': ANSWER_TOKENS,
                       'pick': {'path': str(args.pick), 'sha256': picked['sha256']} if picked else None}
        for path in (selection_path, capture_root/model/'WHOLE.json', capture_root/model/'extraction/outcomes.parquet'):
            inputs[str(path)] = file_sha(path)
    return dict(
        kind_file={'models': spec}, template=verified(REPORTS/TEMPLATE/'RELEASE.json'), models=models, shards=count,
        sweep={m: 'main' for m in models}, cutoff=rome(cutoff), cpu_ceiling=60., sources=None, inputs=inputs,
        scope=OUTCOME_SCOPE, job_prefix='os', stores=FAST/root.name/'stores',
        readme='Experts used more in wrong than in right attempts of the same question, on the tokens of one\n'
               'reasoning class (or on all reasoning, the class-blind set), are nudged down by a logit shift at\n'
               'every prompt and reasoning token, against two disjoint matched random sets. The unedited arm is\n'
               f'the six attempts per question of {args.capture}; stop-early rows cut one of them.\n'
               f'Half {args.half}; cells: {len(spec[models[0]]["cells"])} per model; shifts {list(SWEEP_DELTAS)}.\n'
               f'GPU work stops at {cutoff} Rome.\n')


def continuation(root: Path, args: argparse.Namespace) -> dict:
    """The follow-up of a complement release that could not start or finish all its shards before its cutoff.

    It runs every row the parent never started and continues every row it left PARTIAL
    (steering.whole_reasoning.continuation_manifest), with the parent's UIDs, seeds and actions and a later cutoff, after
    all the parent's shard jobs have ended. Its shards are sized by steering.plan_continuation under --max-gpu-hours. Its
    combine extends the source list of release --chain (the newer efficiency link) with this release, and runs
    under the eight-model link (--link). The parent's extraction jobs are passed to scripts/steering/submit_release_jobs.sh.
    """
    from steering.whole_reasoning import manifest_path
    require(args.parent, '--parent is required: the complement release to continue')
    require(args.cutoff, '--cutoff is required: no GPU row may start after it (Rome time, e.g. 2026-10-27T06:00)')
    parent = REPORTS/args.parent
    parent_release = verified(parent/'RELEASE.json')
    plan = plan_continuation(parent, args.max_gpu_hours or APPROVED_GPU_HOURS, parse_shards(args.shards or ''))
    require(not plan['live_parent_jobs'], f'{args.parent} still has shard jobs that have not ended: {plan["live_parent_jobs"]}')
    cutoff = rome(args.cutoff)
    require(cutoff > max(time.time(), parent_release['cutoff']), "the cutoff must be later than now and than the parent's")
    link = Path(args.link) if args.link else REPORTS/LINK
    sealed_link = verified(link)
    require(set(sealed_link['models']) == set(MODELS), 'the link receipt does not cover the eight models')
    models, shards = list(plan['models']), plan['shards']
    sweep = {m: plan['models'][m]['dose'] for m in models}
    sources = verified(REPORTS/(args.chain or CHAIN)/'SOURCES.json')['sources']
    for model in models:
        source = next((s for s in sources if s['model'] == model and
                       parent.resolve() in {Path(c).resolve() for c in s.get('complements', [])}), None)
        require(source, f'{args.chain or CHAIN} has no {model} source with {args.parent} among its complements')
        source['complements'].append(str(root.resolve()))
    # The continuation reads each parent manifest; the combine reads the link receipt.
    inputs = {str(manifest_path(model, sweep[model], parent)): file_sha(manifest_path(model, sweep[model], parent))
              for model in models}
    inputs[str(link)] = file_sha(link)
    counts = ', '.join(f'{m} {v["never_started"]}+{v["partial"]} in {shards[m]}' for m, v in plan['models'].items())
    return dict(
        kind_file={'models': {m: {'parent': str(parent)} for m in models}, 'parent': args.parent, 'plan': plan,
                   'link': str(link), 'link_sha256': sealed_link['sha256']},
        template=parent_release, models=models, shards=shards, sweep=sweep, cutoff=cutoff, cpu_ceiling=200., sources=sources,
        inputs=inputs,
        scope='tests/steering/test_whole*.py; the allocated validation runs tests/steering, '
              'tests/steering/outcome and tests/routing in full',
        readme=f'Continues the complement release {args.parent}, whose shards could not all start or finish before its\n'
               'science cutoff. It runs the rows the parent never started and continues the rows it left PARTIAL, with\n'
               "the parent UIDs, seeds and actions; each saved partial is copied into this release's own store and\n"
               're-prefilled through the same hooks, and no parent store is written. Rows the parent finished stay\n'
               "with it. Its combined analysis joins every outcome to the parent's rows by UID, with the efficiency\n"
               'link of the eight-model receipt (driver/EFFICIENCY_LINK_V2.json): analysis-combined/EFFECTS_WHOLE.json.\n'
               f'Rows (never started + partial) and shards: {counts}.\n'
               f'GPU work stops at {datetime.fromtimestamp(cutoff, ZoneInfo("Europe/Rome")):%Y-%m-%d %H:%M} Rome.\n')


KINDS = {'complement': complement, 'relink': relink, 'outcome': outcome, 'capture': capture, 'sweep': sweep,
         'continuation': continuation}


def snapshot(root: Path, commit: str) -> tuple[Path, Path]:
    """Archive the commit into the release and unpack it once under mfa-moe/releases/<commit>."""
    archive = root/'SOURCE.tar.gz'
    subprocess.run(['git', 'archive', '--format=tar.gz', '--output='+str(archive), commit, *ARCHIVED],
                   cwd=REPO, check=True)
    folder = SNAPSHOTS/commit
    if not folder.exists():  # unpacked aside and renamed, so a killed tar never leaves a half snapshot behind
        partial = folder.with_name(commit+'.partial')
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True, mode=0o700)
        subprocess.run(['tar', '-xzf', str(archive), '-C', str(partial)], check=True)
        (partial/'RELEASE_COMMIT').write_text(commit+'\n')
        partial.rename(folder)
    return archive, folder


def seal(kind: str, name: str, tests: int, args: argparse.Namespace) -> Path:
    """Write the receipts of a new release and materialize its code snapshot; return the snapshot."""
    require(not subprocess.run(['git', 'status', '--porcelain', '--', *RUN_PATHS], cwd=REPO,
                               capture_output=True, text=True, check=True).stdout, 'commit the code first')
    root = REPORTS/name
    require(not root.exists(), f'{root} exists')
    spec = KINDS[kind](root, args)
    kind_file = KIND_FILES.get(kind, kind.upper()+'.json')
    template, shards = spec['template'], spec['shards']
    stores = spec.get('stores')  # FAST folder that report/NAME/stores links to (captures and sweeps)
    cutoff = spec['cutoff']
    # A 2-GPU shard of 4 hours is 8 GPU-hours; the ceiling is the whole plan plus 10 hours of slack.
    gpu_ceiling = 8.*sum(shards.values())+10 if shards else 0.
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True,
                            check=True).stdout.strip()
    # Which release this one derives from: a relink and a continuation name their parent (a continuation keeps
    # its parent's template too), the others name their template.
    derived = {('parent' if kind in ('relink', 'continuation') else 'template')+'_release_sha256': template['sha256']}
    if kind == 'continuation':
        derived['template_release_sha256'] = template['template_release_sha256']
    require(stores is None or not stores.exists(), f'{stores} exists')
    root.mkdir(mode=0o700)
    try:
        if stores is not None:
            stores.mkdir(parents=True, mode=0o700)
            (root/'stores').symlink_to(stores)
        prior = publish(root/'PRIOR_USAGE.json', prior_usage())
        kind_receipt = publish(root/kind_file, spec['kind_file'])
        validation = publish(root/'LOCAL_VALIDATION.json', {'passed': True, 'tests': tests, 'commit': commit,
                                                             'scope': spec['scope']})
        plan = publish(root/'RESOURCE_PLAN.json', {
            'models': list(spec['models']), 'sweep': spec['sweep'], 'shards': shards, 'new_gpu_hour_ceiling': gpu_ceiling,
            'cpu_core_hour_ceiling': spec['cpu_ceiling'], 'cutoff': cutoff, 'no_budget_reset': True})
        if spec['sources'] is not None:
            publish(root/'SOURCES.json', {'sources': spec['sources']})
        (root/'README.md').write_text(f'# Whole-reasoning {kind} ({name})\n\n'+spec['readme'])

        archive, folder = snapshot(root, commit)
        files = {str(p): file_sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
        # The template's pinned inputs that live in a snapshot are the same files in this one.
        inputs = {k.replace(template['release_root'], str(folder)): v for k, v in template['inputs'].items()}
        inputs.update(spec['inputs'])
        check_files(inputs)
        publish(root/'RELEASE.json', {
            'commit': commit, 'release_root': str(folder), 'models': list(spec['models']), 'sweep': spec['sweep'],
            'shards': shards, 'cutoff': cutoff, kind+'_sha256': kind_receipt['sha256'],
            'source_archive': {'path': str(archive), 'sha256_file': file_sha(archive)},
            'code_files': files, 'inputs': inputs, 'classifier': template['classifier'],
            'nomination_reference': template['nomination_reference'],
            'local_validation_sha256': validation['sha256'], 'resource_plan_sha256': plan['sha256'],
            'new_gpu_hour_ceiling': gpu_ceiling, 'cpu_core_hour_ceiling': spec['cpu_ceiling'], 'engineering_only': False,
            'prior_usage': prior, 'job_prefix': spec.get('job_prefix', 'whole'),
            **({'stores': str(stores)} if stores is not None else {}), **derived})
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)  # a failed seal leaves no half-written release
        if stores is not None:
            shutil.rmtree(stores, ignore_errors=True)
        raise
    return folder


def main() -> None:
    """Seal the release the command line names and print its code snapshot."""
    parser = argparse.ArgumentParser(prog='steering.release', description=__doc__.split('\n\n')[0])
    parser.add_argument('kind', choices=sorted(KINDS))
    parser.add_argument('name')
    parser.add_argument('--tests', type=int, required=True, help='tests that passed locally on this commit')
    parser.add_argument('--models', help='outcome, capture, sweep: the models (default: outcome the first wave, '
                        'capture all eight; sweep requires it)')
    parser.add_argument('--shards', help='outcome, capture, sweep: N per model, or model:N,...; default 3. '
                        'continuation: model:N,... instead of the computed sizes')
    parser.add_argument('--cutoff', help='complement, outcome, capture, sweep, continuation (required): no GPU row '
                        'starts after this time (Rome, unless it names a zone)')
    parser.add_argument('--half', choices=('A', 'B'), help='sweep only: the question half to run')
    parser.add_argument('--capture', help='sweep only: the capture release (report/NAME) the sweep reads')
    parser.add_argument('--pick', type=Path, help='sweep half B only: the PICK.json of half A (steering.outcome pick)')
    parser.add_argument('--parent', help='continuation only: the complement release (report/NAME) to continue')
    parser.add_argument('--chain', help=f'continuation only: the release whose SOURCES.json it extends (default {CHAIN})')
    parser.add_argument('--link', type=Path, help=f'continuation only: efficiency-link receipt (default report/{LINK})')
    parser.add_argument('--max-gpu-hours', type=float,
                        help=f'continuation only: GPU-hours its shards may reserve (default {APPROVED_GPU_HOURS:g})')
    args = parser.parse_args()
    print(seal(args.kind, args.name, args.tests, args))


if __name__ == '__main__':
    main()
