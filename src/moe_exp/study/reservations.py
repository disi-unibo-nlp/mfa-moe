"""One ledger: terminal actuals plus full limits for all outstanding work."""
from pathlib import Path
import re
import time
from moe_exp.io import (
    submission_paths,
    reservation_paths,
    control_root,
    digest,
    file_sha,
    locked,
    publish,
    require,
    verified,
)

TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}
LIMITS = {'preparation_qualification': 1280., 'preparation_reader': 80.,
          'preparation_cpu': 40.}
ENGINEERING_GPU_HOURS_PER_MODEL = 4.
MODEL_LOAD_RESERVE_MINUTES = 15


def charged_hours(card, usage, margin):
    if usage is not None and usage['state'] in TERMINAL:
        return usage['billing_hours'] * margin, 'terminal_actual'
    return card['minutes'] / 60 * card['billing_units'] * margin, 'full_time_limit'


def ledger(binding, extra=(), usage=None):
    from .resources import accounting
    root = Path(binding['root'])
    usage = accounting(root) if usage is None else usage
    cards = {}
    accepted = {}
    for path in submission_paths(root):
        receipt = verified(path)
        owner = path.parents[2]
        card = verified(owner / 'cards' / (receipt['card_sha256'] + '.json'))
        identity = receipt['card_sha256']
        cards[identity] = {k: v for k, v in card.items() if k != 'sha256'}
        accepted[identity] = receipt['job_id']
    for path in reservation_paths(root):
        receipt = verified(path)
        identity = receipt['card_sha256']
        card_path = path.parents[1] / 'cards' / (identity + '.json')
        if identity not in cards and card_path.exists():
            cards[identity] = {k: v for k, v in verified(card_path).items() if k != 'sha256'}
    cards.update({digest(card): card for card in extra})
    totals, rows = {}, []
    for identity, card in cards.items():
        jid = accepted.get(identity)
        require(not jid or jid in usage, 'accepted job has no native accounting: ' + str(jid))
        hours, basis = charged_hours(card, usage.get(jid), binding['config']['budget']['margin'])
        phase = card['budget_phase']
        totals[phase] = totals.get(phase, 0.) + hours
        rows.append({'card_sha256': identity, 'phase': phase, 'hours': hours, 'basis': basis,
                     'job_id': jid, 'source_commit': card.get('source_commit')})
    return {'totals': totals, 'rows': rows, 'includes_margin': True}


def reserve_many(binding, cards):
    from .resources import snapshot, availability
    root, control = Path(binding['root']), control_root(binding)
    with locked(root, 'reservations'):
        proof = ledger(binding, cards)
        limits = {**LIMITS, 'finalization': binding['config']['budget']['closeout_hours']}
        if any(c['budget_phase'] == 'production' for c in cards):
            admission = verified(root / 'ADMISSION.json')
            require(admission['status'] == 'admitted', 'complete study was not admitted')
            limits['production'] = admission['production_allocation_hours']
        for phase in {c['budget_phase'] for c in cards}:
            require(phase in limits, 'unknown reservation phase')
            require(proof['totals'][phase] <= limits[phase] + 1e-9, phase + ' allowance exhausted')
        prep = sum(proof['totals'].get(phase, 0.) for phase in LIMITS)
        require(prep <= binding['config']['budget']['preparation_hours'] + 1e-9, 'original preparation allowance exhausted')
        repair = sum(row['hours'] for row in proof['rows']
                     if row['source_commit'] == binding['source_commit'] and
                     row['phase'] == 'preparation_qualification')
        require(repair <= binding['config']['budget']['gpu_repair_hours'] + 1e-9, 'approved GPU repair envelope exhausted')
        # Native outstanding jobs are already subtracted by availability().
        unaccepted = sum(row['hours'] for row in proof['rows'] if row['job_id'] is None)
        snap = snapshot()
        available = availability(binding['config'], snap)
        closeout = max(0., limits['finalization'] - proof['totals'].get('finalization', 0.))
        require(unaccepted + closeout <= available['available_hours'], 'original live entitlement does not cover reservations')
        check_storage(binding)
        receipts = []
        for card in cards:
            identity = digest(card)
            publish(control / 'cards' / (identity + '.json'), card)
            path = control / 'reservations' / (identity + '.json')
            if path.exists():
                receipts.append(verified(path))
                continue
            receipts.append(publish(path, {'card_sha256': identity, 'phase': card['budget_phase'],
                'hours': card['minutes'] / 60 * card['billing_units'] * binding['config']['budget']['margin'],
                'includes_margin': True, 'attempt': card['attempt'],
                'actual_plus_outstanding_hours': proof['totals'][card['budget_phase']],
                'limit_hours': limits[card['budget_phase']], 'original_allowance_reset': False}))
        return receipts


def reserve(binding, card):
    return reserve_many(binding, [card])[0]


def remaining_reader_minutes(binding):
    available = LIMITS['preparation_reader'] - ledger(binding)['totals'].get('preparation_reader', 0.)
    units = binding['config']['slurm']['gpu']['billing_units']
    margin = binding['config']['budget']['margin']
    return max(0, int((available + 1e-9) * 60 / (units * margin)))


def check_storage(binding):
    from .resources import quota_bytes
    from .scheduler import command
    reserve = binding['config']['budget']['reserve_gib_per_filesystem'] * 1024**3
    available = {}
    for name, project, mount in [('work', '20149688', '/leonardo_work'),
                                 ('fast', '22149688', '/leonardo_scratch/fast')]:
        available[name] = quota_bytes(command(['lfs', 'quota', '-p', project, mount]), mount)
        require(available[name] > reserve, name + ' separate storage reserve exhausted')
    return available


def freeze_predecessor(path: Path, records: list[dict], bulk_root: Path) -> dict:
    """Carry terminal allocations into a separately bound corrected cohort.

    Args:
        path: Preserved predecessor cohort, whose executable sources may differ.
        records: Unchanged complete assignment/disposition roster.
        bulk_root: Fresh destination that cannot adopt the failed release's state.

    Returns:
        Immutable predecessor receipt with exact accepted jobs and native actuals.
    """
    from .cohort import read
    from .ownership import TERMINAL_STATES, accounting
    previous = read(path, verify_source=False)
    require(previous['records'] == records, 'replacement cannot change the frozen scientific population')
    require(Path(previous['paths']['bulk']).resolve() != bulk_root.resolve(), 'replacement requires fresh stores')
    inherited = previous.get('predecessor') or {}
    files = dict(inherited.get('files', {}))
    paths = [Path(path).resolve(), Path(previous['source']['path'])]
    for name in ('SOURCE.tar.gz', 'SOURCE_MATERIALIZED.tar.gz', 'ARCHIVE_VERIFICATION.json',
                 'TERMINAL.json', 'ALLOCATED_TESTS.json', 'ENGINEERING.json', 'OBSERVED_LAUNCH_CLARIFICATION.json'):
        evidence = Path(previous['paths']['control'])/name
        if evidence.exists():
            paths.append(evidence)
    jobs = list(inherited.get('jobs', []))
    for reservation in sorted((Path(previous['paths']['control'])/'schedule').glob('*/RESERVED.json')):
        folder = reservation.parent
        require((folder/'SBATCH.json').exists() and (folder/'ACCEPTED.json').exists(),
                'predecessor has an unresolved reservation; reconcile it before replacement')
        submitted = verified(folder/'SBATCH.json')
        accepted = verified(folder/'ACCEPTED.json')
        require(submitted['job'] == accepted['job'] and verified(reservation)['card'] == accepted['card'],
                'predecessor accepted submission differs')
        jobs.append({'job': accepted['job'], 'card': accepted['card'],
                     'accepted_path': str((folder/'ACCEPTED.json').resolve()),
                     'accepted_sha256': accepted['sha256']})
        paths.extend((reservation, folder/'SBATCH.json', folder/'ACCEPTED.json'))
    require(len({row['job'] for row in jobs}) == len(jobs), 'duplicate inherited allocation')
    observed = [accounting(row['job']) for row in jobs]
    require(all(row['state'] in TERMINAL_STATES for row in observed), 'predecessor allocation is still live')
    files.update({str(item): file_sha(item) for item in paths})
    return {'path': str(Path(path).resolve()), 'sha256': previous['sha256'],
            'files': files, 'jobs': jobs, 'observed_terminal_accounting': observed}


class CohortSchedule:
    """One locked, deadline-bound ledger for the additional causal rescue budget.

    Full limits are reserved before sbatch. An unresolved intent keeps its entire
    reservation until native job-ID reconciliation; it is never resubmitted.
    """

    def __init__(self, cohort_path: Path, run=None):
        """Load the frozen cohort and injectable native scheduler runner."""
        from .cohort import read
        from .scheduler import command
        self.path = Path(cohort_path).resolve()
        self.value = read(self.path)
        self.root = Path(self.value['paths']['control'])
        self.run = run or command

    def card(self, stage: str, model: str | None) -> dict:
        """Return a resource card drawn only from the approved stage limits."""
        limits = {'validate': 60, 'prepare': 60, 'analysis': 60, 'engineering': 120,
                  'generate': 360, 'pilot': 60, 'wave-0': 360, 'wave-1': 360, 'reserve': 540}
        require(stage in limits, 'unknown causal rescue stage')
        require((stage == 'engineering' and model in ('gpt', 'qwen36')) or
                (stage == 'generate' and model == 'qwen36') or
                (stage not in ('engineering', 'generate') and model is None), 'stage/model roster differs')
        gpu = stage not in ('validate', 'prepare', 'analysis')
        card = {
            'cohort_sha256': self.value['sha256'],
            'stage': stage,
            'model': model,
            'minutes': limits[stage],
            'gpus': 2 if gpu else 0,
            'cpus': 16 if gpu else 4,
            'memory_gib': 120 if gpu else 24,
            'partition': 'boost_usr_prod',
            'qos': 'normal',
        }
        if stage == 'engineering':
            # Preserve each model's original four-GPU-hour engineering budget,
            # including failed startup time in every predecessor release.
            import math
            spent = sum(row['charged'] for row in self._predecessor_usage()
                        if row['stage'] == 'engineering' and row['model'] == model)
            remaining_minutes = math.floor((ENGINEERING_GPU_HOURS_PER_MODEL-spent)*60/card['gpus'] + 1e-9)
            require(remaining_minutes > MODEL_LOAD_RESERVE_MINUTES,
                    'model engineering budget cannot cover the minimum load reserve')
            card['minutes'] = min(card['minutes'], remaining_minutes)
        card['job_name'] = 'causal-'+stage+'-'+(model or 'all')+'-'+digest(card)[:10]
        if stage == 'reserve':
            card['operation'], card['operation_model'] = self._reserve_target()
            card['job_name'] += '-'+card['operation']
        return card

    def _reserve_target(self) -> tuple:
        """Choose the first eligible unfinished stage, using no scientific effects."""
        existing = self.root/'schedule'/'reserve-all'/'RESERVED.json'
        if existing.exists():
            card = verified(existing)['card']
            return card['operation'], card['operation_model']
        rows = {(row['stage'], row['model']): row for row in self.status()['records']}
        for stage, model in [('engineering', 'gpt'), ('engineering', 'qwen36'), ('generate', 'qwen36'),
                             ('pilot', None), ('wave-0', None), ('wave-1', None)]:
            row = rows.get((stage, model))
            require(row is not None and row['usage'] is not None, 'reserve cannot replace an unattempted base stage')
            usage = row['usage']
            if stage == 'engineering':
                completed = (self.root/'engineering'/model/'QUALIFIED.json').exists()
            elif stage == 'generate':
                from .driver import Store
                store = Store(Path(self.value['paths']['bulk'])/'driver'/'qwen36'/'rescue'/'units')
                completed = all(store.state(uid) is not None for uid in self.value['generation_uids'])
            else:
                plan = verified(self.root/'READER_WAVES.json')
                from moe_exp.io import Units
                units = Units(Path(self.value['paths']['reader'])/'annotations')
                completed = all(units.state(uid) is not None and units.state(uid)['status'] != 'output'
                                for uid in plan[stage])
            if completed:
                continue
            require(usage['state'] in TERMINAL, 'reserve predecessor is not terminal')
            require(usage['exit_code'] == '2:0' or usage['state'] in ('TIMEOUT', 'NODE_FAIL', 'PREEMPTED'),
                    'software failure halts reserve release')
            return stage, model
        raise ValueError('no eligible unfinished stage for the single reserve')

    def status(self) -> dict:
        """Charge actual terminal use and full outstanding or ambiguous reservations."""
        from .ownership import TERMINAL_STATES, accounting
        totals, rows = {'gpu_hours': 0., 'cpu_core_hours': 0.}, []
        prior_rows = self._predecessor_usage()
        for row in prior_rows:
            totals[row['budget']] += row['charged']
        for path in sorted((self.root/'schedule').glob('*/RESERVED.json')):
            reservation = verified(path)
            card = reservation['card']
            accepted = path.parent/'SBATCH.json'
            usage = accounting(verified(accepted)['job'], self.run) if accepted.exists() else None
            seconds = (usage['elapsed_seconds'] if usage is not None and usage['state'] in TERMINAL_STATES
                       else card['minutes']*60)
            name = 'gpu_hours' if card['gpus'] else 'cpu_core_hours'
            hours = seconds/3600*(card['gpus'] or card['cpus'])
            totals[name] += hours
            rows.append({'stage': card['stage'], 'model': card['model'], 'job': usage['job'] if usage else None,
                         'charged': hours, 'budget': name, 'usage': usage, 'card': card})
        return {'totals': totals, 'records': rows, 'predecessor_records': prior_rows, 'historical_failures_reset': False,
                'scope': 'additional 64 GPU-hours and 16 CPU-only core-hours, historical usage separate'}

    def _predecessor_usage(self) -> list[dict]:
        """Refresh terminal predecessor charges without making them new prerequisites."""
        from .ownership import TERMINAL_STATES, accounting
        rows = []
        for record in (self.value.get('predecessor') or {}).get('jobs', []):
            accepted = verified(Path(record['accepted_path']))
            require(accepted['sha256'] == record['accepted_sha256'] and accepted['job'] == record['job']
                    and accepted['card'] == record['card'], 'inherited acceptance differs')
            usage = accounting(record['job'], self.run)
            require(usage['state'] in TERMINAL_STATES, 'inherited allocation is not terminal')
            card = record['card']
            rows.append({
                'job': record['job'],
                'stage': card['stage'],
                'model': card['model'],
                'budget': 'gpu_hours' if card['gpus'] else 'cpu_core_hours',
                'charged': usage['elapsed_seconds']/3600*(card['gpus'] or card['cpus']),
                'usage': usage,
                'card': card,
            })
        return rows

    def _dependencies(self, card: dict) -> list[str]:
        """Verify native terminal prerequisites and actual successful artifacts."""
        from .ownership import TERMINAL_STATES
        state = self.status()
        by_stage = {(row['stage'], row['model']): row for row in state['records']}
        for row in state['records']:
            if row['stage'] == 'reserve':
                by_stage[(row['card']['operation'], row['card']['operation_model'])] = row
        stage = card['stage']
        if stage == 'analysis':
            require(all(row['usage'] is not None and row['usage']['state'] in TERMINAL_STATES
                        for row in state['records'] if row['stage'] != 'analysis'),
                    'final analysis requires every accepted science owner to be terminal')
        prerequisites = {
            'validate': [], 'engineering': [('validate', None)],
            'generate': [('engineering', 'gpt'), ('engineering', 'qwen36')],
            'prepare': [('generate', 'qwen36')], 'pilot': [('prepare', None)],
            'wave-0': [('pilot', None)], 'wave-1': [('wave-0', None)],
            'analysis': [],
        }
        if stage == 'reserve':
            operation, model = card['operation'], card['operation_model']
            row = by_stage[(operation, model)]
            return [row['job']]
        jobs = []
        for key in prerequisites[stage]:
            require(key in by_stage and by_stage[key]['usage'] is not None, f'prerequisite not accepted: {key}')
            row = by_stage[key]
            accepted = self.root/'schedule'/(row['stage']+'-'+(row['model'] or 'all'))/'ACCEPTED.json'
            require(accepted.exists(), f'prerequisite has no verified acceptance: {key}')
            receipt = verified(accepted)
            require(receipt['job'] == row['job'] and receipt['card']['cohort_sha256'] == self.value['sha256'],
                    'prerequisite acceptance job/cohort differs')
            contingent = stage in ('engineering', 'generate') and row['usage']['state'] not in TERMINAL_STATES
            if contingent:
                jobs.append(row['job'])
                continue
            require(row['usage']['state'] in TERMINAL_STATES, f'prerequisite is not terminal: {key}')
            allowed = {'0:0', '2:0'} if key[0] in ('generate', 'pilot', 'wave-0') else {'0:0'}
            require(row['usage']['exit_code'] in allowed, f'upstream software failure halts release: {key}')
            require(row['usage']['state'] == 'COMPLETED' or row['usage']['exit_code'] == '2:0',
                    f'cancelled or interrupted prerequisite requires operational adjudication: {key}')
            jobs.append(row['job'])
        if stage == 'generate' and all(by_stage[key]['usage']['state'] in TERMINAL_STATES for key in prerequisites[stage]):
            for model in ('gpt', 'qwen36'):
                gate = verified(self.root/'engineering'/model/'QUALIFIED.json')
                require(gate['status'] == 'complete' and gate['cohort_sha256'] == self.value['sha256'], 'engineering gate absent')
        if stage.startswith('wave-'):
            admission = verified(self.root/'READER_ADMISSION.json')
            require(admission['status'] == 'admitted', 'measured reader admission failed')
        return jobs

    def _store(self, card: dict) -> tuple:
        """Locate the canonical worker store and its fixed execution identity."""
        from .cohort import identity
        stage, model = card['stage'], card['model']
        if stage == 'reserve':
            stage, model = card['operation'], card['operation_model']
        bulk = Path(self.value['paths']['bulk'])
        if stage == 'engineering':
            manifest = verified(self.root/'ENGINEERING.json')
            uids = [row['uid'] for row in manifest['records']+manifest['canaries'] if row['model'] == model]
            return bulk/'engineering'/model, identity(self.value, model, 'engineering', uids)
        if stage == 'generate':
            return bulk/'driver'/model/'rescue', identity(self.value, model, 'generation', self.value['generation_uids'])
        if stage in ('pilot', 'wave-0', 'wave-1'):
            tasks = verified(self.root/'READER_TASKS.json')
            return Path(self.value['paths']['reader']), identity(self.value, 'reader', 'annotation',
                                                               [row['uid'] for row in tasks['records']])
        if stage in ('validate', 'prepare', 'analysis'):
            role = 'analysis' if stage == 'analysis' else 'extraction'
            return Path(self.value['paths'][role]), identity(self.value, 'cpu', role,
                    [row['assignment']['uid'] for row in self.value['records']])
        return self.root/'phases'/stage, identity(self.value, 'cpu', stage, [row['assignment']['uid'] for row in self.value['records']])

    def _preflight(self) -> dict:
        """Refresh native credits, grants, partition and separate filesystem reserves."""
        import socket
        import getpass
        require(getpass.getuser() == 'lmolfett' and 'leonardo' in socket.getfqdn(), 'wrong cluster/account')
        balance = self.run(['/cineca/bin/saldo', '-b'])
        fields = next(line.split() for line in balance.splitlines() if line.strip().startswith('IscrC_MIOSR '))
        total, consumed = float(fields[3]), float(fields[4])
        # Two-GPU allocations also reserve sixteen CPU billing units. The full
        # additional envelope is 512 local hours plus sixteen CPU-only hours.
        require(total-consumed >= 528 and consumed+528 <= 42050,
                'live credit or inherited study ceiling does not cover the full additional envelope')
        associations = self.run(['sacctmgr', '-nP', 'show', 'assoc', 'where', 'user=lmolfett',
                                 'format=Account,Partition,QOS,DefaultQOS'])
        require('iscrc_miosr' in associations and 'normal' in associations, 'native association/QoS missing')
        partition = self.run(['scontrol', 'show', 'partition', 'boost_usr_prod'])
        require('State=UP' in partition, 'Booster partition unavailable')
        qos = self.run(['sacctmgr', '-nP', 'show', 'qos', 'where', 'name=normal',
                        'format=Name,MaxWall,MaxJobsPU,MaxSubmitPU,MaxTRESPU,UsageFactor'])
        require(any(line.startswith('normal|') for line in qos.splitlines()), 'normal QoS details unavailable')
        storage = check_storage({'config': {'budget': {'reserve_gib_per_filesystem': 10}}})
        require(storage['work'] > 14*1024**3, 'WORK cannot retain the 10-GiB reserve plus 4-GiB output forecast')
        return {'balance': balance, 'associations': associations, 'partition': partition, 'qos': qos, 'storage': storage}

    def submit(self, stage: str, model: str | None = None) -> dict:
        """Reserve, submit and verify one authorized stage before releasing its worker.

        Args:
            stage: Approved stage name; an accepted stage cannot be redrawn.
            model: Required only for engineering and generation.

        Returns:
            Exact accepted job/resource/owner receipt or an unresolved reservation.
        """
        from .ownership import accept, allocation, check_time
        card = self.card(stage, model)
        folder = self.root/'schedule'/(stage+'-'+(model or 'all'))
        with locked(self.root, 'cohort-submissions'):
            if (folder/'ACCEPTED.json').exists():
                return verified(folder/'ACCEPTED.json')
            check_time(self.value, seconds=card['minutes']*60, reporting=stage == 'analysis')
            dependencies = self._dependencies(card)
            state = self.status()
            totals = state['totals'].copy()
            budget = 'gpu_hours' if card['gpus'] else 'cpu_core_hours'
            if not (folder/'RESERVED.json').exists():
                totals[budget] += card['minutes']/60*(card['gpus'] or card['cpus'])
            require(totals['gpu_hours'] <= 64 and totals['cpu_core_hours'] <= (16 if stage == 'analysis' else 12),
                    'cohort resource ceiling or protected final-analysis reserve would be exceeded')
            preflight = self._preflight()
            if (folder/'RESERVED.json').exists():
                require(verified(folder/'RESERVED.json')['card'] == card, 'reserved resource card changed')
            else:
                publish(folder/'RESERVED.json', {'card': card, 'full_limit_hours': card['minutes']/60*(card['gpus'] or card['cpus']),
                                                'preflight': preflight, 'totals_at_reservation': totals})
            logs = self.root/'logs'
            logs.mkdir(parents=True, exist_ok=True, mode=0o700)
            args = ['sbatch', '--parsable', '--no-requeue', '--account=iscrc_miosr', '--partition='+card['partition'],
                    '--qos='+card['qos'], '--nodes=1', '--ntasks=1', '--cpus-per-task='+str(card['cpus']),
                    '--mem='+str(card['memory_gib'])+'G', '--time='+str(card['minutes']),
                    '--job-name='+card['job_name'], '--chdir='+str(Path(__file__).resolve().parents[3]),
                    '--deadline='+self.value['report_deadline' if stage == 'analysis' else 'science_cutoff'][:19],
                    '--output='+str(logs/'%x-%j.out'), '--error='+str(logs/'%x-%j.err')]
            if card['gpus']:
                args += ['--gres=gpu:'+str(card['gpus'])]
            if dependencies and stage in ('engineering', 'generate'):
                args += ['--dependency=afterok:'+':'.join(dependencies), '--kill-on-invalid-dep=yes']
            args += [str(Path(__file__).resolve().parents[3]/'scripts/slurm/rescue_stage.sbatch'),
                     str(self.path), stage, model or 'all']
            if (folder/'INTENT.json').exists() and not (folder/'SBATCH.json').exists():
                intent = verified(folder/'INTENT.json')
                raw = self.run(['sacct', '-nPX', '-S', intent['date'], '--name='+card['job_name'], '--format=JobIDRaw'])
                found = sorted(set(line.strip('| ') for line in raw.splitlines() if line.strip('| ').isdigit()))
                require(len(found) == 1, 'ambiguous sbatch intent retained; reconcile exact job ID before proceeding')
                publish(folder/'SBATCH.json', {'job': found[0]})
            if not (folder/'SBATCH.json').exists():
                publish(folder/'INTENT.json', {'argv': args, 'date': time.strftime('%Y-%m-%d'), 'dependencies': dependencies})
                raw = self.run(args)
                require(re.fullmatch(r'\d+(;[\w.-]+)?', raw), 'ambiguous sbatch response; full reservation retained')
                publish(folder/'SBATCH.json', {'job': raw.split(';')[0]})
            job = verified(folder/'SBATCH.json')['job']
            fields = allocation(job, card, self.run)
            store, execution = self._store(card)
            owner = accept(store, execution, job, card, run=self.run)
            return publish(folder/'ACCEPTED.json', {'job': job, 'card': card, 'fields': fields, 'owner': owner})
