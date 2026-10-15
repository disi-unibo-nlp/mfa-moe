"""One ledger: terminal actuals plus full limits for all outstanding work."""
from pathlib import Path
from .common import submission_paths, reservation_paths, control_root, digest, locked, publish, require, verified

TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}
LIMITS = {'preparation_qualification': 1280., 'preparation_reader': 80.,
          'preparation_cpu': 40.}


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
