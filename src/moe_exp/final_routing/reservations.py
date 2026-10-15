"""Transactional allocation reservations, including every reserved recovery."""
from pathlib import Path
from .common import digest, locked, publish, require, verified


def reserve(binding, card):
    return reserve_many(binding,[card])[0]


def reserve_many(binding, cards):
    """Check the entire next wave under one lock before any acceptance."""
    root = Path(binding['root'])
    folder = root/'reservations'
    with locked(root,'reservations'):
        existing={p.stem:verified(p) for p in folder.glob('*.json')}
        totals={}
        for receipt in existing.values():totals[receipt['phase']]=totals.get(receipt['phase'],0)+receipt['hours']
        limits={'preparation_qualification':1280.,'preparation_reader':80.,'preparation_cpu':40.,
                'finalization':binding['config']['budget']['closeout_hours']}
        if any(c['budget_phase']=='production' for c in cards):
            admission = verified(root/'ADMISSION.json')
            require(admission['status'] == 'admitted', 'complete study was not admitted')
            limits['production']=admission['production_allocation_hours']
        pending={}
        for card in cards:
            identity=digest(card);phase=card['budget_phase']
            if identity in existing or identity in pending:continue
            require(phase in limits,'unknown reservation phase')
            amount=card['minutes']/60*card['billing_units']*binding['config']['budget']['margin']
            totals[phase]=totals.get(phase,0)+amount
            require(totals[phase]<=limits[phase]+1e-9,'transactional '+phase+' reservation exhausted')
            pending[identity]={'card_sha256':identity,'phase':phase,'hours':amount,
                'aggregate_reserved_hours':totals[phase],'limit_hours':limits[phase],
                'includes_margin':True,'attempt':card['attempt']}
        for identity,receipt in pending.items():existing[identity]=publish(folder/(identity+'.json'),receipt)
        return [existing[digest(card)] for card in cards]


def check_storage(binding):
    from .resources import quota_bytes
    from .scheduler import command
    reserve = binding['config']['budget']['reserve_gib_per_filesystem']*1024**3
    available = {}
    for name, project, mount in [('work','20149688','/leonardo_work'),('fast','22149688','/leonardo_scratch/fast')]:
        available[name] = quota_bytes(command(['lfs','quota','-p',project,mount]),mount)
        require(available[name] > reserve, name+' separate storage reserve exhausted')
    return available
