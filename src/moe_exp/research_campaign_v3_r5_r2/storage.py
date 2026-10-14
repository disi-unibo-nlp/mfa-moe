"""User-authorized private FAST bulk storage, with separate volume budgets."""
from pathlib import Path
import os
from .contracts import FAST_PRIVATE, MODELS, STORAGE_RESERVE, private_path, require

BULK = (*MODELS, 'output', 'snapshots', 'deliveries', 'storage_measurements')


def initialize(root,parent=None):
    root = private_path(root)
    target = FAST_PRIVATE / root.name
    require(root.name.startswith('research-campaign-v3-'), 'invalid FAST campaign identity')
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    require(target.stat().st_uid == os.getuid() and target.resolve()==target
            and target.parent.stat().st_uid==os.getuid(), 'FAST owner/resolved destination differs')
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    mappings = {}
    for name in BULK:
        dest = target / name
        dest.mkdir(exist_ok=True, mode=0o700)
        link = root / name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(dest, target_is_directory=True)
        require(link.is_symlink() and link.resolve() == dest, 'conflicting bulk storage mapping')
        mappings[name] = str(dest)
    from .contracts import REPO,load,file_sha
    source=REPO/'report/storage-recovery-20261012-r1/PLAN.json'
    record=load(source);jobs={}
    for entry in record['entries']:
        jobs[entry['job_id']]=jobs.get(entry['job_id'],0)+entry['allocated_bytes']
    observed_cache=max(jobs.values())
    old=Path(parent['root']) if parent else root
    samples=[p for folder in ('cards','submissions','handoffs') for p in sorted((old/folder).glob('**/*.json'))[:128]]
    largest=max((p.stat().st_size for p in samples),default=65536)
    scientific_receipt=max((p.stat().st_size for p in (old/'stages').glob('*/RESULT.json')),default=65536)
    # Finite receipts plus compact export copies. Recompute dynamically when
    # measured execution shards make the graph larger.
    metadata=largest*4*(2*(899+7*32)+8)+7*4*scientific_receipt
    exports=[p.stat().st_size for p in (old/'output').rglob('*') if p.is_file() and not p.is_symlink()]
    export_peak=sum(exports)+max(exports,default=0)
    paper=REPO/'paper/6a510c6db0d9b08aa438c142'
    paper_bytes=sum(p.stat().st_size for p in paper.rglob('*') if p.is_file() and
                    '.git' not in p.relative_to(paper).parts and p.suffix in ('.tex','.cls','.bib','.pdf','.svg','.png'))
    return {'bulk_root': str(target), 'mappings': mappings,
            'quota_projects': {'WORK': 20149688, 'FAST': 22149688},
            'reserve_bytes_per_volume': STORAGE_RESERVE,
            'work_metadata_bound_bytes': metadata+5*(paper_bytes+5*256*1024**2),
            'metadata_bytes_per_handoff_upper':largest*4,
            'historical_max_scientific_stage_receipt_bytes':scientific_receipt,
            'fast_transient_cache_bound_bytes':int(observed_cache*1.25)*24,
            'historical_cache_measurement':{'path':str(source),'sha256_file':file_sha(source),
                'maximum_terminal_allocation_cache_bytes':observed_cache,'allocation_count':len(jobs)},
            'runtime_cache_peak_requirement':'sampled during every allocated worker; required retained role peaks before production',
            'concurrent_gpu_cache_slots':24,'export_copy_lifetimes':5,
            'measured_report_peak_bytes':export_peak,'measured_paper_snapshot_bytes':paper_bytes,
            'retention_destination':str(root/'RETAINED_EVIDENCE.json'),
            'inherited_artifacts': 'verified references; no bulk copies'}


def verify_layout(campaign):
    policy = campaign.get('storage_policy')
    if not policy:
        return
    root = Path(campaign['root'])
    expected = FAST_PRIVATE / root.name
    require(Path(policy['bulk_root']) == expected, 'sealed FAST root differs')
    for name in BULK:
        dest = expected / name
        link = root / name
        require(policy['mappings'][name] == str(dest) and link.is_symlink()
                and link.resolve() == dest and dest.stat().st_uid == os.getuid(),
                'bulk storage mapping/owner changed: ' + name)


def quota_headroom(text, mount):
    """Lustre may put a long filesystem name on its own line."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        fields = line.split()
        if fields and fields[0] == mount:
            numbers = fields[1:] or lines[i+1].split()
            used, quota = (int(v.rstrip('*')) for v in numbers[:2])
            require(quota > 0, 'finite exact quota unavailable: ' + mount)
            return (quota-used)*1024
    raise ValueError('exact quota row missing: ' + mount)


def forecast(campaign,bulk_bytes):
    from .contracts import load
    from .controller import handoff_bound
    policy=campaign['storage_policy'];root=Path(campaign['root'])
    peaks=[load(p)['peak_cache_allocated_bytes'] for p in (root/'storage_measurements').glob('*.json')]
    cache=max([policy['fast_transient_cache_bound_bytes']//24,*peaks])*24
    metadata=max(policy['work_metadata_bound_bytes'],handoff_bound(campaign)*policy['metadata_bytes_per_handoff_upper']+4*256*1024**2)
    # Raw banks/executions are referenced by the exports, never copied five
    # times. Reserve each report's measured table/atomic-write peak separately.
    return {'output_bytes':metadata,'fast_output_bytes':bulk_bytes+cache+5*policy['measured_report_peak_bytes']}


class CacheMeter:
    """Allocated-job lifetime sampling; the thread ends with its worker."""
    def __init__(self,campaign,card):
        import threading
        self.campaign=campaign;self.card=card;self.stop_event=threading.Event();self.peak=0;self.samples=0
        self.path=Path(os.environ['TMPDIR']);self.thread=threading.Thread(target=self.run)
    def sample(self):
        total=0
        for base,dirs,files in os.walk(self.path,followlinks=False):
            dirs[:]=[d for d in dirs if not (Path(base)/d).is_symlink()]
            for name in files:
                try:
                    p=Path(base)/name
                    if not p.is_symlink():total+=p.stat().st_blocks*512
                except FileNotFoundError:pass
        self.peak=max(self.peak,total);self.samples+=1
    def run(self):
        while not self.stop_event.wait(10):self.sample()
    def start(self):self.sample();self.thread.start()
    def stop(self):
        from .contracts import publish,load
        self.stop_event.set();self.thread.join();self.sample()
        path=Path(self.campaign['root'])/'storage_measurements'/(self.card['sha256']+'.json')
        if path.exists():return load(path)
        return publish(path,
            {'card_sha256':self.card['sha256'],'model':self.card['model'],'stage':self.card['stage'],
             'shard_index':self.card['shard_index'],
             'peak_cache_allocated_bytes':self.peak,'samples':self.samples,'sample_interval_seconds':10,
             'job_id':os.environ['SLURM_JOB_ID'],'temporary_cache_path':str(self.path),
             'sampler_is_lower_bound_between_samples':True})
