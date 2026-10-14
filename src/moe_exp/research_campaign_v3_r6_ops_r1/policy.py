"""Source-verified operational hooks; scientific tables and installed code stay fixed."""
from pathlib import Path

from moe_exp.research_campaign_v3_r6.contracts import (
    REPO, check_files, digest, load, publish, require, seal,
)

WRAPPER = REPO / 'scripts/research_campaign_v3_r6_ops_r1/job.sbatch'
BINDING = 'OPS_R1_BINDING.json'


def check_binding(campaign):
    root = Path(campaign['root'])
    check_files(campaign['code_files'])
    check_files(campaign['source_files'])
    check_files(campaign.get('reader_runtime_files', {}))
    binding = load(root / BINDING)
    require(binding['campaign_sha256'] == campaign['sha256'], 'Operations binding belongs to another campaign')
    check_files(binding['files'])
    check_files(binding['runtime_files'])
    require(binding['remote_writes_allowed'] is False and binding['manuscript_writes_allowed'] is False,
            'Code/report-only operations policy differs')
    return binding


def require_readiness(campaign):
    root = Path(campaign['root'])
    binding = check_binding(campaign)
    readiness = load(root / 'READINESS.json')
    require(readiness.get('passed') is True and readiness.get('source_unchanged_during_tests') is True and
            readiness.get('actual_import_graph_validated') is True, 'Actual full CPU validation has not passed')
    check_files(readiness['files'])
    graph = load(root / 'IMPORT_GRAPH_VALIDATION.json')
    reader = load(root / 'READER_INPUT_VALIDATION.json')
    export = load(root / 'FULL_EXPORT_VALIDATION.json')
    complete = load(root / 'OPS_R1_VALIDATION_COMPLETED.json')
    require(graph['status'] == reader['status'] == export['status'] == 'complete', 'Full validation proof is incomplete')
    require(graph['sha256'] == readiness['actual_validation_sha256'] and
            graph['reader_validation_sha256'] == reader['sha256'] and
            graph['full_export_validation_sha256'] == export['sha256'], 'Validation dependency seal differs')
    require(graph['non_reap_generations'] == 69459 and graph['compact_shapes'] == 346 and
            export['cells'] == 243747 and reader['original_failures_reproduced'] == 64 and
            reader['scientific_draws'] == 0, 'Frozen full-validation coverage differs')
    require(complete['readiness_sha256'] == readiness['sha256'] and
            complete['binding_sha256'] == binding['sha256'] and complete['manuscript_unchanged'] is True,
            'Readiness was not verified under the operations amendment')
    return complete


def verified_local_report(campaign):
    root = Path(campaign['root'])
    path = root / 'LOCAL_REPORT_COMPLETE.json'
    if not path.exists():
        return None
    receipt = load(path)
    report = load(Path(receipt['directory']) / 'REPORT.json')
    require(receipt['complete'] and receipt['report_sha256'] == report['sha256'], 'Local report receipt differs')
    manifest = load(Path(receipt['directory']) / 'EXPORT_FILES.json')
    require(receipt['export_files_sha256'] == manifest['sha256'], 'Local report manifest differs')
    check_files(manifest['files'])
    require(load(root / 'REPORT_COMPLETE.json')['report_sha256'] == report['sha256'], 'Final report pointer differs')
    return receipt


def deliver(campaign, directory, report, *, final=False, phase=None):
    """Commit genuine report completion without manuscript or remote side effects."""
    root, directory = Path(campaign['root']), Path(directory)
    manifest = load(directory / 'EXPORT_FILES.json')
    check_files(manifest['files'])
    if final:
        publish(root / 'SCIENCE_COMPLETE.json', {'complete': report['SCIENCE_COMPLETE'], 'report_sha256': report['sha256']})
        publish(root / 'REPORT_COMPLETE.json', {'complete': True, 'report_sha256': report['sha256']})
    if campaign.get('validation_only'):
        return {'status': 'complete', 'REPORT_COMPLETE': True, 'PAPER_DELIVERED': False,
                'SCIENCE_COMPLETE': report['SCIENCE_COMPLETE'], 'isolated_validation_export': True}
    receipt = publish(directory / 'LOCAL_REPORT.json', {
        'complete': True, 'directory': str(directory), 'report_sha256': report['sha256'],
        'export_files_sha256': manifest['sha256'], 'phase': phase or ('final' if final else 'snapshot'),
        'manuscript_updated': False, 'remote_writes': 0, 'remote_delivery_verified': False,
    })
    if final:
        publish(root / 'LOCAL_REPORT_COMPLETE.json', receipt)
    if phase:
        publish(root / ('PHASE_COMPLETE-' + phase + '.json'), {
            'phase': phase, 'report_sha256': report['sha256'],
            'scientific_cells_complete': report['SCIENCE_COMPLETE'], 'local_report_sha256': receipt['sha256'],
        })
    from .commits import prepare_report
    try:
        prepared = prepare_report(campaign, directory, report)
        commit_status = 'local_commit_prepared'
    except Exception as exc:
        # Commit preparation is independent of durable reporting and science.
        failure = directory / 'LOCAL_COMMIT_FAILURE.json'
        if not failure.exists():
            publish(failure, {'error_type': type(exc).__name__, 'error': str(exc)[:2000],
                             'report_sha256': report['sha256'], 'remote_writes': 0})
        commit_status = 'local_commit_pending'
    return {'status': 'complete', 'REPORT_COMPLETE': final, 'LOCAL_REPORT_COMPLETE': final,
            'PAPER_DELIVERED': False, 'SCIENCE_COMPLETE': report['SCIENCE_COMPLETE'],
            'local_commit_status': commit_status}


def denied(*args, **kwargs):
    raise RuntimeError('Manuscript and remote publication are deferred to the user')


def install(campaign):
    """Install once in each entrypoint process, before importing the sealed launcher."""
    binding = check_binding(campaign)
    from moe_exp.research_campaign_v3_r6 import annotation, controller, export, publication, scheduler
    from .reader import annotate
    if getattr(scheduler.Slurm, '_ops_r1_installed', False):
        require(scheduler.Slurm._ops_r1_binding == binding['sha256'], 'Another operations binding is already installed')
        return
    annotation.annotate = annotate
    export.deliver = deliver
    publication.deliver = publication.build_local = publication.push = publication.push_snapshot = denied
    original_launch, original_control, original_ready = controller.launch, controller.control, controller.ready
    original_command, original_verify, original_card = scheduler.Slurm.command, scheduler.Slurm.verify, scheduler.card

    def launch(value, scheduler=None):
        require_readiness(value)
        return original_launch(value, scheduler)

    def ready(value, scheduler, observed):
        if verified_local_report(value):
            return []
        return original_ready({**value, 'independent_publication': False}, scheduler, observed)

    def control(value, card, scheduler=None):
        require_readiness(value)
        view = {**value, 'independent_publication': False}
        if verified_local_report(value):
            view['results_first_phases'] = False
        return original_control(view, card, scheduler)

    def card(*args, **kwargs):
        value = original_card(*args, **kwargs)
        milestone = Path(value['root']) / 'ops_r1/milestone/ACCEPTED.json'
        if value['stage'] == 'snapshot' and milestone.exists():
            accepted = load(milestone)['snapshot']
            existing = load(Path(value['root']) / 'cards' / (accepted['card_sha256'] + '.json'))
            require(existing['ops_binding_sha256'] == binding['sha256'] and
                    existing['campaign_sha256'] == value['campaign_sha256'] and
                    existing['minutes'] == value['minutes'] and existing['begin'] == value['begin'],
                    'Protected snapshot differs from requested milestone')
            return existing
        value = {k: v for k, v in value.items() if k not in ('sha256', 'comment')}
        value['ops_binding_sha256'] = binding['sha256']
        value['comment'] = 'rc3:' + digest(value)
        return seal(value)

    def command(self, args):
        args = list(args)
        if args[0] == 'sbatch':
            old = str(REPO / 'scripts/research_campaign_v3_r6/job.sbatch')
            require(old in args, 'Unbound scheduler entrypoint rejected')
            args[args.index(old)] = str(WRAPPER)
        return original_command(self, args)

    def verify(self, jid, value):
        require(value.get('ops_binding_sha256') == binding['sha256'], 'Worker card lacks current operations binding')
        proof = original_verify(self, jid, value)
        require(proof['fields']['Command'] == str(WRAPPER), 'Accepted worker lacks code/report-only entrypoint')
        return proof

    controller.launch, controller.control, controller.ready = launch, control, ready
    scheduler.card = controller.card = card
    scheduler.Slurm.command, scheduler.Slurm.verify = command, verify
    scheduler.Slurm._ops_r1_installed = True
    scheduler.Slurm._ops_r1_binding = binding['sha256']
