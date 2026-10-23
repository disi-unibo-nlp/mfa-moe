"""CPU integration and real two-model GPU checks before qualification release."""
from pathlib import Path
import io
import os
import tempfile
import time
import unittest
from moe_exp.io import (
    REPO,
    WORK,
    Units,
    check_files,
    control_root,
    digest,
    engineering_namespace,
    load,
    publish,
    require,
    verified,
)


def fixture(binding):
    from transformers import AutoTokenizer
    from moe_exp.study.renderer import Renderer
    from .lfm import trace_record
    from .contexts import build,measured
    from moe_exp.study.allocation import allocate
    from .export import export
    with tempfile.TemporaryDirectory(prefix='final-routing-fixture-',dir=WORK/'tools/tmp') as temporary:
        root=Path(temporary);bulk=root/'bulk'
        cfg={**binding['config'],'root':str(root),'bulk_root':str(bulk)}
        test={**binding,'root':str(root),'config':cfg}
        row={'model':'lfm25','benchmark':'math500','family':'engineering-fixture','fold':0,
            'native_example':{'dataset':'math500','problem_id':'cpu-fixture','source_problem_id':'cpu-fixture',
                'sample_id':0,'prompt':'What is 2 + 2?','gold_answer':'4'},
            'source':None,'seed':17,'arm':'native','class':None,'depth_third':None,'dose':'central',
            'action':None,'pulse_tokens':1,'horizon':64,'prefix_status':'enrolled','nomination_status':'nominated'}
        row['uid']=digest(row)
        tokenizer=AutoTokenizer.from_pretrained(binding['models']['lfm25']['local_path'],local_files_only=True)
        ids=tokenizer.encode('We compute the sum. Two plus two equals four. </think>\\boxed{4}',add_special_tokens=False)
        units=Units(bulk/'lfm25/executions');units.intent(row,'CPU-FIXTURE');units.admitted(row,'CPU-FIXTURE')
        raw=units.output(row,{'token_ids':ids,'finish_reason':'stop','status':'complete','engineering_only':True})
        trace=trace_record(test,row,raw,tokenizer)
        require(trace['metadata']['token_replay']['completion_token_ids']==ids,'fixture replay lost native token IDs')
        contexts=build(test,'lfm25',[row]);require(contexts['tasks'],'fixture produced no reader contexts')
        renderer=Renderer(load(Path(binding['source_campaign'])/'CAMPAIGN.json'))
        labels=Units(bulk/'reader/annotations')
        try:
            for task in contexts['tasks']:
                conversation=renderer.request('render',inputs=task['inputs'])['conversation']
                require(conversation and all('content' in m for m in conversation),'original renderer input contract failed')
                parsed=renderer.request('parse',content='[[ ## label ## ]]\nImplement\n[[ ## completed ## ]]')
                require(parsed.get('label')=='Implement','original parser fixture failed')
                labels.intent(task,'CPU-FIXTURE');labels.commit(task,{'status':'complete','label':parsed['label'],'engineering_only':True})
        finally:renderer.close()
        outcome=measured(test,row,contexts['records'][0])
        require(outcome['strict_correct'] is True,'fixture final-answer scoring failed')
        require(outcome['first']['Implement']==[1,1],'fixture first-complete-sentence endpoint failed')
        require(outcome['occupancy']['unknown']==0,'fixture closure left unknown occupancy')
        units.commit(row,{'status':'complete','raw_sha256':raw['sha256'],'engineering_only':True})
        prefix=load(Path(binding['source_campaign'])/'gpt/PREFIXES.json')['records']
        publish(root/'ALLOCATION.json',allocate(prefix,cfg['seed']))
        publish(root/'lfm25/ASSIGNMENTS-full.json',{'records':[row]})
        result=export(test,'full')
        require(not result['required_results_complete'],'one fixture cannot imply complete eight-model results')
        manifest=load(Path(result['path'])/'paper_results_manifest.json')
        require(all(Path(s['path']).is_file() for a in manifest['artifacts'] for s in a['sources']), 'fixture export has broken provenance')
        return {'status':'complete','scientific_draws':0,'real_local_tokenizer':True,'original_renderer_and_parser':True,
            'contexts':len(contexts['tasks']),'raw_to_export_passed':True,'partial_eight_model_completion_rejected':True}



def validate(binding, allocated=False):
    import importlib
    import subprocess
    from .config import study_files
    if allocated:
        from .resources import snapshot
        job_id = os.environ['SLURM_JOB_ID']
        attempt = job_id + '-' + os.environ.get('SLURM_RESTART_COUNT', '0')
        publish(control_root(binding) / ('ALLOCATION_PREFLIGHT-' + attempt + '.json'),
                {**snapshot(), 'job_id': job_id, 'source_commit': binding['source_commit']})
    files = study_files()
    for path in (REPO / 'src/moe_exp/study').glob('*.py'):
        if path.stem != '__main__':
            importlib.import_module('moe_exp.study.' + path.stem)
    for path in [*(REPO / 'scripts/study').glob('*.sh'), *(REPO / 'scripts/inference').glob('*.sh')]:
        subprocess.run(['bash', '-n', str(path)], check=True)
    loader = unittest.TestLoader()
    suite = loader.discover(str(REPO / 'tests/study'), top_level_dir=str(REPO / 'tests/study'))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    require(result.wasSuccessful(), stream.getvalue())
    evidence = fixture(binding)
    check_files(files)
    proof = publish(control_root(binding) / 'CPU_VALIDATION.json', {
        'status': 'complete', 'allocated': allocated, 'tests_run': result.testsRun,
        'test_output': stream.getvalue(), 'fixture': evidence, 'source_files': files,
        'slurm_job_id': os.environ.get('SLURM_JOB_ID')})
    return {'status': 'complete', 'tests_run': result.testsRun, 'proof_sha256': proof['sha256']}


def native_output(backend, row, cap):
    from .runtime import prompt_prefix, sampling_parameters, remaining
    from .session import request_session
    prompt, prefix, _ = prompt_prefix(backend.binding, backend.model, row, backend.tokenizer)
    params = sampling_parameters(backend.profile, row, prefix, min(cap, remaining(backend.profile, prompt, prefix)))
    outputs = []
    with request_session(backend.llm, [{'uid': row['uid']}]) as session:
        session.admit(row['uid'], {'prompt_token_ids': prompt + prefix}, params)
        session.bind()
        while backend.llm.llm_engine.has_unfinished_requests():
            require(time.time() < backend.deadline, 'native baseline deadline')
            outputs.extend(session.step())
        session.finish()
    require(len(outputs) == 1 and outputs[0].finished, 'native reference lacks terminal output')
    return list(outputs[0].outputs[0].token_ids)


def canary_row(row, role):
    value = {k: v for k, v in row.items() if k != 'uid'}
    value['engineering_role'] = role
    value['uid'] = digest(value)
    return value


def engine_check(binding, deadline):
    from .qualification import probe_rows
    from .runtime import Backend, prompt_prefix
    observations = []
    bulk = Path(binding['config']['bulk_root'])
    for model in ('lfm25', 'gpt'):
        require(time.time() + 600 < deadline, 'engineering load reserve exhausted')
        backend = Backend(binding, model, 1, deadline, replay=True)
        units = Units(bulk / model / engineering_namespace(binding, 'canary'))
        try:
            native = next(r for r in probe_rows(binding, model, capacity=1) if r['arm'] == 'native')
            row = canary_row(native, 'unhooked-native-fidelity')
            reference = native_output(backend, row, 64)
            hooked = backend.run([row], units, engineering_tokens=64)[0]
            require(hooked['token_ids'] == reference, 'native hook changes continuation tokens')
            noop = canary_row({**native, 'action': {'layer': backend.profile['layers'][0],
                    'experts': [0], 'direction': 'activate', 'start': 0, 'pulse_tokens': 0}}, 'zero-pulse-noop-fidelity')
            noop_raw = backend.run([noop], units, engineering_tokens=64)[0]
            require(noop_raw['token_ids'] == reference, 'zero-pulse hook changes continuation tokens')
            prompt, prefix, _ = prompt_prefix(binding, model, native, backend.tokenizer)
            cap = backend.profile['card']['max_model_len'] - 16
            cycle = prompt + prefix
            expanded = (cycle * ((cap + len(cycle) - 1) // len(cycle)))[:cap]
            stress = canary_row({**native, 'engineering_input': {
                'prompt': expanded[:-1], 'prefix': expanded[-1:]}, 'horizon': 16}, 'full-native-context')
            backend.run([stress], units, engineering_tokens=16)
            replay_trace = {'metadata': {'token_replay': {'prompt_token_ids': prompt,
                'completion_token_ids': prefix + hooked['token_ids']}}, 'engineering_only': True}
            replay_path = bulk / model / engineering_namespace(binding, 'canary-replay') / 'native.npz'
            replay = backend.capture_trace(replay_trace, replay_path)
            full_trace = {'metadata': {'token_replay': {'prompt_token_ids': expanded[:-1],
                'completion_token_ids': expanded[-1:]}}, 'engineering_only': True}
            full_replay = backend.capture_trace(full_trace, replay_path.with_name('full-context.npz'))
            single = {'native_tokens_exact': True, 'zero_pulse_tokens_exact': True,
                      'full_context_uid': stress['uid'], 'replay_sha256': replay['sha256'],
                      'full_context_replay_sha256': full_replay['sha256']}
        finally:
            backend.close()
        require(time.time() + 600 < deadline, 'mixed-request load reserve exhausted')
        backend = Backend(binding, model, 2, deadline)
        try:
            rows = [canary_row(row, 'mixed-request-count-and-direction')
                    for row in probe_rows(binding, model, capacity=2)]
            rows.sort(key=lambda row: digest(['reverse-engineering-order', row['uid']]), reverse=True)
            for offset in range(0, len(rows), 2):
                batch = rows[offset:offset + 2]
                backend.run(batch, units, engineering_tokens=max(64, max(r['pulse_tokens'] for r in batch) + 16))
            boundaries = {}
            for row in rows:
                capture = verified(units.folder(row['uid']) / 'CAPTURE.json')
                require(capture['stats']['noop_exact'] and capture['stats']['target_contract'] and
                        capture['stats']['finite_weights'] and capture['native_validation'], 'routing fidelity failed')
                if not row['action']:
                    continue
                key = (len(row['action']['experts']), row['action']['direction'])
                evidence = boundaries.setdefault(key, {'start': False, 'end': False})
                for record in capture['capture'].values():
                    evidence['start'] |= any(p == 0 and active for p, active in zip(record['positions'], record['active']))
                    evidence['end'] |= any(p == row['pulse_tokens'] and not active for p, active in zip(record['positions'], record['active']))
            require(len(boundaries) == 6 and all(v['start'] and v['end'] for v in boundaries.values()),
                    'actual count/direction pulse coverage is incomplete')
            abort = backend.abort_probe(rows[0])
            require(abort['native_id_randomization_observed'], 'actual randomized internal ID was not exercised')
            observations.append({'model': model, 'status': 'complete', **single,
                'mixed_requests': len(rows), 'pulse_evidence': [dict(count=k[0], direction=k[1], **v) for k, v in boundaries.items()],
                'abort': abort, 'memory': backend.metrics()})
        finally:
            backend.close()
    proof = publish(control_root(binding) / 'ENGINE_VALIDATION.json', {
        'status': 'complete', 'models': observations, 'source_commit': binding['source_commit'],
        'scientific_draws': 0, 'slurm_job_id': os.environ['SLURM_JOB_ID']})
    return {'status': 'complete', 'proof_sha256': proof['sha256'], 'models': ['lfm25', 'gpt']}
