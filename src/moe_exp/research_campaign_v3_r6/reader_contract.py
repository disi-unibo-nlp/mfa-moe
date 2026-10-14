"""Pinned token inputs and narrowly proven recovery before engine admission."""
from pathlib import Path
import ast
import base64
import csv
import hashlib
from importlib.metadata import distribution
from .contracts import Units, check_files, digest, file_sha, load, publish, require


def token_ids(tokenizer, conversation, config):
    tokens = tokenizer.apply_chat_template(conversation, tokenize=True,
        add_generation_prompt=True, return_dict=False,
        enable_thinking=config['enable_thinking'], preserve_thinking=config['preserve_thinking'],
        reasoning_effort=config['reasoning_effort'])
    require(isinstance(tokens, (list, tuple)) and tokens and
            all(type(t) is int and t >= 0 for t in tokens), 'reader requires integer token IDs')
    return list(tokens)


def runtime_binding(annotator):
    paths = []
    for package, names in {
        'transformers': ['tokenization_utils_base.py'],
        'vllm': ['entrypoints/offline_utils.py', 'entrypoints/llm.py',
                 'v1/engine/llm_engine.py', 'v1/engine/input_processor.py',
                 'reasoning/qwen3_engine_reasoning_parser.py',
                 'entrypoints/openai/chat_completion/protocol.py'],
    }.items():
        dist = distribution(package)
        for name in names:
            path = Path(dist.locate_file(package + '/' + name))
            require(path.is_file(), 'reader runtime source missing: ' + str(path))
            paths.append(path)
        paths += [Path(dist.locate_file(p)) for p in dist.files or ()
                  if str(p).endswith(('.dist-info/METADATA', '.dist-info/RECORD'))]
    for folder in ('vllm/reasoning','vllm/tokenizers','vllm/renderers'):
        paths += list(Path(distribution('vllm').locate_file(folder)).rglob('*.py'))
    paths += [Path(distribution('transformers').locate_file('transformers/utils/chat_template_utils.py'))]
    paths += [p for p in Path(annotator['path']).iterdir() if p.is_file() and
              ('tokenizer' in p.name or 'chat_template' in p.name or p.name == 'config.json')]
    return {str(p): file_sha(p) for p in paths}


def _distribution_source(package, relative, before):
    dist = distribution(package)
    path = Path(dist.locate_file(relative))
    record = next(Path(dist.locate_file(p)) for p in dist.files or () if str(p).endswith('.dist-info/RECORD'))
    with record.open() as stream:
        hashes = {r[0]: r[1] for r in csv.reader(stream) if len(r) > 1}
    hashed = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip('=')
    require(hashes.get(relative) == 'sha256=' + hashed, 'installed reader source differs from distribution')
    require(path.stat().st_mtime < before, 'reader source postdates original execution')
    return {'path': str(path), 'sha256': file_sha(path), 'distribution': dist.version,
            'record_sha256': file_sha(record), 'mtime': path.stat().st_mtime}


def adjudicate(campaign):
    """Absence of an output is insufficient: require this exact input-rejection proof."""
    root = Path(campaign['root']); parent = Path(campaign['resume_from'])
    target = root/'READER_RECOVERY_ADJUDICATION.json'
    if target.exists(): return load(target)
    first = parent/'stages/reader_price-all-00-00000/FAILED_ATTEMPT-0.json'
    table_path = parent/'ANNOTATOR_PRICE_TASKS.json'
    if not first.exists() or not table_path.exists():
        return publish(target, {'records': [], 'proof': 'no matching original pre-admission failure'})
    failure = load(first); table = load(table_path)
    require(failure['job_id'] == '60454682' and failure['error_type'] == 'TypeError' and
            failure['error'] == "'<' not supported between instances of 'str' and 'int'",
            'original reader failure differs from authorized recovery audit')
    trace = failure['traceback']
    require('process_inputs' in trace and '_validate_model_input' in trace and
            '_run_engine' not in trace, 'failure does not prove rejection before engine admission')
    rows = table['records']; require(len(rows) == 64 and {r['model'] for r in rows} == {'nemotron'},
                                    'original reader roster differs')
    records = []; starts = []
    for row in rows:
        folder = parent/row['model']/'annotations'
        state = load(folder/(row['uid']+'.result.json'))
        require(state['assignment'] == row and state['status'] == 'execution_failed' and
                state['error_type'] == failure['error_type'] and state['error'] == failure['error'] and
                not state.get('generated_tokens') and not state.get('raw_completion'),
                'reader assignment contains an ambiguous execution')
        unit_starts = [load(p) for p in folder.glob(row['uid']+'.start-*.json')]
        require(len(unit_starts) == 1 and unit_starts[0]['job_id'] == failure['job_id'] and
                unit_starts[0]['row_sha256'] == digest(row), 'reader start identity differs')
        starts.append(unit_starts[0]['started_epoch'])
        records.append({'model':row['model'], 'uid':row['uid'], 'assignment_sha256':digest(row),
                        'original_result':str(folder/(row['uid']+'.result.json')),
                        'original_result_sha256':file_sha(folder/(row['uid']+'.result.json')),
                        'disposition':'rejected_before_engine_admission', 'recovery_allowed':True})
    sources = [_distribution_source(p, f, min(starts)) for p,f in (
        ('transformers','transformers/tokenization_utils_base.py'),
        ('vllm','vllm/entrypoints/offline_utils.py'),
        ('vllm','vllm/v1/engine/llm_engine.py'),
        ('vllm','vllm/v1/engine/input_processor.py'))]
    base = Path(distribution('vllm').locate_file('vllm'))
    offline = (base/'entrypoints/offline_utils.py').read_text()
    engine = (base/'v1/engine/llm_engine.py').read_text()
    def method(text, name):
        for n in ast.walk(ast.parse(text)):
            if isinstance(n, ast.FunctionDef) and n.name == name: return ast.get_source_segment(text,n)
        raise ValueError('reader admission method missing')
    run = method(offline,'_run_completion'); add = method(engine,'add_request')
    require(run.index('self._add_completion_requests(') < run.index('self._run_engine(') and
            add.index('self.input_processor.process_inputs(') < add.index('self.engine_core.add_request('),
            'installed reader execution ordering changed')
    tokenizer = Path(distribution('transformers').locate_file('transformers/tokenization_utils_base.py')).read_text()
    require('return_dict: bool = True' in method(tokenizer,'apply_chat_template'),
            'original dictionary return default cannot be established')
    old = load(parent/'CAMPAIGN.json')
    scripts = [p for p in old['code_files'] if p.endswith('/job.sbatch')]
    require(len(scripts)==1 and 'VLLM_ENABLE_V1_MULTIPROCESSING=0' in Path(scripts[0]).read_text(),
            'original in-process reader mode unproven')
    annotation = next(Path(p) for p in old['code_files'] if p.endswith('/annotation.py') and '/research_campaign_' in p)
    require('async_scheduling=False' in annotation.read_text(), 'original synchronous reader mode unproven')
    check_files(campaign['reader_runtime_files'])
    return publish(target, {'records':records, 'source_failure':str(first), 'source_failure_sha256':file_sha(first),
        'original_table_sha256':table['sha256'], 'runtime_proofs':sources,
        'proof':'all 64 inputs had the same dictionary representation; the first request failed validation before engine admission',
        'absence_of_output_alone_used':False, 'original_identities_prompts_seeds_preserved':True,
        'sampling_repeated':False, 'ambiguous_executions_remain_unknown':True})


class ReaderUnits(Units):
    def state(self, uid):
        local = self.root/(uid+'.result.json')
        starts = list(self.root.glob(uid+'.start-*.json'))
        proof = self.root.parent.parent/'READER_RECOVERY_ADJUDICATION.json'
        if not local.exists() and not starts and proof.exists():
            for record in load(proof)['records']:
                if record['uid']==uid and record['model']==self.root.parent.name and record['recovery_allowed']:
                    check_files({record['original_result']:record['original_result_sha256']})
                    return None
        return super().state(uid)
