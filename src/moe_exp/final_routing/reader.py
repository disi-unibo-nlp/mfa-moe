"""Fixed original reader, whole-batch prices and durable engine admission."""
from pathlib import Path
import math
import os
import time
from .common import CLASSES, Units, check_files, digest, load, publish, require


class Reader:
    def __init__(self, binding, deadline):
        from .renderer import Renderer
        from vllm import LLM
        parent = load(Path(binding['source_campaign'])/'CAMPAIGN.json')
        self.binding, self.deadline, self.annotator = binding, deadline, parent['annotator']
        require(self.annotator['config']['max_tokens'] == 16384, 'fixed reader allowance changed')
        require(os.environ.get('SLURM_JOB_GPUS'), 'reader requires allocated GPUs')
        self.renderer = Renderer(parent)
        a = self.annotator
        started = time.monotonic()
        self.llm = LLM(model=a['path'], tokenizer=a['path'], tensor_parallel_size=2, dtype='bfloat16',
            max_model_len=a['native_context'], max_num_seqs=64, enforce_eager=True,
            enable_prefix_caching=True, async_scheduling=False, enable_chunked_prefill=True,
            max_num_batched_tokens=8192, gpu_memory_utilization=.90, kv_cache_dtype='bfloat16',
            attention_config={'backend': 'FLASH_ATTN'}, language_model_only=True, generation_config='vllm',
            worker_extension_cls='moe_exp.final_routing.observer.RoutingWorkerExtension')
        self.load_seconds = time.monotonic()-started
        self.tokenizer = self.llm.get_tokenizer()
        require(type(self.llm.llm_engine.engine_core).__name__ == 'InprocClient', 'reader executor changed')
        self.root = Path(binding['config']['bulk_root'])/'reader'
        self.units = Units(self.root/'annotations')
        self.old_indices = {}

    def parse(self, row, raw):
        from vllm.reasoning import ReasoningParserManager
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        a = self.annotator
        parser = ReasoningParserManager.get_reasoning_parser('qwen3')(self.tokenizer)
        request = ChatCompletionRequest(model=a['config']['judge_model'],
            messages=self.renderer.request('render', inputs=row['inputs'])['conversation'],
            reasoning_effort=a['config']['reasoning_effort'])
        text = self.tokenizer.decode(raw['token_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        _, content = parser.extract_reasoning(text, request)
        parsed = self.renderer.request('parse', content=content or '') if raw['finish_reason'] == 'stop' else {}
        label = parsed.get('label')
        return {'status': 'complete' if label in CLASSES else 'unknown',
                'label': label if label in CLASSES else None, 'raw_sha256': raw['sha256'],
                'finish_reason': raw['finish_reason'], 'generated_tokens': len(raw['token_ids'])}

    def run(self, rows, *, input_bin='all', admission=None):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        from moe_exp.research_campaign_v3_r6.reader_contract import token_ids
        a, engine = self.annotator, self.llm.llm_engine
        preparation_started = time.monotonic()
        pending = []
        for row in rows:
            state = self.units.state(row['uid'])
            if state is None:
                state = self.reuse(row)
            if state and state['status'] == 'output':
                self.units.commit(row, self.parse(row, load(self.units.folder(row['uid'])/'OUTPUT.json')))
            elif state is None:
                conversation = self.renderer.request('render', inputs=row['inputs'])['conversation']
                prompt = token_ids(self.tokenizer, conversation, a['config'])
                require(len(prompt)+16384 <= a['native_context'], 'reader input cannot retain the unchanged 16384-token allowance')
                if admission is not None:
                    require(admission['input_min'] <= len(prompt) <= admission['input_max'], 'unpriced reader input range')
                pending.append((row, prompt))
        preparation_seconds = time.monotonic()-preparation_started
        result = []
        for offset in range(0, len(pending), 64):
            batch = pending[offset:offset+64]
            reserve = admission['batch_seconds'] if admission else 600
            require(time.time()+reserve+60 < self.deadline, 'reader will not admit batch past deadline reserve')
            require(not engine.has_unfinished_requests(), 'neighboring reader batch still active')
            lookup = {r['uid']: r for r, _ in batch}
            tick, outputs = time.monotonic(), {}
            try:
                for row, prompt in batch:
                    params = SamplingParams(**a['decoding'], max_tokens=16384,
                        seed=int(digest([20261008, 'reader', row['uid']])[:8], 16),
                        skip_special_tokens=False, output_kind=RequestOutputKind.FINAL_ONLY)
                    require(self.units.intent(row, os.environ['SLURM_JOB_ID']), 'concurrent reader unit claim')
                    engine.add_request(row['uid'], {'prompt_token_ids': prompt}, params)
                    self.units.admitted(row, os.environ['SLURM_JOB_ID'])
                while engine.has_unfinished_requests():
                    require(time.time() < self.deadline, 'reader allocation deadline after admission')
                    for response in engine.step():
                        require(response.request_id in lookup and len(response.outputs) == 1, 'reader response attribution differs')
                        if response.finished:
                            output = response.outputs[0]
                            row = lookup[response.request_id]
                            outputs[row['uid']] = self.units.output(row, {'token_ids': list(output.token_ids),
                                'finish_reason': output.finish_reason, 'elapsed_seconds': time.monotonic()-tick})
                require(len(outputs) == len(batch), 'missing terminal reader output')
                parsing_started = time.monotonic()
                for uid, raw in outputs.items():
                    self.units.commit(lookup[uid], self.parse(lookup[uid], raw))
                parsing_seconds = time.monotonic()-parsing_started
                seconds = time.monotonic()-tick + preparation_seconds*len(batch)/len(pending)
                receipt = publish(self.root/'batches'/(digest(list(lookup))+'.json'), {
                    'status': 'complete', 'batch_size': len(batch), 'seconds': seconds,
                    'load_seconds': self.load_seconds, 'input_bin': str(input_bin),
                    'input_min': min(len(p) for _, p in batch), 'input_max': max(len(p) for _, p in batch),
                    'max_output_tokens': 16384, 'generated_tokens': sum(len(r['token_ids']) for r in outputs.values()),
                    'parsing_seconds':parsing_seconds, 'worker_memory':self.llm.collective_rpc('fr_metrics'),
                    'output_bytes':sum(p.stat().st_size for uid in lookup for p in self.units.folder(uid).iterdir() if p.is_file()),
                    'temporary_bytes':sum(p.stat().st_size for p in Path(os.environ['TMPDIR']).rglob('*') if p.is_file()),
                    'units': list(lookup)})
                result.append(receipt)
            except BaseException:
                if engine.has_unfinished_requests():
                    engine.abort_request(list(lookup), internal=True)
                raise
        return result

    def reuse(self, row):
        """Only complete rendered-context identities match; original unknowns stay unknown."""
        import sqlite3
        from moe_exp.research_campaign_v3_r6.contracts import Units as OldUnits
        model = row['model']
        parent = Path(self.binding['source_campaign'])
        old = OldUnits(parent/model/'annotations').state(row['uid']) if model != 'lfm25' else None
        source = None
        label = None
        if old is not None:
            label = old.get('label')
            source = {'kind': 'original-unit', 'model': model, 'uid': row['uid'],
                      'original_status': old['status']}
        else:
            index = parent/model/'LABEL_INDEX.sqlite'
            if index.exists():
                if model not in self.old_indices:
                    self.old_indices[model] = sqlite3.connect(f'file:{index}?mode=ro', uri=True)
                match = self.old_indices[model].execute('select label,source,row_index from labels where context=?',
                                                       (row['uid'],)).fetchone()
                if match is not None:
                    label = match[0]
                    source = {'kind': 'exact-original-context', 'path': match[1], 'row_index': match[2]}
        if source is not None:
            self.units.intent(row, 'exact-context-reuse')
            return self.units.commit(row, {'status': 'complete' if label in CLASSES else 'unknown',
                                          'label': label if label in CLASSES else None, 'source': source,
                                          'stochastic_redraw': False})
        return None

    def close(self):
        try:
            self.llm.llm_engine.engine_core.shutdown()
        finally:
            for db in self.old_indices.values():
                db.close()
            self.renderer.close()
