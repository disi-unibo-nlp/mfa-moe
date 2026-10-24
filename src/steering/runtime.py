"""Native vLLM loader and per-request durable generation."""
from __future__ import annotations
from pathlib import Path
import time
from store import check_files, digest, load, publish, require


def remaining(profile, prompt, prefix):
    card = profile['card']
    cap = card['max_model_len']-len(prompt)-len(prefix)
    if card['max_tokens'] is not None:
        cap = min(cap, card['max_tokens']-len(prefix))
    require(cap > 0, 'native remaining context/output allowance exhausted')
    return cap


def closure(tokenizer, model):
    if model != 'lfm25':
        from models.engine import closure_sequences
        return closure_sequences(tokenizer, model)
    ids = tokenizer.encode('</think>', add_special_tokens=False)
    require(ids and tokenizer.decode(ids).strip() == '</think>', 'LFM reasoning closure is not qualified')
    return [ids]


def prompt_prefix(binding, model, row, tokenizer):
    if row.get('native_example'):
        example = row['native_example']
        messages = example.get('generation_messages') or (
            ([{'role': 'system', 'content': example['system_prompt']}] if example.get('system_prompt') else [])+
            [{'role': 'user', 'content': example['prompt']}])
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=False, **binding['models'][model]['template_kwargs'])
        require(isinstance(tokens, list) and all(type(t) is int for t in tokens), 'native prompt tokenization differs')
        return tokens, [], {**example, 'metadata': example.get('metadata', {})}
    from store import read_source
    root = Path(binding['source_campaign']) if model != 'lfm25' else Path(binding['root'])
    table = load(root/model/'SOURCES.json')['records']
    require(row['source'] is not None, 'unavailable frozen prefix')
    source = next(r for r in table if r['attempt_id'] == row['source']['attempt_id'])
    trace = read_source(source['source_location'])
    replay = trace['metadata']['token_replay']
    prompt, completion = replay['prompt_token_ids'], replay['completion_token_ids']
    require(not row.get('prompt_sha256') or digest(prompt) == row['prompt_sha256'], 'original prompt token IDs changed')
    n = row['source']['prefix_tokens']
    require(0 <= n <= len(completion), 'frozen prefix token boundary differs')
    require(n > 0 or row.get('original_prompt'), 'empty prefix requires an original-prompt assignment')
    if row.get('budget_prefix'):
        return list(prompt), budget_tokens(row['budget_prefix']), trace
    return list(prompt), list(completion[:n]), trace


def budget_tokens(spec):
    """Prefix of a length-limit row: the first `cut` generated tokens of a finished native run, then the
    model's closing suffix, so the model only writes its answer (the length-limit arm of the outcome releases)."""
    from store import file_sha
    require(file_sha(spec['path']) == spec['sha256_file'], 'budget source output changed')
    tokens = load(spec['path'])['token_ids']
    require(0 < spec['cut'] <= len(tokens), 'budget cut outside the source output')
    return list(tokens[:spec['cut']])+list(spec['suffix'])


class Backend:
    def __init__(self, binding, model, capacity, deadline, *, replay=False):
        import os
        require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS'), 'GPU inference requires Slurm GPUs')
        from vllm import LLM
        from models.engine import engine_kwargs
        self.binding, self.model, self.profile, self.deadline = binding, model, binding['models'][model], deadline
        check_files(self.profile['config_files'])
        if model == 'lfm25':
            from store import verified
            proof = verified(Path(binding['root'])/'LFM_CHECKPOINT_VERIFIED.json')
            require(proof['status'] == 'complete', 'LFM checksum validation is incomplete')
        require((Path(self.profile['local_path'])/'config.json').is_file(), 'pinned checkpoint is unavailable')
        kwargs = engine_kwargs(self.profile, context=self.profile['card']['max_model_len'])
        kwargs.update(max_num_seqs=capacity, enable_chunked_prefill=True,
                      worker_extension_cls='steering.observer.RoutingWorkerExtension',
                      logits_processors=['steering.penalties:PrefixPenalties'])
        if self.model == 'lfm25':
            kwargs['kv_cache_memory_bytes'] = self.profile['kv_cache_memory_bytes']
        self.kwargs = kwargs
        started = time.monotonic()
        self.llm = LLM(**kwargs)
        self.load_seconds = time.monotonic()-started
        self.tokenizer = self.llm.get_tokenizer()
        require(type(self.llm.llm_engine.engine_core).__name__ == 'InprocClient', 'unqualified executor')
        self.closure = closure(self.tokenizer, model)
        self.capacity = capacity
        self.replay = replay

    def capture_trace(self, trace, destination):
        import numpy as np
        from vllm import SamplingParams
        from store import file_sha, verified
        destination = Path(destination)
        receipt = destination.with_suffix('.json')
        if receipt.exists():
            value = verified(receipt)
            require(file_sha(destination) == value['file_sha256'], 'replay bytes changed')
            return value
        require(self.replay and self.capacity == 1, 'unqualified replay profile')
        replay = trace['metadata']['token_replay']
        prompt, tokens = replay['prompt_token_ids'], replay['completion_token_ids']
        require(len(prompt)+len(tokens) <= self.profile['card']['max_model_len'], 'replay exceeds native input length')
        original_tokens=list(tokens)
        terminal_reasoning=0
        if len(prompt)+len(tokens)==self.profile['card']['max_model_len']:
            from schemas import TraceRecord
            from capture.dynamics.classes import saved_layout
            reasoning=saved_layout(TraceRecord(**trace))['reasoning_tokens']
            # The final token of a trace that filled the context has no room for teacher forcing. Its
            # identity is preserved and its routing is recorded as unavailable (a capped trace ends inside
            # reasoning, so this is one reasoning token of ~131k); never extend the model context limit.
            terminal_reasoning=int(len(tokens)-1 in reasoning)
            tokens=tokens[:-1]
        uid = digest(['deterministic-native-replay', prompt, tokens, self.profile['profile_sha256']])
        request = {'uid': uid, 'prompt': prompt, 'prefix': tokens, 'closure': self.closure,
                   'capture_layers': self.profile['layers'], 'window': len(tokens), 'replay': True, 'audit': True}
        engine = self.llm.llm_engine
        require(not engine.has_unfinished_requests(), 'replay has neighboring work')
        from steering.session import request_session
        tick = time.monotonic()
        with request_session(self.llm, [request], self.profile) as session:
            session.admit(uid, {'prompt_token_ids': prompt + tokens},
                          SamplingParams(temperature=0., max_tokens=1, seed=0))
            session.bind()
            while engine.has_unfinished_requests():
                require(time.time() < self.deadline, 'replay deadline')
                session.step()
            report = session.finish()
            capture = report['capture'][uid]
            require(all(capture[str(l)]['positions'] == list(range(len(tokens))) for l in self.profile['layers']), 'native replay token coverage differs')
            ids = np.array([capture[str(l)]['native_ids'] for l in self.profile['layers']], dtype=np.uint8)
            weights = np.array([capture[str(l)]['native_weights'] for l in self.profile['layers']], dtype=np.float32)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix('.pending')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, ids=ids, weights=weights, token_ids=np.asarray(tokens, np.int32), layers=self.profile['layers'])
            temporary.replace(destination)
            return publish(receipt, {'status': 'complete', 'path': str(destination), 'file_sha256': file_sha(destination),
                'trace_sha256': digest(trace), 'input_tokens': len(prompt)+len(tokens), 'completion_tokens': len(original_tokens),
                'captured_completion_tokens':len(tokens), 'unavailable_terminal_nonreasoning_tokens':len(original_tokens)-len(tokens)-terminal_reasoning,
                'unavailable_terminal_reasoning_tokens':terminal_reasoning,
                'seconds': time.monotonic()-tick, 'bytes': destination.stat().st_size,
                'profile': 'teacher-forced-native-input-token', 'discarded_output_tokens': 1,
                'native_validation': report['native_validation'], 'load_seconds': self.load_seconds})

    def close(self):
        self.llm.llm_engine.engine_core.shutdown()
