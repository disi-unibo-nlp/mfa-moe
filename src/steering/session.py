"""One request lifecycle for generation, replay and reader batches."""
from contextlib import contextmanager
from store import digest, require


class Session:
    def __init__(self, llm, requests, contract=None):
        self.llm = llm
        self.engine = llm.llm_engine
        self.requests = {row['uid']: row for row in requests}
        require(len(self.requests) == len(requests), 'duplicate external request ID')
        self.contract = contract
        self.workers = llm.llm_engine.vllm_config.parallel_config.tensor_parallel_size
        require(self.workers == 2, 'study requires the qualified two-rank execution profile')
        self.attempted = []
        self.internal_to_external = {}
        self.bound = False

    def admit(self, uid, prompt, params):
        require(not self.bound and uid in self.requests, 'invalid or late request admission')
        require(uid not in self.attempted, 'external request was already attempted')
        # Track before calling vLLM: admission may raise after partially succeeding.
        self.attempted.append(uid)
        internal_id = self.engine.add_request(uid, prompt, params)
        require(isinstance(internal_id, str) and internal_id, 'vLLM returned no internal ID')
        require(internal_id not in self.internal_to_external, 'duplicate internal request ID')
        self.internal_to_external[internal_id] = uid
        return internal_id

    def bind(self):
        require(not self.bound, 'request map already bound')
        require(set(self.internal_to_external.values()) == set(self.requests), 'batch admission incomplete')
        if self.contract is not None:
            reports = self.llm.collective_rpc('fr_bind', kwargs={'aliases': self.internal_to_external})
            expected = digest(self.internal_to_external)
            require(len(reports) == self.workers and all(r['mapping_sha256'] == expected for r in reports),
                    'tensor-parallel request maps disagree')
        self.bound = True

    def step(self):
        require(self.bound, 'cannot execute before binding request identities')
        responses = self.engine.step()
        for response in responses:
            require(response.request_id in self.requests, 'output has an unregistered external ID')
        return responses

    def finish(self):
        require(not self.engine.has_unfinished_requests(), 'batch still has unfinished requests')
        if self.contract is None:
            return None
        reports = self.llm.collective_rpc('fr_end')
        require(len(reports) == self.workers and reports[0]['capture_sha256'] == reports[1]['capture_sha256'],
                'tensor-parallel captures disagree')
        require(all(report['request_map_sha256'] == digest(self.internal_to_external) for report in reports),
                'finished capture has another request map')
        return reports[0]

    def cleanup(self):
        try:
            if self.attempted:
                self.engine.abort_request(self.attempted, internal=False)
            require(not self.engine.has_unfinished_requests(), 'abort left unfinished engine requests')
        finally:
            if self.contract is not None:
                reports = self.llm.collective_rpc('fr_abort')
                require(len(reports) == self.workers and all(r['status'] == 'detached' for r in reports),
                        'routing hooks were not detached on every rank')


@contextmanager
def request_session(llm, requests, contract=None):
    session = Session(llm, requests, contract)
    require(not session.engine.has_unfinished_requests(), 'engine has neighboring requests')
    try:
        if contract is not None:
            llm.collective_rpc('fr_begin', kwargs={'contract': contract, 'requests': requests})
        yield session
    finally:
        try:
            session.cleanup()
        except BaseException:
            # A failed cleanup makes this engine unusable, even if hooks detached.
            llm.llm_engine.engine_core.shutdown()
            raise
