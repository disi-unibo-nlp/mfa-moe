import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from moe_exp.study.common import digest
from moe_exp.study.session import request_session


class FakeEngine:
    def __init__(self):
        self.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=2))
        self.live = {}
        self.attempts = 0
        self.partial_failure = False
        self.abort_failure = False
        self.closed = False
        self.output_id = None
        self.abort_calls = []
        self.engine_core = SimpleNamespace(shutdown=self.shutdown)

    def shutdown(self):
        self.closed = True

    def has_unfinished_requests(self):
        return bool(self.live)

    def add_request(self, uid, prompt, params):
        self.attempts += 1
        internal = uid + '-%08d' % self.attempts
        self.live[internal] = uid
        if self.partial_failure:
            raise RuntimeError('admission failed after registration')
        return internal

    def abort_request(self, ids, internal=False):
        self.abort_calls.append((list(ids), internal))
        if self.abort_failure:
            raise RuntimeError('abort failed')
        self.live = {key: uid for key, uid in self.live.items()
                     if (key if internal else uid) not in ids}

    def step(self):
        outputs = [SimpleNamespace(request_id=self.output_id or uid, finished=True)
                   for uid in reversed(list(self.live.values()))]
        self.live.clear()
        return outputs


class FakeLLM:
    def __init__(self):
        self.llm_engine = FakeEngine()
        self.attached = False
        self.aliases = None
        self.calls = []
        self.attach_failure = False

    def collective_rpc(self, method, kwargs=None):
        self.calls.append(method)
        if method == 'fr_begin':
            self.attached = True
            if self.attach_failure:
                raise RuntimeError('second rank failed to attach')
            return [{'status': 'attached'}] * 2
        if method == 'fr_bind':
            self.aliases = kwargs['aliases']
            return [{'mapping_sha256': digest(self.aliases)}] * 2
        if method == 'fr_end':
            self.attached = False
            return [{'capture_sha256': 'same', 'request_map_sha256': digest(self.aliases),
                     'capture': {uid: {} for uid in self.aliases.values()}}] * 2
        if method == 'fr_abort':
            self.attached = False
            return [{'status': 'detached'}] * 2
        raise AssertionError(method)


class SessionTests(unittest.TestCase):
    def test_installed_vllm_generates_distinct_internal_ids(self):
        source = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/lib/python3.11/site-packages/vllm/v1/engine/input_processor.py')
        node = next(n for n in ast.walk(ast.parse(source.read_text()))
                    if isinstance(n, ast.FunctionDef) and n.name == 'assign_request_id')
        node.decorator_list = []
        values = iter(('11111111ffffffff', '22222222ffffffff'))
        namespace = {'EngineCoreRequest': SimpleNamespace,
                     'envs': SimpleNamespace(VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=False),
                     'random_uuid': lambda: next(values)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), 'exec'), namespace)
        a = SimpleNamespace(request_id='same', external_req_id=None)
        b = SimpleNamespace(request_id='same', external_req_id=None)
        namespace['assign_request_id'](a); namespace['assign_request_id'](b)
        self.assertEqual(a.external_req_id, 'same')
        self.assertNotEqual(a.request_id, b.request_id)
        self.assertNotEqual(a.request_id, a.external_req_id)

    def test_reordered_outputs_keep_stable_assignment_ids(self):
        llm = FakeLLM(); requests = [{'uid': 'a'}, {'uid': 'b'}]
        with request_session(llm, requests, {'model': 'fixture'}) as session:
            a = session.admit('a', {}, None); b = session.admit('b', {}, None)
            with self.assertRaisesRegex(ValueError, 'before binding'):
                session.step()
            session.bind()
            self.assertEqual(llm.aliases, {a: 'a', b: 'b'})
            self.assertEqual([r.request_id for r in session.step()], ['b', 'a'])
            self.assertEqual(set(session.finish()['capture']), {'a', 'b'})
        self.assertFalse(llm.attached)
        self.assertFalse(llm.llm_engine.has_unfinished_requests())
        self.assertEqual(llm.llm_engine.abort_calls[-1], (['a', 'b'], False))

    def test_partial_admission_uses_attempted_external_uid_for_cleanup(self):
        llm = FakeLLM(); llm.llm_engine.partial_failure = True
        with self.assertRaisesRegex(RuntimeError, 'after registration'):
            with request_session(llm, [{'uid': 'a'}], {}) as session:
                session.admit('a', {}, None)
        self.assertFalse(llm.attached); self.assertFalse(llm.llm_engine.live)
        self.assertEqual(llm.llm_engine.abort_calls, [(['a'], False)])

    def test_abort_failure_still_detaches_and_closes_engine(self):
        llm = FakeLLM(); llm.llm_engine.abort_failure = True
        with self.assertRaisesRegex(RuntimeError, 'abort failed'):
            with request_session(llm, [{'uid': 'a'}], {}) as session:
                session.admit('a', {}, None)
        self.assertFalse(llm.attached); self.assertTrue(llm.llm_engine.closed)

    def test_partial_hook_attachment_is_restored(self):
        llm = FakeLLM(); llm.attach_failure = True
        with self.assertRaisesRegex(RuntimeError, 'second rank'):
            with request_session(llm, [{'uid': 'a'}], {}):
                self.fail('failed attachment yielded a session')
        self.assertFalse(llm.attached)

    def test_unknown_output_fails_and_cleanup_runs(self):
        llm = FakeLLM(); llm.llm_engine.output_id = 'neighbor'
        with self.assertRaisesRegex(ValueError, 'unregistered external'):
            with request_session(llm, [{'uid': 'a'}], {}) as session:
                session.admit('a', {}, None); session.bind(); session.step()
        self.assertFalse(llm.attached)

    def test_missing_admission_and_duplicate_uid_are_rejected(self):
        llm = FakeLLM()
        with self.assertRaisesRegex(ValueError, 'duplicate external'):
            with request_session(llm, [{'uid': 'a'}, {'uid': 'a'}]):pass
        with request_session(llm, [{'uid': 'a'}, {'uid': 'b'}]) as session:
            session.admit('a', {}, None)
            with self.assertRaisesRegex(ValueError, 'already attempted'):session.admit('a', {}, None)
            with self.assertRaisesRegex(ValueError, 'incomplete'):session.bind()
        self.assertFalse(llm.llm_engine.live)

    def test_unrelated_engine_work_is_never_aborted(self):
        llm = FakeLLM(); llm.llm_engine.live = {'old-internal': 'neighbor'}
        with self.assertRaisesRegex(ValueError, 'neighboring'):
            with request_session(llm, [{'uid': 'a'}]):pass
        self.assertEqual(llm.llm_engine.live, {'old-internal': 'neighbor'})
        self.assertFalse(llm.llm_engine.abort_calls)

    def test_tensor_parallel_map_disagreement_stops_before_execution(self):
        llm = FakeLLM(); original = llm.collective_rpc
        def rpc(method, kwargs=None):
            reports = original(method, kwargs)
            return [reports[0], {'mapping_sha256': 'wrong'}] if method == 'fr_bind' else reports
        llm.collective_rpc = rpc
        with self.assertRaisesRegex(ValueError, 'maps disagree'):
            with request_session(llm, [{'uid': 'a'}], {}) as session:
                session.admit('a', {}, None); session.bind()
        self.assertFalse(llm.attached); self.assertFalse(llm.llm_engine.live)

    def test_observer_binding_rejects_alias_collisions_and_unregistered_assignments(self):
        from moe_exp.study.observer import RequestCapture
        capture = RequestCapture.__new__(RequestCapture)
        capture.aliases = None; capture.requests = {'a': {}, 'b': {}}
        with self.assertRaisesRegex(ValueError, 'one-to-one'):capture.bind({'x': 'a', 'y': 'a'})
        with self.assertRaisesRegex(ValueError, 'does not match'):capture.bind({'x': 'a', 'y': 'neighbor'})
        result = capture.bind({'y': 'b', 'x': 'a'})
        self.assertEqual(result['mapping_sha256'], digest({'x': 'a', 'y': 'b'}))
