import importlib.util
from pathlib import Path

SOURCE=Path(__file__).resolve().parents[2]/'src/moe_exp/routing_control/closure.py'
spec=importlib.util.spec_from_file_location('m9_closure_test',SOURCE)
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)


def test_single_cut_integer_emission_has_no_future_input():
    prefix=[1]*C.CUT
    calls=[]
    def decode(ids):
        calls.append(ids);return 'old \\boxed{2} and latest \\boxed{7}'
    def candidates(text):
        return [{'start':4,'end':13,'value':'2'},{'start':25,'end':34,'value':'7'}]
    encode=lambda s:list(s.encode())
    first=C.prepare_integer_comparator(prefix+[9]*3000,native_natural_stop=False,
        decode=decode,encode=encode,candidates=candidates)
    second=C.prepare_integer_comparator(prefix+[8]*6000,native_natural_stop=False,
        decode=decode,encode=encode,candidates=candidates)
    assert first==second
    assert calls==[prefix,prefix]
    assert first['candidate']=='7'
    assert first['emission_token_ids']==list(b'Final answer: 7')
    assert first['tokens_charged']==C.CUT+len(b'Final answer: 7')


def test_finished_failures_and_no_integer_remain_in_intention_to_treat():
    def forbidden(_):raise AssertionError('finished requests need no parser or injection')
    finished=C.prepare_integer_comparator([4,5],native_natural_stop=True,
        decode=forbidden,encode=forbidden,candidates=forbidden)
    assert finished['status']=='retain_native_finished'
    incomplete=C.prepare_integer_comparator([4,5],native_natural_stop=False,
        decode=forbidden,encode=forbidden,candidates=forbidden)
    assert incomplete['status']=='failed_source_prefix' and incomplete['operational_correct'] is False
    no_integer=C.prepare_integer_comparator([1]*(C.TOTAL_BUDGET+10),native_natural_stop=False,
        decode=lambda _: 'unsupported symbolic answer',encode=forbidden,candidates=lambda _: [])
    assert no_integer['status']=='fallback_native_no_integer'
    assert no_integer['tokens_charged']==C.TOTAL_BUDGET


def test_closure_injection_is_charged_and_has_single_answer_allowance():
    result=C.prepare_closure([3,4],[1]*(C.CUT+9000),native_natural_stop=False,encode=lambda _: [99,100,101])
    assert result['prefix_presence_start']==2
    assert result['prompt_token_ids']==[3,4]+[1]*C.CUT+[99,100,101]
    assert result['max_new_tokens']+result['prefix_len']==C.TOTAL_BUDGET
    assert result['closure_text']=='</think>\n\n'
