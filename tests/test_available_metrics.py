import importlib.util
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_available_metrics as a


def test_question_balance_and_within_question_contrast():
    rows=[dict(question_id='a',features={'x':1},is_correct=True,completion_tokens=100,capped=False),
          dict(question_id='a',features={'x':3},is_correct=False,completion_tokens=100,capped=True),
          dict(question_id='b',features={'x':4},is_correct=True,completion_tokens=200,capped=False)]
    x,y,c,w,_=a.grouped(rows,'x')
    assert x.tolist()==[2,4] and y.tolist()==[.5,1] and w==[-2]
    assert c[0,1]==.5


def test_missing_outcomes_excluded_caps_retained():
    rows=[dict(question_id='a',features={'x':1},is_correct=False,completion_tokens=100,capped=True),
          dict(question_id='b',features={'x':4},is_correct=None,completion_tokens=200,capped=False)]
    x,y,c,_,_=a.grouped(rows,'x')
    assert len(x)==1 and y.tolist()==[0] and c[0,1]==1


def test_no_transition_across_missing_labels():
    shared=dict(trace_id='t',model='m',dataset='d',question_id='q',segment=0,token_start=None,tokens=None,termination='generation_cap')
    rows=[dict(shared,sentence_index=i,label=c,previous_class=None) for i,c in enumerate(['Read',None,'Verify','Monitor'])]
    d=a.class_dynamics(rows)
    assert [(r['source'],r['destination']) for r in d['transitions']]==[('Verify','Monitor')]
    assert d['runs'][-1]['right_censored']


def test_constant_predictor_is_unavailable():
    assert a.stats(np.ones(10),np.arange(10))['status']=='insufficient_support'


def test_all_unknown_class_features_are_missing(monkeypatch):
    from types import SimpleNamespace
    import copy
    shared=dict(trace_id='t',model='m',dataset='d',question_id='q',segment=0,
                token_start=None,tokens=None,termination='generation_cap')
    sentences=[dict(shared,sentence_index=i,label=None,previous_class=None) for i in range(3)]
    monkeypatch.setattr(a,'saved_layout',lambda t:{})
    monkeypatch.setattr(a,'sentence_table',lambda *args:copy.deepcopy(sentences))
    monkeypatch.setattr(a,'base_row',lambda t:dict(features={}))
    t=SimpleNamespace(metadata={},step_labels=SimpleNamespace(
        backtracking_steps=[],contradiction_steps=[],self_correction_steps=[]))
    row,d=a.label_features(t,{})
    assert row['coverage']==0 and row['class_fraction_denominator']==0
    assert row['unknown_sentences']==3 and row['labeled_sentences']==0
    assert all(row['features']['class_fraction/'+c] is None for c in a.CLASSES)
    assert not d['transitions'] and not d['runs']


def test_partial_class_coverage_preserves_zero_and_missing_gap(monkeypatch):
    from types import SimpleNamespace
    import copy
    shared=dict(trace_id='t',model='m',dataset='d',question_id='q',segment=0,
                token_start=None,tokens=None,termination='generation_cap')
    sentences=[dict(shared,sentence_index=i,label=label,previous_class=None)
               for i,label in enumerate(['Read',None,'Monitor'])]
    monkeypatch.setattr(a,'saved_layout',lambda t:{})
    monkeypatch.setattr(a,'sentence_table',lambda *args:copy.deepcopy(sentences))
    monkeypatch.setattr(a,'base_row',lambda t:dict(features={}))
    t=SimpleNamespace(metadata={},step_labels=SimpleNamespace(
        backtracking_steps=[],contradiction_steps=[],self_correction_steps=[]))
    row,d=a.label_features(t,{})
    assert row['coverage']==2/3 and row['class_fraction_denominator']==2
    assert row['features']['class_fraction/Read']==.5
    assert row['features']['class_fraction/Explore']==0
    assert not d['transitions']


def test_base_row_does_not_reclassify_unknown_answer(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(a,'keys',lambda t:dict(question_id='q'))
    monkeypatch.setattr(a,'termination',lambda t:'generation_cap')
    t=SimpleNamespace(is_correct=None,gold_answer='1',model_answer=None,
        metadata={},prompt='question',generation_messages=[],dataset='d')
    assert a.base_row(t)['is_correct'] is None
