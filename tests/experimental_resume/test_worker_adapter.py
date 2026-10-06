import numpy as np
import pytest
from types import SimpleNamespace as NS
from moe_exp.routing_control.design import digest
from moe_exp.routing_control.worker_adapter import OrderedPulse


def table():
    policies=[NS(name=n,operator=NS(kind='bias',sign=1,magnitude=b),
                 schedule=NS(kind='always'),targets=NS(experts=((3,(12,)),(4,(20,21)))))
              for n,b in [('a',.5),('b',1.)]]
    return NS(policies=policies,index_of=lambda n:['a','b'].index(n))


def pulse(names=('a','b')):
    value={'action_policy_names':list(names),'slots':[0,512][:len(names)],'horizon':1024}
    return OrderedPulse.load({**value,'sha256':digest(value)},table(),names[0])


def test_order_boundaries_native_neighbors_and_recompute():
    p=pulse();pos=np.array([-32,-1,0,255,256,511,512,767,768,1024])
    active=np.ones(len(pos),bool);mask,indices=p.rows(pos,active)
    assert mask.tolist()==[False,False,True,True,False,False,True,True,False,False]
    assert indices.tolist()==[0,0,0,0,0,0,1,1,0,0]
    perm=np.array([6,2,9,0,7,3,5,1,8,4])
    remask,reindex=p.rows(pos[perm],active[perm])
    assert np.array_equal(remask,mask[perm]) and np.array_equal(reindex,indices[perm])
    # The base closure FSM supplies the observed native-active mask.
    active[pos>=600]=False
    closed,_=p.rows(pos,active)
    assert closed.tolist()==[False,False,True,True,False,False,True,False,False,False]
    assert not p.rows(pos,np.zeros(len(pos),bool))[0].any()


def test_reverse_multiset_order_and_metadata_validation():
    mask,forward=pulse().rows(np.array([0,512]),[True,True])
    _,backward=pulse(('b','a')).rows(np.array([0,512]),[True,True])
    assert forward.tolist()==[0,1] and backward.tolist()==[1,0] and mask.all()
    v={'action_policy_names':['a'],'slots':[0],'horizon':1024,'future_answer':'secret'}
    with pytest.raises(ValueError,match='unknown'):
        OrderedPulse.load({**v,'sha256':digest(v)},table(),'a')
    v.pop('future_answer');v['slots']=[512]
    with pytest.raises(ValueError,match='ordered'):
        OrderedPulse.load({**v,'sha256':digest(v)},table(),'a')
