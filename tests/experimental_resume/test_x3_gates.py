import hashlib
import json
import pytest
from x3_build import digest,pinned_order,verified


def test_enrollment_is_hash_ordered_and_keeps_all_96():
    rows=[{'dataset':'d','source_problem_id':str(i),'subsplit':'dev-disc'} for i in range(96)]
    rows += [{'dataset':'d','source_problem_id':'confirm','subsplit':'confirm'}]
    order=pinned_order({'questions':rows[::-1]})
    expected=sorted(['d|'+str(i) for i in range(96)],key=lambda q:hashlib.sha256(('forum-v1|'+q).encode()).hexdigest())
    assert order==expected and len(set(order[:64]))==64
    with pytest.raises(ValueError,match='96'):pinned_order({'questions':rows[:95]})


def test_changed_selection_or_price_cannot_pass_seal(workdir):
    path=workdir/'gate.json';body={'schema':'price','all_in_GPU_h':1.}
    path.write_text(json.dumps({**body,'sha256':digest(body)}))
    assert verified(path)['all_in_GPU_h']==1.
    path.write_text(json.dumps({**body,'all_in_GPU_h':20.,'sha256':digest(body)}))
    with pytest.raises(ValueError,match='changed'):verified(path)
