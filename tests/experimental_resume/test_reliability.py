import importlib.util
from pathlib import Path
import numpy as np

source=Path(__file__).resolve().parents[2]/'src/moe_exp/routing_control/reliability.py'
spec=importlib.util.spec_from_file_location('repeat_icc_test',source)
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)


def test_absolute_agreement_penalizes_launch_shift():
    x=np.arange(100,dtype=float)
    assert R.absolute_agreement_icc(np.column_stack((x,x)))['ICC']==1.
    shifted=R.absolute_agreement_icc(np.column_stack((x,x+100)))
    assert shifted['status']=='COMPLETE' and shifted['ICC']<.7


def test_missing_pairs_and_constant_scores_cannot_qualify():
    x=np.ones((100,2));assert R.absolute_agreement_icc(x)['ICC'] is None
    x[3,1]=np.nan
    assert R.absolute_agreement_icc(x)['status']=='INCOMPLETE'
    assert R.absolute_agreement_icc(x[:99])['ICC'] is None
