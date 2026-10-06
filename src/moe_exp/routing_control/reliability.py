"""Paired scalar/probe-score reliability, with fixed enrollment and two launches."""
import numpy as np


def absolute_agreement_icc(values, *, expected_n=100):
    """Two-way single-measure absolute-agreement ICC(A,1); no deletion of missing pairs."""
    x=np.asarray(values,float)
    if x.ndim!=2 or x.shape[1]!=2:raise ValueError('expected questions by two launches')
    complete=np.isfinite(x).all(axis=1)
    if len(x)!=expected_n or not complete.all():
        return {'status':'INCOMPLETE','ICC':None,'assigned':expected_n,'provided':len(x),
            'complete_pairs':int(complete.sum())}
    if expected_n<2:raise ValueError('at least two question pairs required')
    n,k=x.shape;grand=x.mean();rows=x.mean(axis=1);columns=x.mean(axis=0)
    ms_rows=k*np.var(rows,ddof=1);ms_columns=n*np.var(columns,ddof=1)
    ms_error=np.square(x-rows[:,None]-columns[None,:]+grand).sum()/((n-1)*(k-1))
    denominator=ms_rows+(k-1)*ms_error+k*(ms_columns-ms_error)/n
    if denominator<=0 or not np.isfinite(denominator):
        return {'status':'UNIDENTIFIED_CONSTANT','ICC':None,'assigned':n,'complete_pairs':n}
    return {'status':'COMPLETE','ICC':float((ms_rows-ms_error)/denominator),'assigned':n,'complete_pairs':n,
        'method':'two-way single-measure absolute-agreement ICC(A,1)',
        'MS_questions':float(ms_rows),'MS_launches':float(ms_columns),'MS_error':float(ms_error)}
