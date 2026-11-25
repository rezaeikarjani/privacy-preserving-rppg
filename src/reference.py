"""Strict timestamped contact-reference helpers; no extrapolation or lag fitting."""
import numpy as np


def load_ubfc_reference(path):
    a=np.loadtxt(path)
    if a.ndim!=2 or a.shape[0]!=3 or a.shape[1]<2 or not np.isfinite(a).all():
        raise ValueError('UBFC Dataset2 requires three finite rows: PPG, device HR, time')
    t=a[2]-a[2,0]
    if np.any(np.diff(t)<0):raise ValueError('Reference timestamps go backwards')
    duplicate=np.flatnonzero(np.diff(t)==0)
    for i in duplicate:
        if not np.array_equal(a[:2,i],a[:2,i+1]):
            raise ValueError('Conflicting measurements at duplicate timestamp')
    # Remove only exact duplicate timestamp/value rows; no averaging or time shifts.
    keep=np.r_[True,np.diff(t)>0]
    return t[keep],a[0,keep],a[1,keep]


def sample_reference(t,values,target_times,max_gap_s=.5):
    t=np.asarray(t,dtype=float);values=np.asarray(values,dtype=float);q=np.asarray(target_times,dtype=float)
    if len(t)!=len(values) or len(t)<2 or np.any(np.diff(t)<=0):raise ValueError('Invalid reference arrays')
    if not np.isfinite(t).all() or not np.isfinite(values).all() or not np.isfinite(q).all():raise ValueError('Nonfinite reference')
    if len(q)==0 or q.min()<t[0] or q.max()>t[-1]:raise ValueError('Reference coverage insufficient; no extrapolation')
    lo=max(0,np.searchsorted(t,q.min(),side='right')-1);hi=min(len(t),np.searchsorted(t,q.max(),side='left')+1)
    if np.any(np.diff(t[lo:hi])>max_gap_s):raise ValueError('Reference gap too large')
    return np.interp(q,t,values)


def time_weighted_mean(t,values,start,end,max_gap_s=.5):
    if end<=start:raise ValueError('Invalid reference window')
    inside=np.asarray(t)[(np.asarray(t)>start)&(np.asarray(t)<end)]
    knots=np.r_[start,inside,end]
    values=sample_reference(t,values,knots,max_gap_s)
    return float(np.trapezoid(values,knots)/(end-start))
