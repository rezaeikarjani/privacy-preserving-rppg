import numpy as np
import pytest
from scripts.audit_reference import peaks_summary


@pytest.mark.parametrize('polarity',[1,-1])
def test_pulse_interval_rate_on_known_periodic_signal(polarity):
    fs=30;t=np.arange(900)/fs;x=np.sin(2*np.pi*1.5*t)
    r,_,_=peaks_summary(x,t,fs,polarity,.5,1/3)
    assert abs(r['median_ibi_hr_bpm']-90)<1
    assert abs(r['span_count_hr_bpm']-90)<1


def test_flat_signal_does_not_generate_a_reference_hr():
    fs=30;t=np.arange(900)/fs
    r,_,_=peaks_summary(np.zeros(900),t,fs,1,.5,1/3)
    assert r['pulse_count']==0 and np.isnan(r['median_ibi_hr_bpm'])


def test_median_interval_and_span_rate_are_different_summaries():
    fs=100;t=np.arange(3000)/fs;events=np.r_[np.arange(.5,15,.5),np.arange(15.5,30,1)]
    x=np.exp(-((t[:,None]-events[None,:])/.04)**2).sum(axis=1)
    r,_,_=peaks_summary(x,t,fs,1,.5,1/3)
    assert abs(r['median_ibi_hr_bpm']-120)<1
    assert 85<r['span_count_hr_bpm']<95
