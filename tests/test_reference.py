import numpy as np
import pytest
from src.reference import sample_reference,time_weighted_mean,load_ubfc_reference


def test_time_weighting_for_irregular_samples():
    t=np.array([0,.1,.4,.5]);v=60+10*t
    assert np.isclose(time_weighted_mean(t,v,0,.5),62.5)
    assert np.isclose(time_weighted_mean(t,v,.05,.45),62.5)


def test_no_extrapolation_or_interpolation_across_large_gaps():
    with pytest.raises(ValueError,match='coverage'):sample_reference([0,1],[2,3],[-.1])
    with pytest.raises(ValueError,match='gap'):sample_reference([0,1],[2,3],[.5])


def test_nonmonotonic_timestamps_rejected():
    with pytest.raises(ValueError):sample_reference([0,0],[1,2],[0])


def test_ubfc_row_order_and_timestamp_origin(tmp_path):
    p=tmp_path/'reference.txt';np.savetxt(p,[[3,4,5],[60,61,62],[100,100.1,100.2]])
    t,ppg,h=load_ubfc_reference(p)
    assert np.allclose(t,[0,.1,.2]) and np.array_equal(ppg,[3,4,5]) and np.array_equal(h,[60,61,62])


def test_exact_duplicate_reference_can_be_removed_but_conflict_rejected(tmp_path):
    p=tmp_path/'reference.txt';np.savetxt(p,[[3,4,4,5],[60,61,61,62],[0,.1,.1,.2]])
    t,ppg,h=load_ubfc_reference(p)
    assert np.array_equal(t,[0,.1,.2]) and np.array_equal(ppg,[3,4,5])
    np.savetxt(p,[[3,4,9,5],[60,61,61,62],[0,.1,.1,.2]])
    with pytest.raises(ValueError,match='Conflicting'):load_ubfc_reference(p)
