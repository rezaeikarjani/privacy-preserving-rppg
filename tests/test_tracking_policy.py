import numpy as np
import pytest
from src.tracking_policies import filter_target_region


def test_larger_shirt_false_positive_excluded_by_explicit_region():
    faces=[[286,185,147,147],[270,302,158,158]]
    assert np.array_equal(filter_target_region(faces,[250,140,220,210]),[faces[0]])


def test_no_detection_is_not_replaced_with_manual_face_box():
    assert filter_target_region([], [250,140,220,210]).shape==(0,4)
    assert len(filter_target_region([[10,10,60,60]], [250,140,220,210]))==0


def test_multiple_candidates_inside_target_remain_ambiguous():
    assert len(filter_target_region([[280,180,147,147],[290,190,140,140]],[250,140,220,210]))==2
    with pytest.raises(ValueError):filter_target_region([], [1,2,0,4])
