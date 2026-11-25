import numpy as np
import pytest
from src.linkability import gallery_comparison


def test_unique_genuine_match_and_impostor_match_distinguished():
    g=np.eye(3)
    assert gallery_comparison(g[1],g,1)['rank1_unique']
    r=gallery_comparison(g[0],g,1)
    assert not r['rank1_unique'] and r['margin']<0


def test_tied_scores_are_not_counted_as_success():
    r=gallery_comparison(np.array([1.,0]),np.array([[1.,0],[1.,0]]),0)
    assert r['tied_top'] and not r['rank1_unique']


def test_low_absolute_similarity_does_not_prevent_forced_rank_match():
    # An arbitrary low genuine score may still beat the alternatives; not verification.
    r=gallery_comparison([1.,0],np.array([[.1,np.sqrt(.99)],[0.,1.]]),0)
    assert r['rank1_unique'] and r['genuine_cosine']==.1


def test_invalid_gallery_rejected():
    with pytest.raises(ValueError):gallery_comparison([1.,0],[[1.,0]],0)
    with pytest.raises(ValueError):gallery_comparison([1.,0],np.eye(2),3)
