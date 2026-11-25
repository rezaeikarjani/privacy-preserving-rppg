import numpy as np
import pytest
from src.quality import spectral_quality
from src.signals import hr


def tones(*pairs):
    t=np.arange(900)/30
    return sum(amplitude*np.sin(2*np.pi*frequency*t) for frequency,amplitude in pairs)


def test_equal_competing_tones_flag_without_correcting_hr():
    x=tones((1.,1.),(1.6,1.));before=hr(x,30)
    q=spectral_quality(x,30)
    assert q['flagged'] and q['ambiguous']
    assert q['peak_ratio']<1.2
    assert hr(x,30)==before==q['dominant_bpm']


def test_isolated_tone_does_not_count_zero_padding_bins_as_competitors():
    q=spectral_quality(tones((1.4,1.)),30)
    assert q['valid'] and not q['flagged']
    assert abs(q['dominant_bpm']-84)<1


def test_dominant_tone_with_small_competitor_not_ambiguous():
    q=spectral_quality(tones((1.,1.),(1.6,.2)),30)
    assert not q['ambiguous'] and q['peak_ratio']>10


def test_band_edge_is_flagged_separately():
    q=spectral_quality(tones((.71,1.)),30)
    assert q['flagged'] and q['near_band_edge']


@pytest.mark.parametrize('x',[np.zeros(900),np.full(900,np.nan),np.ones(20),np.ones((30,30))])
def test_unusable_signals_are_invalid_not_confident(x):
    q=spectral_quality(x,30)
    assert not q['valid'] and q['flagged'] and q['status']=='invalid_signal'


def test_invalid_settings_rejected():
    with pytest.raises(ValueError):spectral_quality(tones((1.,1.)),5)
    with pytest.raises(ValueError):spectral_quality(tones((1.,1.)),30,ratio_threshold=1)
