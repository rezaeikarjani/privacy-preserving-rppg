import cv2
import numpy as np
import pytest
from src.signals import classical,clean,hr,correlation
from src.video import transform,patch_rgb,FaceTracker

@pytest.mark.parametrize('method',['POS','CHROM'])
@pytest.mark.parametrize('bpm',[54,72,96,132])
def test_frequency_recovery(method,bpm):
    fps=30;t=np.arange(900)/fps;p=np.sin(2*np.pi*bpm/60*t)
    rgb=np.array([180,140,100])[None,:]*(1+.02*np.sin(2*np.pi*.15*t[:,None]))
    rgb=rgb+p[:,None]*np.array([1,3,.5])[None,:]
    bvp=clean(classical(rgb,fps,method),fps)[60:-60]
    assert abs(hr(bvp,fps)-bpm)<1


def test_constant_and_mask_have_no_hr():
    rgb=np.ones((900,3))*100
    assert np.isnan(hr(clean(classical(rgb,30),30),30))
    assert np.isnan(correlation(np.zeros(20),np.ones(20)))


def test_transform_boundary_and_channel_order():
    frame=np.zeros((100,100,3),np.uint8);frame[20:80,20:80]=[50,100,200]
    assert np.allclose(patch_rgb(frame,[20,20,60,60]),[200,100,50])
    out=transform(frame,[20,20,60,60],'mask')
    assert not out.any()
    assert np.array_equal(frame[:20],transform(frame,[20,20,60,60],'blur')[:20])


def test_invalid_and_short_inputs():
    with pytest.raises(ValueError): classical(np.zeros((10,4)),30)
    with pytest.raises(ValueError): clean(np.ones(30),30)
    with pytest.raises(ValueError): clean(np.ones(900),5)
    with pytest.raises(ValueError): classical(np.ones((900,3)),30,backend='silent-fallback')


def test_detector_rejects_blank():
    tracker=FaceTracker()
    with pytest.raises(ValueError,match='Face undetected'):
        tracker.update(np.zeros((240,320,3),np.uint8),0)


def test_lk_translation_with_controlled_detection():
    # Controlled box isolates tracker motion logic; does NOT validate real face detection.
    class Detector:
        def detectMultiScale(self,*args,**kwargs): return np.array([[40,40,100,100]])
    tracker=FaceTracker();tracker.detector=Detector()
    rng=np.random.default_rng(5)
    frame=rng.integers(0,255,(200,200,3),dtype=np.uint8)
    box,_=tracker.update(frame,0)
    moved=cv2.warpAffine(frame,np.float32([[1,0,2],[0,1,1]]),(200,200))
    box2,detected=tracker.update(moved,1)
    assert not detected
    assert np.allclose(box2[:2]-box[:2],[2,1],atol=1)
