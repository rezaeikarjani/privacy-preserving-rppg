"""Explicit manually assisted target-region policy, separate from strict baseline."""
import numpy as np
from .video import FaceTracker


def filter_target_region(faces,region):
    faces=np.asarray(faces,dtype=float).reshape(-1,4)
    x,y,w,h=region
    if w<=0 or h<=0:raise ValueError('Invalid target region')
    if len(faces)==0:return faces
    centres=faces[:,:2]+faces[:,2:]/2
    keep=(centres[:,0]>=x)&(centres[:,0]<x+w)&(centres[:,1]>=y)&(centres[:,1]<y+h)
    return faces[keep]


class TargetRegionTracker(FaceTracker):
    def __init__(self,region):
        super().__init__()
        raw=self.detector;self.region=tuple(region)
        class Detector:
            def detectMultiScale(inner,*args,**kwargs):
                return filter_target_region(raw.detectMultiScale(*args,**kwargs),self.region)
        self.detector=Detector()
