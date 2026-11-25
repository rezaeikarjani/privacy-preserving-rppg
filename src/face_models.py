"""Local OpenCV CPU face models, no network or identification of real-world names."""
from pathlib import Path
import cv2
import numpy as np
from .video import bounds


def unit_vector(x):
    x=np.asarray(x,dtype=float).ravel()
    if not np.isfinite(x).all() or np.linalg.norm(x)<1e-12:raise ValueError('Invalid face embedding')
    return x/np.linalg.norm(x)


class YuNetTracker:
    """Single-person per-frame detector. No manual ROI or silent face selection."""
    def __init__(self,model='vendor/opencv_models/yunet.onnx',score_threshold=.9):
        if not Path(model).is_file():raise FileNotFoundError(model)
        self.detector=cv2.FaceDetectorYN_create(model,'',(320,320),score_threshold,.3,5000)
        self.last_face=None

    def detect_one(self,frame):
        self.detector.setInputSize((frame.shape[1],frame.shape[0]))
        _,faces=self.detector.detect(frame)
        if faces is None or len(faces)==0:raise ValueError('YuNet: no face detected')
        if len(faces)!=1:raise ValueError('YuNet: multiple faces detected')
        return faces[0]

    def update(self,frame,index):
        self.last_face=self.detect_one(frame)
        return np.asarray(bounds(self.last_face[:4],frame.shape),dtype=float),True
