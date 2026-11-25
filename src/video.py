"""Video transforms and simple frontal-face ROIs. No claim of biometric anonymity."""
import cv2
import numpy as np

# Relative to frontal face rectangle: approximate skin patches, no learned skin segmentation.
PATCHES = [(0.30,0.12,0.70,0.26), (0.12,0.52,0.34,0.72), (0.66,0.52,0.88,0.72)]


def bounds(box, shape):
    x,y,w,h = [int(round(v)) for v in box]
    x1,y1=max(0,x),max(0,y)
    x2,y2=min(shape[1],x+w),min(shape[0],y+h)
    if x2-x1<10 or y2-y1<10:
        raise ValueError("Invalid face rectangle")
    return x1,y1,x2-x1,y2-y1


def transform(frame, box, kind, sigma_fraction=.1, grid=8):
    out=frame.copy()
    x,y,w,h=bounds(box,frame.shape)
    crop=frame[y:y+h,x:x+w]
    if kind == "blur":
        crop=cv2.GaussianBlur(crop,(0,0),max(1,min(w,h)*sigma_fraction))
    elif kind == "pixelate":
        crop=cv2.resize(cv2.resize(crop,(grid,grid),interpolation=cv2.INTER_AREA),(w,h),interpolation=cv2.INTER_NEAREST)
    elif kind == "mask":
        crop=np.zeros_like(crop)
    elif kind != "original":
        raise ValueError("Unknown transform")
    out[y:y+h,x:x+w]=crop
    return out


def patch_rgb(frame, box):
    x,y,w,h=bounds(box,frame.shape)
    colors=[]
    for a,b,c,d in PATCHES:
        patch=frame[y+int(b*h):y+int(d*h),x+int(a*w):x+int(c*w)]
        colors.append(patch.mean(axis=(0,1))[::-1]) # OpenCV BGR -> RGB
    return np.mean(colors,axis=0) # Equal weight per anatomical patch.


class FaceTracker:
    """Haar detection every 15 frames; LK feature translation between detections.
    Reject prolonged loss; no interpolation over invisible faces.
    """
    def __init__(self):
        self.detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        if self.detector.empty():
            raise RuntimeError("Missing OpenCV Haar cascade")
        self.box=None
        self.prev=None
        self.points=None
        self.since_detection=0

    def update(self,frame,index):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        detected=False
        if index%15==0 or self.box is None:
            faces=self.detector.detectMultiScale(gray,1.1,5,minSize=(60,60))
            if len(faces)>1:
                raise ValueError("Multiple faces: use a single-participant video")
            if len(faces)==1:
                self.box=faces[0].astype(float)
                self.since_detection=0
                detected=True
        if not detected and self.box is not None:
            self.since_detection+=1
            if self.points is None or len(self.points)<5:
                raise ValueError(f"Insufficient tracking features at frame {index}")
            new,status,_=cv2.calcOpticalFlowPyrLK(self.prev,gray,self.points,None)
            if new is None or status.sum()<5:
                raise ValueError(f"Tracking lost at frame {index}")
            shift=np.median((new-self.points)[status.ravel()==1],axis=0).ravel()
            if np.linalg.norm(shift)>min(self.box[2:])*0.15:
                raise ValueError("Abrupt motion: reject recording")
            self.box[:2]+=shift
        if self.box is None or self.since_detection>30:
            raise ValueError(f"Face undetected/lost at frame {index}")
        self.box=np.array(bounds(self.box,frame.shape),dtype=float)
        x,y,w,h=self.box.astype(int)
        mask=np.zeros_like(gray)
        mask[y:y+h,x:x+w]=255
        self.points=cv2.goodFeaturesToTrack(gray,80,.01,5,mask=mask)
        self.prev=gray
        return self.box.copy(),detected
