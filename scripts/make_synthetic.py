"""Non-human cartoon with injected 72-BPM colour modulation. Engineering fixture only."""
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd


def generate(root=Path("."), bpm=72, seconds=30, fps=30):
    root=Path(root); (root/"data").mkdir(exist_ok=True,parents=True)
    writer=cv2.VideoWriter(str(root/"data/synthetic.avi"),cv2.VideoWriter_fourcc(*"FFV1"),fps,(320,240))
    if not writer.isOpened():
        raise RuntimeError("FFV1 writer unavailable")
    rng=np.random.default_rng(42)
    boxes=[]; gt=[]
    yy,xx=np.mgrid[:240,:320]
    for i in range(seconds*fps):
        t=i/fps
        dx=int(3*np.sin(2*np.pi*.13*t)); dy=int(2*np.sin(2*np.pi*.09*t))
        x,y,w,h=80+dx,25+dy,160,195
        skin=((xx-(160+dx))/76)**2+((yy-(122+dy))/96)**2<=1
        pulse=np.sin(2*np.pi*bpm/60*t)+.15*np.sin(4*np.pi*bpm/60*t)
        lighting=1+.025*np.sin(2*np.pi*.18*t)
        frame=np.full((240,320,3),35.,dtype=float)
        texture=3*np.sin(xx/9)*np.cos(yy/11)
        # BGR pulse amplitudes deliberately unequal; not a skin/optics simulation.
        for ch,(base,amp) in enumerate(zip([105,145,185],[.6,2.5,1.0])):
            layer=(base+texture)*lighting+amp*pulse+rng.normal(0,.7,(240,320))
            frame[:,:,ch][skin]=layer[skin]
        frame=np.clip(frame,0,255).astype(np.uint8)
        cv2.circle(frame,(130+dx,99+dy),8,(30,30,30),-1)
        cv2.circle(frame,(190+dx,99+dy),8,(30,30,30),-1)
        cv2.ellipse(frame,(160+dx,163+dy),(23,9),0,0,180,(40,40,40),2)
        writer.write(frame)
        boxes.append([x,y,w,h]); gt.append([t,pulse,bpm])
    writer.release()
    pd.DataFrame(boxes,columns=['x','y','w','h']).to_csv(root/"data/synthetic_boxes.csv",index=False)
    pd.DataFrame(gt,columns=['time_s','ppg','hr_bpm']).to_csv(root/"data/synthetic_gt.csv",index=False)
    (root/"data/synthetic_provenance.json").write_text(json.dumps({'kind':'procedural cartoon, not human video','seed':42,'bpm':bpm,'fps':fps,'seconds':seconds},indent=2))

if __name__=='__main__':
    generate()
