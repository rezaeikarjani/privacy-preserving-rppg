"""Read-only diagnostics of the recorded baseline; no tuning against watch labels."""
from pathlib import Path
import cv2,json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.video import PATCHES
from src.signals import spectrum
p=Path('results/private/garmin_15_120');tracks=pd.read_csv(p/'tracking.csv');v=cv2.VideoCapture('data/private/analysis_15_120.avi')
fig,axes=plt.subplots(1,4,figsize=(12,6),layout='constrained')
for ax,t in zip(axes,[20,45,75,105]):
 frameidx=round((t-15)*30);v.set(cv2.CAP_PROP_POS_FRAMES,frameidx);ok,f=v.read();assert ok
 x,y,w,h=tracks.iloc[frameidx][['x','y','w','h']].astype(int)
 cv2.rectangle(f,(x,y),(x+w,y+h),(0,255,255),2)
 for name,(a,b,c,d) in zip(['forehead','cheek L','cheek R'],PATCHES):
  aa=(x+int(a*w),y+int(b*h));bb=(x+int(c*w),y+int(d*h));cv2.rectangle(f,aa,bb,(0,255,0),2);cv2.putText(f,name,aa,cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,0),1)
 ax.imshow(cv2.cvtColor(f,cv2.COLOR_BGR2RGB));ax.set_title(f'Video {t}s');ax.axis('off')
fig.savefig(p/'roi_review_PRIVATE.png',dpi=130);plt.close(fig);v.release()
bvp=np.load(p/'bvp.npz');fps=float(bvp['fps']);freqrows=[]
# BVP begins at source 17 s (15 s crop + 2 s edge trim).
a=int((32-17)*fps);n=30*30
fig,axs=plt.subplots(2,1,figsize=(10,6),layout='constrained')
for k in ['original','blur','pixelate']:
 s=bvp['POS_'+k][a:a+n];f,power=spectrum(s,fps);valid=(f>=.7)&(f<=3)
 peaks,_=find_peaks(power);peaks=[i for i in peaks if valid[i]];peaks=sorted(peaks,key=lambda i:power[i],reverse=True)[:4]
 for rank,i in enumerate(peaks,1):freqrows.append(dict(variant=k,rank=rank,bpm=f[i]*60,power=power[i]))
 axs[0].plot(np.arange(len(s))/fps+32,s,label=k,alpha=.7);axs[1].plot(f*60,power,label=k)
axs[0].set(xlabel='Source-video time (s)',ylabel='Standardized BVP',title='POS failure window: 32-62 seconds')
axs[1].set(xlim=(42,180),xlabel='Frequency (BPM)',ylabel='Welch PSD');axs[0].legend();axs[1].legend();fig.savefig(p/'failure_window.png',dpi=150);plt.close(fig)
pd.DataFrame(freqrows).to_csv(p/'failure_spectral_peaks.csv',index=False)
print(pd.DataFrame(freqrows).to_string(index=False))
# Geometry jump stats, association is diagnostic, not proof of causation.
t=tracks.copy();t['video_time_s']=t.frame/fps+15
for field in ['x','y','w','h']:t['delta_'+field]=t[field].diff()
geometry=[]
for start,end in [(17,47),(32,62),(47,77),(62,92),(77,107)]:
 w=t[(t.video_time_s>=start)&(t.video_time_s<end)]
 geometry.append(dict(start_s=start,end_s=end,max_box_step_px=float(w[['delta_x','delta_y','delta_w','delta_h']].abs().max().max()),std_width_px=float(w.w.std())))
pd.DataFrame(geometry).to_csv(p/'geometry_diagnostics.csv',index=False)
print(pd.DataFrame(geometry).to_string(index=False))
