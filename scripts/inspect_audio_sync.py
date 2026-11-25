from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
fs,a=wavfile.read('data/private/audio_sync.wav');a=a.astype(float)
f,t,z=signal.stft(a,fs,nperseg=512,noverlap=448);p=np.abs(z)**2
fig,axs=plt.subplots(2,1,figsize=(12,7),layout='constrained')
for ax,(start,end) in zip(axs,[(3,10),(126,134)]):
 m=(t>=start)&(t<=end);ax.pcolormesh(t[m],f,10*np.log10(p[:,m]+1),shading='auto',vmin=15,vmax=75);ax.set(ylim=(500,7000),xlabel='Video time (s)',ylabel='Frequency (Hz)')
 # Most tonally concentrated STFT peaks, allow 0.12s minimum distance
 band=(f>=1500)&(f<=6500);q=p[band][:,m];score=q.max(axis=0)/(q.sum(axis=0)+1e-12);tt=t[m]
 peaks,_=signal.find_peaks(score,distance=30,prominence=.10)
 order=sorted(peaks,key=lambda k:score[k],reverse=True)[:10]
 print(start,[(round(float(tt[k]),3),round(float(f[band][q[:,k].argmax()]),1),round(float(score[k]),3)) for k in order])
fig.savefig('results/private/audio_sync_spectrogram.png',dpi=160)
