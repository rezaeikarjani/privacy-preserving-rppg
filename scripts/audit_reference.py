"""Contact-reference audit: retain all diagnostic settings, never replace baseline truth."""
from pathlib import Path
import hashlib,json
import cv2,numpy as np,pandas as pd
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.reference import load_ubfc_reference,sample_reference,time_weighted_mean
from src.signals import clean,hr,spectrum


def peaks_summary(x,t,fps,polarity,prominence_fraction,min_distance_s):
    peaks,_=signal.find_peaks(polarity*x,distance=int(np.ceil(min_distance_s*fps)),prominence=prominence_fraction*np.std(x))
    pt=t[peaks];intervals=np.diff(pt)
    if len(intervals)<2:return dict(pulse_count=len(peaks),median_ibi_hr_bpm=float('nan'),span_count_hr_bpm=float('nan'),ibi_iqr_s=float('nan')),pt,intervals
    return dict(pulse_count=len(peaks),median_ibi_hr_bpm=float(60/np.median(intervals)),span_count_hr_bpm=float(60*len(intervals)/(pt[-1]-pt[0])),
                ibi_iqr_s=float(np.percentile(intervals,75)-np.percentile(intervals,25))),pt,intervals


def run():
    cfg=json.loads(Path('configs/reference_audit.json').read_text());out=Path('results/private/reference_audit');out.mkdir(parents=True,exist_ok=True)
    existing=[]
    for p in Path('results/private').glob('external_ubfc_subject*/*'):
        if p.name in ['reference_windows.csv','reference_summary.csv','bvp.npz','rgb_traces.npz','results.csv','windows.csv']:existing.append(p)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in existing}
    pulse_rows=[];spectral_rows=[];interval_rows=[];timing=[]
    for subject in cfg['subjects']:
        base=Path('data/public/ubfc')/subject;t,ppg,device=load_ubfc_reference(base/'ground_truth.txt')
        cap=cv2.VideoCapture(str(base/'vid.avi'));fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
        timeline=np.arange(n)/fps;timeline=timeline[timeline<=t[-1]];resampled=sample_reference(t,ppg,timeline)
        standard=clean(resampled,fps,cfg['primary_band_hz'])
        shape=signal.sosfiltfilt(signal.butter(3,cfg['pulse_shape_band_hz'],fs=fps,btype='bandpass',output='sos'),signal.detrend(resampled));shape=(shape-shape.mean())/shape.std()
        win=pd.read_csv('results/private/external_ubfc_'+subject+'_yunet/reference_windows.csv')
        win=win[(win.method=='POS')&(win.variant=='original')]
        timing.append(dict(subject=subject,reference_duration_s=t[-1],video_last_s=(n-1)/fps,delta_duration_s=(n-1)/fps-t[-1],
                           reference_samples=len(t),median_timestamp_step_s=np.median(np.diff(t)),max_timestamp_gap_s=np.max(np.diff(t)),fps=fps))
        fig,axs=plt.subplots(len(win),3,figsize=(13,3.5*len(win)),layout='constrained',squeeze=False)
        for window_idx,(_,w) in enumerate(win.iterrows()):
            a=round(w.start_s*fps);b=round(w.end_s*fps);ts=timeline[a:b];x=standard[a:b];broad=shape[a:b]
            device_mean=time_weighted_mean(t,device,w.start_s,w.end_s)
            f,power=spectrum(x,fps);where=(f>=.7)&(f<=3);peak_indices,_=signal.find_peaks(power);peak_indices=sorted([i for i in peak_indices if where[i]],key=lambda i:power[i],reverse=True)[:5]
            for rank,i in enumerate(peak_indices,1):spectral_rows.append(dict(subject=subject,window=window_idx,start_s=w.start_s,end_s=w.end_s,method='resampled_Welch',rank=rank,hr_bpm=f[i]*60,power=power[i]))
            native=(t>=w.start_s)&(t<w.end_s);native_t=t[native];native_x=signal.detrend(ppg[native]);freq=np.linspace(.7,3,2301)
            lp=signal.lombscargle(native_t-native_t[0],native_x,2*np.pi*freq,normalize=True)
            ind,_=signal.find_peaks(lp);ind=sorted(ind,key=lambda i:lp[i],reverse=True)[:5]
            for rank,i in enumerate(ind,1):spectral_rows.append(dict(subject=subject,window=window_idx,start_s=w.start_s,end_s=w.end_s,method='native_timestamp_LombScargle',rank=rank,hr_bpm=freq[i]*60,power=lp[i]))
            for name,values in [('primary_band',x),('shape_band',broad)]:
                for polarity in cfg['polarities']:
                    for prom in cfg['prominence_std_fractions']:
                        info,pt,ibi=peaks_summary(values,ts,fps,polarity,prom,cfg['minimum_pulse_distance_s'])
                        pulse_rows.append(dict(subject=subject,window=window_idx,start_s=w.start_s,end_s=w.end_s,filter=name,polarity=polarity,prominence_fraction=prom,
                            primary_spectral_hr_bpm=w.contact_ppg_hr_bpm,device_mean_hr_bpm=device_mean,**info))
                        for j,v in enumerate(ibi):interval_rows.append(dict(subject=subject,window=window_idx,filter=name,polarity=polarity,prominence_fraction=prom,pulse_time_s=pt[j+1],interval_s=v,instantaneous_bpm=60/v))
            # Display a preselected central 10s region, not one chosen for reference agreement.
            centre=(w.start_s+w.end_s)/2;show=(ts>=centre-5)&(ts<=centre+5)
            axs[window_idx,0].plot(ts[show],x[show],label='0.7-3 Hz');axs[window_idx,0].plot(ts[show],broad[show],alpha=.5,label='0.4-5 Hz')
            _,pt,_=peaks_summary(broad,ts,fps,1,.5,cfg['minimum_pulse_distance_s']);pt=pt[(pt>=centre-5)&(pt<=centre+5)]
            axs[window_idx,0].scatter(pt,np.interp(pt,ts,broad),marker='x',color='red',label='shape-band + peaks, prominence0.5 SD')
            axs[window_idx,0].set(title=f'Window {window_idx+1}: central waveform',xlabel='Time (s)',ylabel='Standardized contact PPG');axs[window_idx,0].legend(fontsize=7)
            axs[window_idx,1].plot(f*60,power,label='Welch');axs[window_idx,1].axvline(w.contact_ppg_hr_bpm,color='red',label='Baseline spectral HR');axs[window_idx,1].axvline(device_mean,color='black',linestyle='--',label='Device HR mean')
            axs[window_idx,1].set(xlim=(42,180),title='Competing contact-reference frequencies',xlabel='Frequency (BPM)',ylabel='PSD');axs[window_idx,1].legend(fontsize=7)
            axs[window_idx,2].plot(freq*60,lp);axs[window_idx,2].axvline(device_mean,color='black',linestyle='--');axs[window_idx,2].set(title='Raw native-time Lomb-Scargle',xlabel='Frequency (BPM)',ylabel='Normalized power')
        fig.suptitle(subject+': descriptive contact-reference audit, no ground-truth replacement');fig.savefig(out/(subject+'_audit.png'),dpi=140);plt.close(fig)
    pd.DataFrame(pulse_rows).to_csv(out/'pulse_estimates.csv',index=False);pd.DataFrame(spectral_rows).to_csv(out/'spectral_peaks.csv',index=False)
    pd.DataFrame(interval_rows).to_csv(out/'pulse_intervals.csv',index=False);pd.DataFrame(timing).to_csv(out/'timing.csv',index=False)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (out/'manifest.json').write_text(json.dumps(dict(config=cfg,baseline_file_hashes=hashes,baselines_unchanged=True,
        note='Pulse-count methods and native-time periodogram are post-hoc diagnostics, not independently validated references. Inverted peaks, dicrotic notches, missed pulses, filters and changing HR can affect results. No method chosen to match rPPG.'),indent=2))
    p=pd.DataFrame(pulse_rows);s=p[p.subject=='subject3']
    print(s[['window','filter','polarity','prominence_fraction','pulse_count','median_ibi_hr_bpm','span_count_hr_bpm','primary_spectral_hr_bpm','device_mean_hr_bpm']].to_string(index=False))
    print('\nNative-time top peaks:\n',pd.DataFrame(spectral_rows).query("subject=='subject3' and rank<=3").to_string(index=False))

if __name__=='__main__':run()
