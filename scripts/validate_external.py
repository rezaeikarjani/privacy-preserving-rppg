"""Frozen-method evaluation of an official UBFC Dataset2 recording.
Primary endpoint: window HR error against contact-PPG spectral HR.
Does not estimate clinical accuracy or optimize thresholds on external results.
"""
import argparse,hashlib,json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.reference import load_ubfc_reference,sample_reference,time_weighted_mean
from src.signals import clean,hr,correlation
from src.quality import spectral_quality
from src.run import run


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def evaluate(folder,output,extract=True,target_region=None,detector="haar"):
    folder=Path(folder);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(Path('configs/external_validation_protocol.json').read_text())
    for path,h in protocol['frozen_code_sha256'].items():
        if digest(path)!=h:raise ValueError('Frozen code/settings changed: '+path)
    t,ppg,device_hr=load_ubfc_reference(folder/'ground_truth.txt')
    video=folder/'vid.avi';c=cv2.VideoCapture(str(video));fps=c.get(cv2.CAP_PROP_FPS);n=int(c.get(cv2.CAP_PROP_FRAME_COUNT));c.release()
    if not np.isfinite(fps) or fps<=6:raise ValueError('Invalid video FPS')
    # No result-driven lag fitting. Dataset-relative first timestamps are aligned.
    # Reject a substantial duration mismatch rather than stretching signals to match.
    if abs((n-1)/fps-t[-1])>2:raise ValueError('Video/reference duration mismatch exceeds 2 seconds')
    cfg=dict(data_kind='public_research_dataset',input=str(video),output=str(out),backend='numpy',methods=protocol['methods'],tracking='haar_lk',synthetic=False,
             band_hz=protocol['band_hz'],window_seconds=protocol['window_seconds'],step_seconds=protocol['step_seconds'],
             blur_sigma_fraction=.1,pixel_grid=8,save_videos=False)
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    if detector=='yunet' and target_region is not None:raise ValueError('Do not mix automatic YuNet and manual target region')
    if detector=='yunet':
        cfg['tracking']='yunet_per_frame_automatic'
        cfg['detector_model']='vendor/opencv_models/yunet.onnx'
        (out/'config.json').write_text(json.dumps(cfg,indent=2))
    if target_region is not None:
        cfg['tracking']='haar_lk_manual_target_region'
        cfg['target_region']=list(target_region)
        (out/'config.json').write_text(json.dumps(cfg,indent=2))
    if extract:
        cv2.setNumThreads(1)
        if detector=='yunet':
            from src.face_models import YuNetTracker
            run(out/'config.json',tracker_factory=YuNetTracker)
        elif target_region is None:
            run(out/'config.json')
        else:
            from src.tracking_policies import TargetRegionTracker
            run(out/'config.json',tracker_factory=lambda:TargetRegionTracker(target_region))
    # Correct only the source-kind metadata on extraction reuse, never numeric results.
    baseline_results=pd.read_csv(out/'results.csv')
    baseline_results['data_kind']='public_research_dataset'
    baseline_results.to_csv(out/'results.csv',index=False)
    bm=json.loads((out/'manifest.json').read_text());bm['config'].update(data_kind='public_research_dataset');bm['source_kind']='public_research_dataset';(out/'manifest.json').write_text(json.dumps(bm,indent=2))
    waves=np.load(out/'bvp.npz');windows=pd.read_csv(out/'windows.csv');fps=float(waves['fps']);qc=json.loads(Path('configs/quality.json').read_text())
    trim=int(2*fps);ref_times=np.arange(n)/fps
    # Reference may end slightly before the video: interpolate only covered times.
    valid_times=ref_times[(ref_times>=t[0])&(ref_times<=t[-1])]
    ref_raw=sample_reference(t,ppg,valid_times);ref_clean=clean(ref_raw,fps,protocol['band_hz'])
    rows=[]
    for _,w in windows.iterrows():
        start,end=float(w.start_s),float(w.end_s);a=round(start*fps)-trim;b=round(end*fps)-trim
        x=waves[w.method+'_'+w.variant][a:b]
        if not np.isclose(hr(x,fps,protocol['band_hz']),w.hr_bpm,equal_nan=True):raise ValueError('Window indexing does not reproduce baseline HR')
        q=spectral_quality(x,fps,protocol['band_hz'],qc['ratio_threshold'],qc['min_separation_hz'],qc['edge_margin_hz'])
        rt=np.arange(round(start*fps),round(end*fps))/fps
        try:
            sample_reference(t,ppg,rt) # Check reference coverage/gaps within window.
            aa=round(start*fps);bb=round(end*fps)
            r=ref_clean[aa:bb]
            if len(r)!=len(x):raise ValueError('Reference window length mismatch')
            ref_hr=hr(r,fps,protocol['band_hz']);device=time_weighted_mean(t,device_hr,start,end)
            ref_quality=spectral_quality(r,fps,protocol['band_hz'],qc['ratio_threshold'],qc['min_separation_hz'],qc['edge_margin_hz'])
            ref_status='covered'
        except ValueError as exc:
            ref_hr=device=float('nan');r=None;ref_quality={'flagged':True,'peak_ratio':float('nan')};ref_status=str(exc)
        estimate=float(w.hr_bpm);error=abs(estimate-ref_hr)
        rows.append(dict(method=w.method,variant=w.variant,start_s=start,end_s=end,hr_bpm=estimate,
                         contact_ppg_hr_bpm=ref_hr,contact_device_hr_bpm=device,abs_error_bpm=error,
                         device_disagreement_bpm=abs(device-ref_hr),reference_status=ref_status,contact_reference_quality_warning=ref_quality['flagged'],contact_reference_peak_ratio=ref_quality['peak_ratio'],
                         zero_lag_pearson_contact=correlation(x,r) if r is not None else float('nan'),
                         error_over_5_bpm=bool(error>5) if np.isfinite(error) else None,**q))
    table=pd.DataFrame(rows);table.to_csv(out/'reference_windows.csv',index=False)
    summaries=[];event=float(protocol.get('error_event_threshold_bpm',5))
    for (method,variant),s in table.groupby(['method','variant'],sort=False):
        finite=s[np.isfinite(s.abs_error_bpm)];retained=finite[~finite.flagged]
        truth=finite.abs_error_bpm>event;warn=finite.flagged
        tp=int((truth&warn).sum());fp=int((~truth&warn).sum());fn=int((truth&~warn).sum());tn=int((~truth&~warn).sum())
        summaries.append(dict(method=method,variant=variant,total_windows=len(s),evaluable_windows=len(finite),
                             invalid_estimate_windows=int((~s.valid).sum()),flagged_windows=int(s.flagged.sum()),
                             hr_mae_bpm=finite.abs_error_bpm.mean(),hr_rmse_bpm=np.sqrt(np.mean(finite.abs_error_bpm**2)),
                             retained_windows=len(retained),retained_mae_bpm=retained.abs_error_bpm.mean(),
                             retained_fraction=len(retained)/len(finite) if len(finite) else float('nan'),
                             tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else float('nan'),recall=tp/(tp+fn) if tp+fn else float('nan')))
    summary=pd.DataFrame(summaries);summary.to_csv(out/'reference_summary.csv',index=False)
    fig,axes=plt.subplots(2,1,figsize=(10,6),layout='constrained',sharex=True)
    for ax,method in zip(axes,protocol['methods']):
        for variant in ['original','blur','pixelate']:
            s=table[(table.method==method)&(table.variant==variant)];centre=(s.start_s+s.end_s)/2
            ax.plot(centre,s.hr_bpm,'o-',label=variant)
            ax.scatter(centre[s.flagged],s.hr_bpm[s.flagged],marker='x',s=100,color='red')
        s=table[(table.method==method)&(table.variant=='original')]
        ax.plot((s.start_s+s.end_s)/2,s.contact_ppg_hr_bpm,'ks--',label='contact-PPG spectral HR')
        ax.set(title=method+' | red x = fixed quality warning',ylabel='HR (BPM)');ax.legend(ncol=2)
    axes[-1].set_xlabel('Window centre (seconds)');fig.suptitle('UBFC '+folder.name+': independent recording, not population/clinical validation')
    fig.savefig(out/'reference_comparison.png',dpi=150);plt.close(fig)
    (out/'validation_manifest.json').write_text(json.dumps(dict(protocol=protocol,target_region=target_region,detector=detector,evaluation_scope='automatic YuNet exploratory detector variant' if detector=='yunet' else ('ROI-assisted fallback after strict baseline multiple-face failure' if target_region is not None else 'strict automatic baseline'),video_sha256=digest(video),reference_sha256=digest(folder/'ground_truth.txt'),
        frame_count=n,video_fps=fps,reference_max_gap_s=float(np.max(np.diff(t))),reference_source_samples=int(np.loadtxt(folder/'ground_truth.txt').shape[1]),reference_samples=len(t),exact_duplicate_reference_rows_removed=int(np.loadtxt(folder/'ground_truth.txt').shape[1]-len(t)),reference_median_interval_s=float(np.median(np.diff(t))),
        reference_duration_s=float(t[-1]),video_last_frame_time_s=(n-1)/fps,
        alignment='Dataset-relative t=0 aligned; no measured lag correction or drift stretching. Original metadata FPS retained.',
        uncertainty='Different sensor sites and timing/physiology affect waveform phase; zero-lag waveform correlation is descriptive only.',
        contact_reference='PPG interpolated on uniform video timeline, same fixed clean/Welch extraction; device HR secondary.',
        interpretation='Small independent-recording test with overlapping windows, not clinical/population accuracy or a validated quality classifier.'),indent=2))
    print(summary.to_string(index=False))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset-dir',default='data/public/ubfc/subject1');p.add_argument('--output',default='results/private/external_ubfc_subject1');p.add_argument('--reuse-extraction',action='store_true');p.add_argument('--target-region',nargs=4,type=int);p.add_argument('--detector',choices=['haar','yunet'],default='haar');a=p.parse_args()
    evaluate(a.dataset_dir,a.output,not a.reuse_extraction,a.target_region,a.detector)
