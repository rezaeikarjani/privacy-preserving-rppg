"""Run from repository root: python -m src.run --config configs/synthetic.json"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .video import FaceTracker,patch_rgb,transform
from .signals import classical,clean,hr,correlation,reference_band_ratio,spectrum


def run(config, tracker_factory=None):
    cfg=json.loads(Path(config).read_text())
    output=Path(cfg['output']); output.mkdir(parents=True,exist_ok=True)
    source=Path(cfg['input'])
    if not source.exists():
        raise FileNotFoundError(f"Missing input {source}; see docs/RECORDING.md")
    cap=cv2.VideoCapture(str(source)); fps=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not np.isfinite(fps) or fps<=2*cfg['band_hz'][1]:
        raise ValueError("Cannot read video or unsuitable frame rate")
    sidecar=cfg.get('tracking')=='sidecar'
    if sidecar and not cfg.get('synthetic'):
        raise ValueError("Sidecar fixture mode reserved for synthetic tests")
    boxes=pd.read_csv(cfg['boxes']).values if sidecar else None
    tracker=None if sidecar else (tracker_factory or FaceTracker)()
    kinds=['original','blur','pixelate','mask']
    rgb={k:[] for k in kinds}; writers={}; tracks=[]
    start=time.perf_counter(); index=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            if sidecar:
                box=boxes[index]; detected=False
            else:
                box,detected=tracker.update(frame,index)
            tracks.append([index,*box,int(detected)])
            for k in kinds:
                modified=transform(frame,box,k,cfg['blur_sigma_fraction'],cfg['pixel_grid'])
                rgb[k].append(patch_rgb(modified,box))
                if cfg.get('save_videos') and k!='original':
                    if k not in writers:
                        writers[k]=cv2.VideoWriter(str(output/(k+'.avi')),cv2.VideoWriter_fourcc(*'FFV1'),fps,(frame.shape[1],frame.shape[0]))
                        if not writers[k].isOpened(): raise RuntimeError("FFV1 video writer failed")
                    writers[k].write(modified)
            index+=1
    finally:
        cap.release()
        for writer in writers.values(): writer.release()
    if index/fps<12: raise ValueError("Need at least 12 seconds; record 45–60 seconds")
    if sidecar and len(boxes)!=index: raise ValueError("Sidecar length mismatch")
    # Re-read saved lossless transformed videos: verifies output decoding as well as in-memory transforms.
    if cfg.get('save_videos'):
        for k in kinds[1:]:
            decoded=[]; video=cv2.VideoCapture(str(output/(k+'.avi')))
            for row in tracks:
                ok,frame=video.read()
                if not ok: raise RuntimeError("Truncated saved video")
                decoded.append(patch_rgb(frame,row[1:5]))
            video.release()
            if not np.allclose(decoded,rgb[k],atol=0): raise RuntimeError("Lossless output differs")
            rgb[k]=decoded
    pd.DataFrame(tracks,columns=['frame','x','y','w','h','detected']).to_csv(output/'tracking.csv',index=False)
    arrays={k:np.asarray(v) for k,v in rgb.items()}
    np.savez_compressed(output/'rgb_traces.npz',fps=fps,**arrays)
    gt=pd.read_csv(cfg['ground_truth']) if cfg.get('ground_truth') else None
    if gt is not None:
        if not {'time_s','hr_bpm'}.issubset(gt): raise ValueError("GT requires time_s,hr_bpm")
        if not np.isfinite(gt[['time_s','hr_bpm']]).all().all() or not np.all(np.diff(gt.time_s)>0):
            raise ValueError("Invalid ground-truth timestamps or values")
        if gt.time_s.iloc[0]>0 or gt.time_s.iloc[-1]<(index-1)/fps:
            raise ValueError("Ground truth must cover the full video with aligned timestamps")
    trim=int(2*fps); rows=[]; window_rows=[]; signals={}
    for method in cfg['methods']:
        wave={k:clean(classical(arrays[k],fps,method,cfg['backend']),fps,cfg['band_hz'])[trim:-trim] for k in kinds}
        signals.update({method+'_'+k:v for k,v in wave.items()})
        reference=hr(wave['original'],fps,cfg['band_hz'])
        for k,bvp in wave.items():
            estimate=hr(bvp,fps,cfg['band_hz'])
            errors=[]
            width=int(cfg['window_seconds']*fps); step=int(cfg['step_seconds']*fps)
            for a in range(0,len(bvp)-width+1,step):
                ts=(a+trim)/fps; te=ts+width/fps
                value=hr(bvp[a:a+width],fps,cfg['band_hz'])
                target=float(gt.loc[(gt.time_s>=ts)&(gt.time_s<te),'hr_bpm'].mean()) if gt is not None else float('nan')
                if np.isfinite(target) and np.isfinite(value): errors.append(abs(value-target))
                window_rows.append(dict(method=method,variant=k,start_s=ts,end_s=te,hr_bpm=value,gt_hr_bpm=target,abs_error_bpm=abs(value-target)))
            rows.append(dict(data_kind=cfg.get('data_kind', 'synthetic_fixture' if cfg.get('synthetic') else 'consented_video'),backend=cfg['backend'],method=method,variant=k,hr_bpm=estimate,abs_hr_change_bpm=abs(estimate-reference),pearson_original=correlation(bvp,wave['original']),reference_band_ratio_db=reference_band_ratio(bvp,fps,reference,cfg['band_hz']),window_hr_mae_bpm=np.mean(errors) if errors else float('nan'),valid_hr_windows=len(errors) if gt is not None else sum(np.isfinite(r['hr_bpm']) for r in window_rows if r['method']==method and r['variant']==k)))
        fig,axes=plt.subplots(2,1,figsize=(9,6),layout='constrained')
        for k,bvp in wave.items():
            axes[0].plot(np.arange(len(bvp))/fps+2,bvp,label=k,alpha=.75)
            f,power=spectrum(bvp,fps); axes[1].plot(f*60,power,label=k)
        axes[0].set(xlim=(2,10),xlabel='Time (s)',ylabel='Standardized BVP',title=f'{method} | '+('SYNTHETIC ENGINEERING TEST' if cfg.get('synthetic') else 'Single consented video'))
        axes[1].set(xlim=(42,180),xlabel='Frequency (BPM)',ylabel='Welch PSD')
        axes[0].legend(ncol=4); axes[1].legend(ncol=4)
        fig.savefig(output/(method+'_waveform_psd.png'),dpi=160); plt.close(fig)
    pd.DataFrame(rows).to_csv(output/'results.csv',index=False)
    pd.DataFrame(window_rows).to_csv(output/'windows.csv',index=False)
    np.savez_compressed(output/'bvp.npz',fps=fps,**signals)
    manifest={'config':cfg,'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'frames':index,'fps':fps,'duration_s':index/fps,'elapsed_s':time.perf_counter()-start,'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,'opencv':cv2.__version__,'tracking_mode':cfg['tracking'],'face_detection_verified_on_real_video':False if sidecar else True,'identity_evaluation':'NOT RUN; no recognition weights or identity cohort','evaluation_design':'paired original-geometry ROI evaluation; no detector rerun on transformed faces','codec':'FFV1 lossless; audio discarded','trim_seconds':2,'ground_truth':'injected synthetic HR' if cfg.get('synthetic') else ('aligned reference HR' if gt is not None else 'unavailable: consistency only')}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved {output}; runtime {manifest['elapsed_s']:.2f}s")
    return rows

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True)
    run(parser.parse_args().config)
