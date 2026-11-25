"""Paired transform-strength experiment using original cached tracking geometry.
No watch labels, threshold fitting, saved facial previews, or identity claims.
"""
import argparse,hashlib,json,time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.video import patch_rgb,transform
from src.signals import classical,clean,hr,correlation
from src.quality import spectral_quality


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(baseline,output,config):
    base=Path(baseline);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(config).read_text());m=json.loads((base/'manifest.json').read_text())
    q=json.loads(Path(cfg['quality_config']).read_text());old=np.load(base/'rgb_traces.npz');fps=float(old['fps'])
    track=pd.read_csv(base/'tracking.csv');oldwaves=np.load(base/'bvp.npz')
    source=Path(m['config']['input']);source_hash=digest(source)
    if source_hash!=m['input_sha256']:raise ValueError('Input differs from baseline')
    hashes={n:digest(base/n) for n in ['results.csv','windows.csv','bvp.npz','tracking.csv','rgb_traces.npz']}
    variants={};traces={'original':old['original']};new=[]
    for fraction in cfg['blur_sigma_fractions']:
        key=f'blur_{fraction:g}';variants[key]=dict(family='blur',setting=fraction)
        if fraction==m['config']['blur_sigma_fraction']:traces[key]=old['blur']
        else:traces[key]=[];new.append(key)
    for grid in cfg['pixel_grids']:
        key=f'pixel_{grid}';variants[key]=dict(family='pixelate',setting=grid)
        if grid==m['config']['pixel_grid']:traces[key]=old['pixelate']
        else:traces[key]=[];new.append(key)
    cap=cv2.VideoCapture(str(source));cv2.setNumThreads(1);started=time.perf_counter()
    try:
        for index,row in track.iterrows():
            ok,frame=cap.read()
            if not ok:raise ValueError('Video truncated relative to tracking')
            box=row[['x','y','w','h']].values
            if not np.allclose(patch_rgb(frame,box),old['original'][index],rtol=0,atol=1e-10):
                raise ValueError('Decoded RGB differs from original baseline')
            for key in new:
                v=variants[key]
                f=transform(frame,box,v['family'],sigma_fraction=v['setting'] if v['family']=='blur' else .1,
                            grid=int(v['setting']) if v['family']=='pixelate' else 8)
                traces[key].append(patch_rgb(f,box))
            if index%450==0:print(f'Processed {index}/{len(track)} frames',flush=True)
        ok,_=cap.read()
        if ok:raise ValueError('Video longer than tracking')
    finally:cap.release()
    traces={k:np.asarray(v) for k,v in traces.items()};np.savez_compressed(out/'rgb_traces.npz',fps=fps,**traces)
    trim=round(m['trim_seconds']*fps);width=round(cfg['window_seconds']*fps);step=round(cfg['step_seconds']*fps)
    rows=[];windows=[];waves={}
    for method in cfg['methods']:
        ref=clean(classical(traces['original'],fps,method),fps,cfg['band_hz'])[trim:-trim]
        assert np.allclose(ref,oldwaves[method+'_original'])
        for key,variant in variants.items():
            bvp=clean(classical(traces[key],fps,method),fps,cfg['band_hz'])[trim:-trim];waves[method+'_'+key]=bvp
            differences=[];flagged=[]
            for start in range(0,len(bvp)-width+1,step):
                s=bvp[start:start+width];r=ref[start:start+width]
                value=hr(s,fps,cfg['band_hz']);reference=hr(r,fps,cfg['band_hz'])
                qual=spectral_quality(s,fps,cfg['band_hz'],q['ratio_threshold'],q['min_separation_hz'],q['edge_margin_hz'])
                differences.append(abs(value-reference));flagged.append(qual['flagged'])
                windows.append(dict(method=method,variant=key,crop_start_s=start/fps+trim/fps,crop_end_s=(start+width+trim)/fps,
                                    hr_bpm=value,original_hr_bpm=reference,abs_hr_change_bpm=abs(value-reference),**qual))
            rows.append(dict(method=method,variant=key,**variant,hr_bpm=hr(bvp,fps,cfg['band_hz']),
                             whole_abs_hr_change_bpm=abs(hr(bvp,fps,cfg['band_hz'])-hr(ref,fps,cfg['band_hz'])),
                             pearson_original=correlation(bvp,ref),mean_window_abs_hr_change_bpm=np.mean(differences),
                             max_window_abs_hr_change_bpm=np.max(differences),flagged_windows=sum(flagged),windows=len(flagged)))
    summary=pd.DataFrame(rows);summary.to_csv(out/'summary.csv',index=False);pd.DataFrame(windows).to_csv(out/'windows.csv',index=False)
    np.savez_compressed(out/'bvp.npz',fps=fps,**waves)
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for col,family in enumerate(['blur','pixelate']):
        for method in cfg['methods']:
            s=summary[(summary.family==family)&(summary.method==method)]
            x=np.arange(len(s));labels=[str(v) for v in s.setting]
            axs[0,col].plot(x,s.pearson_original,'o-',label=method)
            axs[1,col].plot(x,s.mean_window_abs_hr_change_bpm,'o-',label=method)
            for ax in axs[:,col]:ax.set_xticks(x,labels);ax.legend()
        axs[0,col].set(title=family+' | increasing spatial degradation →',ylabel='Whole-waveform correlation')
        axs[1,col].set(xlabel='Sigma / face size' if family=='blur' else 'Grid cells per side (fewer = coarser)',ylabel='Mean window HR change (BPM)')
    fig.suptitle('Paired preservation on one recording; no identity privacy measurement')
    fig.savefig(out/'strength_sweep.png',dpi=150);plt.close(fig)
    assert all(digest(base/n)==h for n,h in hashes.items())
    (out/'manifest.json').write_text(json.dumps(dict(config=cfg,quality_config=q,source_sha256=source_hash,
        baseline_sha256=hashes,baseline_unchanged=True,frames=len(track),fps=fps,elapsed_s=time.perf_counter()-started,
        geometry='cached original tracking; identical across strengths',reference='original rPPG only; no watch labels used',
        privacy_evaluation='NOT RUN; strength describes image processing parameters only',numpy=np.__version__,opencv=cv2.__version__),indent=2))
    print(summary.to_string(index=False),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',default='results/private/garmin_15_120');p.add_argument('--output',default='results/private/strength_sweep');p.add_argument('--config',default='configs/strength_sweep.json');a=p.parse_args()
    run(a.baseline,a.output,a.config)
