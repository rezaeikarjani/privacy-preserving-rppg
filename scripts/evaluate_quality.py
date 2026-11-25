"""Annotate cached BVP windows; no video upload, watch-label input or HR correction."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.quality import spectral_quality
from src.signals import hr


def run(directory, config='configs/quality.json',source_offset_s=0):
    directory=Path(directory);cfg=json.loads(Path(config).read_text())
    if cfg['ratio_threshold'] not in cfg['sensitivity_thresholds']:
        raise ValueError('Primary threshold must be in sensitivity thresholds')
    waves=np.load(directory/'bvp.npz');fps=float(waves['fps']);windows=pd.read_csv(directory/'windows.csv')
    manifest=json.loads((directory/'manifest.json').read_text());band=manifest['config']['band_hz']
    trim=manifest['trim_seconds'];output=[]
    inputs=[directory/'bvp.npz',directory/'windows.csv',directory/'results.csv']
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    for _,row in windows.iterrows():
        a=round((row.start_s-trim)*fps);b=round((row.end_s-trim)*fps)
        wave=waves[row.method+'_'+row.variant][a:b]
        baseline=hr(wave,fps,band)
        if not np.isclose(baseline,row.hr_bpm,equal_nan=True):
            raise ValueError('Cached data do not reproduce stored baseline HR')
        for threshold in cfg['sensitivity_thresholds']:
            q=spectral_quality(wave,fps,band,threshold,cfg['min_separation_hz'],cfg['edge_margin_hz'])
            output.append(dict(method=row.method,variant=row.variant,
                               video_start_s=row.start_s+source_offset_s,
                               video_end_s=row.end_s+source_offset_s,
                               ratio_threshold=threshold,baseline_hr_bpm=row.hr_bpm,**q))
    table=pd.DataFrame(output);table.to_csv(directory/'quality_windows.csv',index=False)
    summary=table.groupby(['ratio_threshold','method','variant'],sort=False).agg(
        windows=('flagged','size'),valid_windows=('valid','sum'),
        flagged_windows=('flagged','sum'),ambiguous_windows=('ambiguous','sum'),
        band_edge_windows=('near_band_edge','sum')).reset_index()
    summary.to_csv(directory/'quality_summary.csv',index=False)
    # Keep all original estimates visible, with warning markers overlaid.
    selected=table[table.ratio_threshold==cfg['ratio_threshold']]
    fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained',sharex=True)
    for ax,method in zip(axes,['POS','CHROM']):
        for variant,color in zip(['original','blur','pixelate'],['tab:blue','tab:orange','tab:green']):
            s=selected[(selected.method==method)&(selected.variant==variant)]
            t=(s.video_start_s+s.video_end_s)/2
            ax.plot(t,s.baseline_hr_bpm,'o-',color=color,label=variant)
            flag=s.flagged
            ax.scatter(t[flag],s.baseline_hr_bpm[flag],marker='x',s=130,linewidths=2,color=color)
        ax.set(ylabel='Baseline HR (BPM)',title=method+' | x = quality warning (estimate unchanged)');ax.legend(ncol=3)
    axes[-1].set_xlabel('Source-video window centre (s)')
    fig.suptitle('Exploratory quality flags; not validated confidence or accuracy')
    fig.savefig(directory/'quality_flags.png',dpi=150);plt.close(fig)
    after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    assert before==after,'Baseline files changed'
    (directory/'quality_manifest.json').write_text(json.dumps(dict(
        quality_config=cfg,baseline_sha256=before,baseline_unchanged=True,
        reference_hr_used=False,source_offset_s=source_offset_s,
        spectrum='same Welch parameters as baseline; 10-second segments, nfft4096',
        limitation='Post-hoc exploratory flag; no rejection/correction, false-positive/negative performance unvalidated.'),indent=2))
    print(summary[summary.ratio_threshold==cfg['ratio_threshold']].to_string(index=False))
    print('\nFlagged non-mask windows at primary threshold:')
    print(selected[(selected.variant!='mask')&selected.flagged][['method','variant','video_start_s','video_end_s','baseline_hr_bpm','peak_ratio','status']].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--config',default='configs/quality.json');p.add_argument('--source-offset-s',type=float,default=0)
    a=p.parse_args();run(a.directory,a.config,a.source_offset_s)
