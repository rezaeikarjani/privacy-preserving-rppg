"""Exploratory comparison to sparse consumer-watch HR, NOT ground-truth accuracy."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path('results/private/garmin_15_120')
w=pd.read_csv('results/private/garmin_reference/watch_samples.csv')
w['video_time_s']=w.activity_time_s+5.556
windows=pd.read_csv(out/'windows.csv')
rows=[]
for _,r in windows.iterrows():
 start=r.start_s+15;end=r.end_s+15
 s=w.loc[(w.video_time_s>=start)&(w.video_time_s<end)]
 # Observed-sample mean only; does not fabricate missing per-second watch data.
 ref=s.hr_bpm.mean()
 rows.append(dict(method=r.method,variant=r.variant,video_start_s=start,video_end_s=end,rppg_hr_bpm=r.hr_bpm,watch_observed_sample_mean_bpm=ref,watch_samples=len(s),absolute_difference_bpm=abs(r.hr_bpm-ref)))
comp=pd.DataFrame(rows);comp.to_csv(out/'watch_window_comparison.csv',index=False)
summary=comp.groupby(['method','variant'],sort=False).agg(mean_absolute_difference_bpm=('absolute_difference_bpm','mean'),windows=('absolute_difference_bpm','count'),min_watch_samples=('watch_samples','min'))
summary.to_csv(out/'watch_comparison_summary.csv');print(summary.to_string())
fig,ax=plt.subplots(figsize=(10,4),layout='constrained');ax.scatter(w.video_time_s,w.hr_bpm,c='black',label='Garmin observed samples',zorder=5)
for method in ['POS','CHROM']:
 s=comp[(comp.method==method)&(comp.variant=='original')];ax.plot((s.video_start_s+s.video_end_s)/2,s.rppg_hr_bpm,'o-',label=method+' original')
ax.set(xlabel='Original video time (s)',ylabel='HR (BPM)',title='Exploratory agreement with sparse Garmin HR; not raw PPG validation');ax.legend();fig.savefig(out/'watch_comparison.png',dpi=160)
(out/'synchronization.json').write_text(json.dumps({'source':'audio spectral tone candidates; not directly verified button timestamps','start_tone_onset_s':5.556,'end_matching_tone_onset_s':130.068,'matching_tone_separation_s':124.512,'garmin_activity_duration_s':124.664,'duration_discrepancy_s':.152,'analysis_source_interval_s':[15,120],'resampling':'FFmpeg fps=30, scale width480; original variable/noninteger frame cadence','watch_reference':'observed sparse sample arithmetic mean per window; unequal sampling can bias mean','claim':'exploratory consumer-device agreement, not clinical MAE'},indent=2))
