"""Upstream POS comparison on exactly the same ROI RGB traces. No full-toolbox claim.
Requires a separately cloned rPPG-Toolbox and its imports. Failure is explicit.
"""
import argparse,sys,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from src.signals import classical,clean,hr,correlation
p=argparse.ArgumentParser();p.add_argument('--toolbox',required=True);p.add_argument('--traces',required=True);p.add_argument('--output',required=True);p.add_argument('--numpy2-compat',action='store_true')
a=p.parse_args(); repo=Path(a.toolbox).resolve()
if a.numpy2_compat and not hasattr(np, 'mat'):
    np.mat=np.asmatrix  # Explicit legacy NumPy alias; upstream source remains unchanged.
sys.path.insert(0,str(repo))
from unsupervised_methods.methods.POS_WANG import POS_WANG
traces=np.load(a.traces);fps=float(traces['fps']);rows=[]
for variant in ['original','blur','pixelate']:
    rgb=traces[variant]
    # Toolbox expects [frames,height,width,RGB]; traces are one pooled ROI pixel.
    upstream=clean(np.asarray(POS_WANG(rgb[:,None,None,:],fps)).reshape(-1),fps)[int(2*fps):-int(2*fps)]
    local=clean(classical(rgb,fps),fps)[int(2*fps):-int(2*fps)]
    rows.append(dict(variant=variant,toolbox_hr_bpm=hr(upstream,fps),numpy_hr_bpm=hr(local,fps),pearson=correlation(upstream,local)))
revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
Path(a.output).mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(Path(a.output)/'toolbox_comparison.csv',index=False)
(Path(a.output)/'toolbox_revision.json').write_text(json.dumps({'commit':revision,'numpy2_compat_alias':a.numpy2_compat,'note':'Upstream internal detrending and bandpass retained, then common local postfilter applied','scope':'upstream classical POS module on pooled RGB; not training or full dataset benchmark'},indent=2))
print(pd.DataFrame(rows).to_string(index=False))
