"""Execute unmodified pyVHR CPU methods in isolation from optional video/GUI imports.
This is NOT a run of pyVHR's complete Pipeline. No replacement algorithms are injected.
"""
import sys,types,importlib.util,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from src.signals import clean,hr,correlation
base=Path('vendor/python-probe/pyVHR').resolve()
for name,path in [('pyVHR',base),('pyVHR.BVP',base/'BVP')]:
 pkg=types.ModuleType(name);pkg.__path__=[str(path)];sys.modules[name]=pkg
for name,file in [('pyVHR.BVP.utils',base/'BVP/utils.py'),('pyVHR.BVP.methods',base/'BVP/methods.py')]:
 spec=importlib.util.spec_from_file_location(name,file);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
methods=sys.modules['pyVHR.BVP.methods'];p=Path('results/private/garmin_15_120');rgb=np.load(p/'rgb_traces.npz');local=np.load(p/'bvp.npz');fps=float(rgb['fps']);rows=[];waves={}
for method in ['POS','CHROM']:
 for k in ['original','blur','pixelate']:
  fn=getattr(methods,'cpu_'+method);kw={'fps':fps} if method=='POS' else {}
  data=rgb[k].T[None].astype(np.float32)
  if method=='CHROM':
   # CHROM upstream function is a projection, not a full temporally normalized pipeline.
   # Apply documented, explicit global channel-mean normalization first.
   data=data/np.mean(data,axis=2,keepdims=True)-1
  wave=clean(np.asarray(fn(data,**kw)).reshape(-1),fps)[60:-60];waves[method+'_'+k]=wave
  rows.append(dict(method=method,variant=k,upstream_hr_bpm=hr(wave,fps),local_hr_bpm=hr(local[method+'_'+k],fps),pearson_local=correlation(wave,local[method+'_'+k])))
pd.DataFrame(rows).to_csv(p/'pyvhr_isolated_comparison.csv',index=False);np.savez_compressed(p/'pyvhr_isolated_bvp.npz',fps=fps,**waves)
(p/'pyvhr_isolated_manifest.json').write_text(json.dumps({'distribution':'pyVHR-cpu 1.0.2','methods_sha256':hashlib.sha256((base/'BVP/methods.py').read_bytes()).hexdigest(),'scope':'unmodified upstream CPU methods loaded with importlib; root video/GUI initializers bypassed','full_pipeline_ran':False,'CHROM_preprocessing':'global channel-mean RGB normalization before projection; not full original CHROM pipeline','postprocessing':'same local Butterworth, trim and Welch as independent baseline'},indent=2));print(pd.DataFrame(rows).to_string(index=False))
